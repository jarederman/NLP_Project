from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.a10_quality import QualityProblem, Data, _init_worker, fit_quality, fold_job, item_features, item_matrix, \
    item_rows, qnorm_items, regress
from src.a6_combine import qe_features
from src.a6_data import SYSTEMS
from src.a6_signals import fit_logistic_np


def toy(seed: int = 0, n_docs: int = 5):
    rng = np.random.default_rng(seed)
    docs, cmat, term, y = [], [], [], []
    for i in range(n_docs):
        n = int(rng.integers(8, 20))
        docs.append(SimpleNamespace(doc_id=f"d{i}", candidate_offsets=list(range(1, n + 1)), genre=f"g{i % 2}"))
        gold = rng.random(n) < 0.3
        y.append(gold.astype(float))
        rows = np.array([gold ^ (rng.random(n) < r) for r in (0.0, 0.1, 0.2, 0.4)])
        cmat.append(rows)
        term.append(rng.random(n) < 0.4)
    y, term = np.concatenate(y), np.concatenate(term)
    f1 = np.concatenate([2 * (c @ g) / np.maximum(c.sum(1) + g.sum(), 1)
                         for c, g in zip(cmat, np.split(y, np.cumsum([len(d.candidate_offsets) for d in docs])[:-1]))])
    x = np.column_stack([y + rng.normal(0, 0.7, len(y)), term + rng.normal(0, 1, len(y)), rng.normal(0, 1, len(y))])
    genres = sorted({d.genre for d in docs})
    fold = np.concatenate([[genres.index(d.genre)] * len(d.candidate_offsets) for d in docs])
    data = Data("toy", docs, x, term, cmat, f1, None, np.array([d.genre for d in docs]),
                rng.random(n_docs * len(SYSTEMS)), y, fold)
    return data


class A10Tests(unittest.TestCase):
    def test_features_match_amendment6(self):
        data = toy()
        p = 1 / (1 + np.exp(-data.x[:, 0]))
        preds = {(d.doc_id, s): {o for o, b in zip(d.candidate_offsets, c[k]) if b}
                 for d, c in zip(data.docs, data.cmat) for k, s in enumerate(SYSTEMS)}
        soft = qnorm_items(data, p)          # normalized; recover raw soft below
        sl_soft = []
        for d, c, start in zip(data.docs, data.cmat, np.cumsum([0] + [len(d.candidate_offsets) for d in data.docs[:-1]])):
            q = p[start:start + len(d.candidate_offsets)]
            sl_soft.extend(2 * (c @ q) / (c.sum(1) + q.sum()))
        ref = qe_features(data.docs, preds, p, data.term, np.array(sl_soft), soft, data.stability)
        ours = item_matrix(data, p, "S2", range(len(data.docs)))
        np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(item_matrix(data, p, "S2-ns", range(len(data.docs))), ref[:, :-1], atol=1e-12)

    def test_h_features(self):
        q = np.array([0.9, 0.1, 0.6])
        c = np.array([[1, 0, 1], [0, 0, 0]], bool)
        h = item_features(q, c, np.array([1, 0, 0], bool), "h")
        self.assertAlmostEqual(h[0, 0], 2 * 1.5 / (2 + 1.6))           # Soft F1
        self.assertAlmostEqual(h[0, 2], 2 / 4)                          # boundaries per word
        self.assertAlmostEqual(h[0, 3], 0.1 / 4)                        # missed mass per word
        self.assertAlmostEqual(h[1, 4], (0.9 + 0.1 + 0.6) / 3)          # mean |y - p|

    def test_torch_qnorm_matches_numpy(self):
        import torch
        data = toy(1)
        model = fit_logistic_np(data.x, data.y, 1e-3)
        z = (data.x - model["mu"]) / model["sd"]
        theta = np.append(model["w"], model["b"])
        idx = np.arange(len(data.docs))
        prob = QualityProblem(torch, z, data.y, data, idx)
        _, _, _, q = prob.losses(torch.tensor(theta, dtype=torch.float64))
        p = 1 / (1 + np.exp(-(z @ theta[:-1] + theta[-1])))
        np.testing.assert_allclose(q.detach().numpy().ravel(), qnorm_items(data, p, idx), atol=1e-10)
        keep = [0, 2, 3]
        sub = QualityProblem(torch, z, data.y, data, idx, keep)
        _, _, _, q3 = sub.losses(torch.tensor(theta, dtype=torch.float64))
        np.testing.assert_allclose(q3.detach().numpy().ravel(), qnorm_items(data, p, idx, keep), atol=1e-10)

    def test_bce_only_objective_stays_at_stack_solution(self):
        import torch
        data = toy(2, 8)
        model = fit_logistic_np(data.x, data.y, 1e-3)
        z = (data.x - model["mu"]) / model["sd"]
        theta = np.append(model["w"], model["b"])
        prob = QualityProblem(torch, z, data.y, data, np.arange(len(data.docs)))
        np.testing.assert_allclose(fit_quality(torch, prob, theta, (0.0, 1.0, 0.0)), theta, atol=1e-5)
        moved = fit_quality(torch, prob, theta, (1.0, 0.0, 0.0))
        lq = lambda t: float(prob.losses(torch.tensor(t, dtype=torch.float64))[0])
        self.assertLessEqual(lq(moved), lq(theta) + 1e-12)

    def test_item_rows_and_residual(self):
        self.assertEqual(item_rows([1, 3], [0, 2]).tolist(), [4, 6, 12, 14])
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        q = np.array([0.5, 0.5, 0.5, 0.5])
        f = q + 0.1
        np.testing.assert_allclose(regress("R", x, f, q, x[:2], q[:2]), [0.6, 0.6], atol=1e-9)


    def test_fold_jobs_cover_every_item_in_a_spawn_pool(self):
        import multiprocessing as mp
        data = toy(3, 9)
        data.genres = np.array([f"g{i % 3}" for i in range(len(data.docs))])
        for d, g in zip(data.docs, data.genres):
            d.genre = g
        data.fold = np.concatenate([[int(g[1])] * len(d.candidate_offsets) for d, g in zip(data.docs, data.genres)])
        regs = ["S2", "S2-ns", "A", "R"]
        with mp.get_context("spawn").Pool(2, initializer=_init_worker, initargs=(data, regs, 1)) as pool:
            results = pool.map(fold_job, range(3))
        n_items = len(data.docs) * len(SYSTEMS)
        for name in ["S1", *regs, "B-q", "B-qbr", "LOSO:A", "LOSO:B-qb"]:
            rows = np.concatenate([r["scores"][name][0] for r in results])
            self.assertEqual(sorted(rows.tolist()), list(range(n_items)), name)
        self.assertEqual(sum(len(r["p_out"]) for r in results), len(data.y))


if __name__ == "__main__":
    unittest.main()
