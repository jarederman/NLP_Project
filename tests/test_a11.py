from __future__ import annotations

import unittest

import numpy as np

from src.a11_analysis import LEVELS, TYPES2, candidates2, error_masks2, ranking_stats, soft_f1_end, top1_hit
from src.a8_supplement import NOPUNCT, PUNCT, RATES, TERMINAL, TYPES, DocCtx, candidates


def toy(seed=0, n=60):
    rng = np.random.default_rng(seed)
    gold = rng.random(n) < 0.25
    cats = rng.choice([TERMINAL, PUNCT, NOPUNCT], size=n, p=[0.5, 0.2, 0.3])
    par = gold & (rng.random(n) < 0.4)
    return gold, cats, par


class A11Tests(unittest.TestCase):
    def test_original_types_and_rates_reproduce_amendment8(self):
        gold, cats, par = toy()
        ctx = DocCtx("d", "g", gold, cats, {}, {}, np.zeros((4, len(gold)), bool), np.zeros((4, len(gold)), bool),
                     np.zeros((4, len(gold))), np.ones(2))
        old, t_old, r_old = candidates(ctx, 3)
        new, t_new, l_new = candidates2(gold, cats, par, 3)
        keep = (t_new < len(TYPES)) & (l_new < len(RATES))
        keep[0] = True
        np.testing.assert_array_equal(new[keep], old)
        np.testing.assert_array_equal(t_new[keep], t_old)
        np.testing.assert_array_equal(l_new[keep], r_old)

    def test_new_merge_types_respect_paragraph_breaks(self):
        gold, cats, par = toy(1)
        cand, t_idx, l_idx = candidates2(gold, cats, par, 0)
        for name, where in (("merge_other_par", par), ("merge_other_inpar", ~par)):
            rows = cand[t_idx == TYPES2.index(name)]
            removed = gold[None, :] & ~rows
            self.assertTrue(removed.any())
            self.assertFalse((removed & ~(where & (cats != TERMINAL))[None, :]).any())
        k1 = cand[(l_idx == LEVELS.index("k=1")) & (t_idx == TYPES2.index("merge_terminal"))]
        self.assertTrue(((gold[None, :] != k1).sum(1) == 1).all())

    def test_error_masks_partition_errors(self):
        gold, cats, par = toy(2)
        row = gold ^ (np.random.default_rng(5).random(len(gold)) < 0.2)
        m = error_masks2(row, gold, cats, par)
        total = sum(v.astype(int) for v in m.values())
        np.testing.assert_array_equal(total, (row != gold).astype(int))

    def test_ranking_stats_perfect_and_reversed(self):
        f1 = np.tile([0.9, 0.8, 0.7, 0.6], 6)
        genre = np.array(["a", "a", "b", "b", "c", "c"])
        idx = np.arange(6)
        np.testing.assert_allclose(ranking_stats(f1, f1, genre, idx), [1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(ranking_stats(-f1, f1, genre, idx)[1], -1.0)

    def test_top1_is_tie_aware(self):
        self.assertEqual(top1_hit(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.1, 0.2, 0.3, 0.9])), 0.25)
        self.assertEqual(top1_hit(np.array([0.1, 0.2, 0.9, 0.3]), np.array([0.9, 0.2, 0.9, 0.3])), 1.0)
        self.assertEqual(top1_hit(np.array([0.9, 0.2, 0.9, 0.3]), np.array([0.1, 0.2, 0.9, 0.3])), 0.5)

    def test_end_boundary_variant(self):
        q = np.array([0.02, 0.01, 0.03])
        rows = np.array([[0, 0, 0], [1, 0, 0]], bool)
        s = soft_f1_end(rows, q)
        self.assertAlmostEqual(s[0], 1.0)              # no split in a one-sentence document: perfect
        self.assertLess(s[1], s[0])
        q2 = np.array([0.9, 0.1, 0.8])
        s2 = soft_f1_end(np.array([[1, 0, 1], [1, 0, 0]], bool), q2)
        self.assertAlmostEqual(s2[0], 1.0)
        self.assertLess(s2[1], 1.0)


if __name__ == "__main__":
    unittest.main()
