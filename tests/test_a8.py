import unittest

import numpy as np

from src.a6_data import SYSTEMS
from src.a8_supplement import NOPUNCT, PUNCT, TERMINAL, TYPES, DocCtx, apply_op, candidates, consensus_refs, credit, \
    doc_error_pricing, doc_perturbation_stats, error_masks, fixed, gold_vs_systems, hard_f1, length_score, n_ops, \
    ols_slope, pair_agreement, perturb, resample_plan, score_matrix, sentence_lengths, soft_f1, top1_credit
from src.meta_eval import max_soft_f1
from src.scoring import soft_fbeta


def make_ctx(n=40, seed=0):
    rng = np.random.default_rng(seed)
    gold = np.zeros(n, dtype=bool)
    gold[[4, 9, 15, 22, 30, 37]] = True
    cats = np.full(n, NOPUNCT)
    cats[[4, 9, 15, 30]] = TERMINAL                      # terminal Gold boundaries
    cats[[2, 12, 26]] = TERMINAL                         # terminal non-Gold (abbreviations)
    cats[[6, 18, 33]] = PUNCT
    p = np.where(gold, 0.85, 0.05)
    p[[22, 37]] = 0.2                                    # evidence blind to the non-terminal Gold boundaries
    ev = {"E0": p, "stack": p, "E1": p, "E2": p}
    votes = np.array([gold.copy() for _ in SYSTEMS])
    votes[0, [22, 37]] = False                           # "punkt" misses the non-terminal ones
    votes[1, 12] = True                                  # "sat" splits at an abbreviation
    ds = np.tile(np.where(gold, 0.9, 0.1), (4, 1))
    prior = np.full(201, 0.5)
    prior[0] = 0
    prior /= prior.sum()
    return DocCtx("d", "g", gold, cats, ev, {k: max_soft_f1(tuple(v)) for k, v in ev.items()}, votes,
                  consensus_refs(votes), ds, prior)


class PrimitiveTests(unittest.TestCase):
    def test_soft_f1_matches_reference_implementation(self):
        rng = np.random.default_rng(1)
        p = rng.random(30)
        c = rng.random((5, 30)) > 0.6
        mine = soft_f1(c, p)
        for row, v in zip(c, mine):
            self.assertAlmostEqual(v, soft_fbeta(row.astype(int).tolist(), p))

    def test_hard_f1_and_empty(self):
        self.assertAlmostEqual(hard_f1(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))[0], 0.5)
        self.assertEqual(hard_f1(np.zeros(4), np.zeros(4))[0], 1.0)

    def test_sentence_lengths_cover_all_words(self):
        row = np.array([0, 1, 0, 0, 1, 0])               # 7 words, boundaries after words 1 and 4
        self.assertEqual(sentence_lengths(row).tolist(), [2, 3, 2])
        self.assertEqual(sentence_lengths(row).sum(), 7)

    def test_length_score_prefers_likely_lengths(self):
        prior = np.full(11, 1e-6)
        prior[3] = 1.0
        good, bad = np.array([0, 0, 1, 0, 0]), np.array([1, 1, 1, 1, 1])
        self.assertGreater(length_score(good, prior)[0], length_score(bad, prior)[0])

    def test_credit_and_top1(self):
        self.assertEqual(credit(np.array([0.2, 0.0, -0.1])).tolist(), [1.0, 0.5, 0.0])
        self.assertEqual(top1_credit([1, 1, 1, 1]), 1.0)
        self.assertEqual(top1_credit([1, 0.5, 1, 1]), 0.5)
        self.assertEqual(top1_credit([1, 0, 1, 1]), 0.0)

    def test_slope_and_pair_agreement(self):
        x = np.array([1.0, 0.8, 0.6])
        self.assertAlmostEqual(ols_slope(x, 2 * x + 1), 2.0)
        self.assertTrue(np.isnan(ols_slope(np.ones(3), x)))
        self.assertEqual(pair_agreement(x, x), 1.0)
        self.assertEqual(pair_agreement(x, -x), 0.0)
        self.assertEqual(pair_agreement(x, np.zeros(3)), 0.5)

    def test_consensus_refs_exclude_the_system(self):
        votes = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
        refs = consensus_refs(votes)
        self.assertEqual(refs[0].tolist(), [False, True])   # others: 1 vote at gap 0, 2 at gap 1
        self.assertEqual(refs[2].tolist(), [True, False])


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        self.ctx = make_ctx()
        self.rng = np.random.default_rng(0)

    def test_each_operation_touches_the_right_positions(self):
        g, cats = self.ctx.gold, self.ctx.cats
        for op, check in (("merge_terminal", lambda c: (g & ~c & (cats == TERMINAL)).sum() == 1),
                          ("merge_other", lambda c: (g & ~c & (cats != TERMINAL)).sum() == 1),
                          ("split_terminal", lambda c: (c & ~g & (cats == TERMINAL)).sum() == 1),
                          ("split_punct", lambda c: (c & ~g & (cats == PUNCT)).sum() == 1),
                          ("split_nopunct", lambda c: (c & ~g & (cats == NOPUNCT)).sum() == 1)):
            c = g.copy()
            self.assertTrue(apply_op(c, g, cats, op, self.rng))
            self.assertTrue(check(c), op)
            self.assertEqual((c != g).sum(), 1, op)

    def test_shift_moves_within_window(self):
        g = self.ctx.gold
        c = g.copy()
        self.assertTrue(apply_op(c, g, self.ctx.cats, "shift", self.rng))
        removed, added = np.flatnonzero(g & ~c), np.flatnonzero(c & ~g)
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(added), 1)
        self.assertLessEqual(abs(int(removed[0]) - int(added[0])), 3)
        self.assertEqual(c.sum(), g.sum())

    def test_unavailable_operation_is_skipped(self):
        g = self.ctx.gold
        cats = np.full(len(g), TERMINAL)
        c = g.copy()
        self.assertFalse(apply_op(c, g, cats, "merge_other", self.rng))
        self.assertTrue((c == g).all())

    def test_perturb_counts_and_rates(self):
        self.assertEqual(n_ops(0.05, 6), 1)
        self.assertEqual(n_ops(0.40, 6), 2)
        c, done = perturb(self.ctx.gold, self.ctx.cats, "merge_terminal", 10, self.rng)
        self.assertEqual(done, 4)                        # only 4 terminal Gold boundaries exist
        self.assertEqual(c.sum(), 2)

    def test_candidates_are_seeded_and_start_with_gold(self):
        a = candidates(self.ctx, 3)
        b = candidates(self.ctx, 3)
        self.assertTrue((a[0] == b[0]).all())
        self.assertTrue((a[0][0] == self.ctx.gold).all())
        self.assertEqual(a[1][0], -1)
        self.assertTrue(set(a[1][1:].tolist()) <= set(range(len(TYPES))))

    def test_doc_stats_show_the_evidence_blind_spot(self):
        st = doc_perturbation_stats(self.ctx, 0)
        from src.a8_supplement import SCORERS
        i = SCORERS.index("Stack")
        mt, mo = TYPES.index("merge_terminal"), TYPES.index("merge_other")
        self.assertGreater(st["slope"][i, mt], st["slope"][i, mo])   # merges the evidence cannot see cost less
        self.assertTrue(np.isnan(st["detect"][SCORERS.index("PuncEv")]).all())   # no PuncEv evidence here
        self.assertEqual(st["mixed_true"][0], 1.0)


class RealSystemTests(unittest.TestCase):
    def setUp(self):
        self.ctx = make_ctx()

    def test_gold_vs_systems_uses_the_same_reference(self):
        wins = gold_vs_systems(self.ctx)
        self.assertEqual(wins["Rule"].shape, (4,))
        self.assertEqual(wins["Cons"][2], 0.5)          # identical to Gold
        self.assertEqual(wins["Stack"][1], 1.0)         # the abbreviation split is penalised

    def test_error_masks_and_fix(self):
        c = self.ctx.votes[0]
        m = error_masks(c, self.ctx.gold, self.ctx.cats)
        self.assertEqual(int(m["miss_other"].sum()), 2)
        self.assertEqual(int(m["miss_terminal"].sum()), 0)
        self.assertTrue((fixed(c, m["miss_other"], "miss_other") == self.ctx.gold).all())
        c2 = self.ctx.votes[1]
        m2 = error_masks(c2, self.ctx.gold, self.ctx.cats)
        self.assertEqual(int(m2["fp_terminal"].sum()), 1)
        self.assertTrue((fixed(c2, m2["fp_terminal"], "fp_terminal") == self.ctx.gold).all())

    def test_pricing_shows_unpunished_misses(self):
        out = doc_error_pricing(self.ctx)
        e = out[(SYSTEMS[0], "miss_other")]
        self.assertEqual(e["count"], 2)
        self.assertGreater(e["d_true"], 0)
        self.assertLess(e["d_score"]["Stack"], e["d_true"])

    def test_score_matrix_reference_choice(self):
        c = self.ctx.votes[[0]]
        avg = score_matrix(c, self.ctx)["Cons"][0]
        own = score_matrix(c, self.ctx, ref=0)["Cons"][0]
        self.assertTrue(0 <= avg <= 1 and 0 <= own <= 1)


class BootstrapTests(unittest.TestCase):
    def test_plan_is_stratified(self):
        genres = np.array(["a"] * 3 + ["b"] * 2)
        for idx in resample_plan(genres, 20):
            self.assertEqual(int((idx < 3).sum()), 3)
            self.assertEqual(int((idx >= 3).sum()), 2)


if __name__ == "__main__":
    unittest.main()
