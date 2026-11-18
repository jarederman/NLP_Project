import unittest

import numpy as np

from src.a9_casing import A9, EVIDENCE, MODELS, gap_probabilities, macro_f1, paired_mean_ci, platt, thresholded


class Doc:
    def __init__(self, doc_id, offsets, genre="g"):
        self.doc_id, self.candidate_offsets, self.genre = doc_id, offsets, genre


class CasingHelperTests(unittest.TestCase):
    def test_gap_probability_takes_the_larger_side(self):
        prob = np.array([0.1, 0.9, 0.2, 0.0, 0.3, 0.6])   # "ab cd e" style: gaps at chars 2 and 4
        self.assertEqual(gap_probabilities(prob, [2, 4]).tolist(), [0.9, 0.3])
        self.assertEqual(len(gap_probabilities(prob, [])), 0)

    def test_thresholds_and_macro_f1(self):
        docs = [Doc("a", [3, 7, 11]), Doc("b", [2])]
        p = np.array([0.9, 0.25, 0.1, 0.3])
        self.assertEqual(thresholded(docs, p, 0.25), [{3, 7}, {2}])
        self.assertEqual(thresholded(docs, p, 0.25, strict=True), [{3}, {2}])
        f = macro_f1(docs, [{3, 7}, set()], {"a": {3, 99}, "b": set()})   # 99 is not a gap: ignored
        self.assertAlmostEqual(f[0], 2 / 3)
        self.assertEqual(f[1], 1.0)

    def test_platt_logo_and_transfer(self):
        rng = np.random.default_rng(0)
        y = (rng.random(400) < 0.3).astype(float)
        raw = np.clip(np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.05, 400), 0.01, 0.99)
        fold = np.repeat(np.arange(4), 100)
        logo = platt(raw, y, fold, None)
        self.assertEqual(logo.shape, raw.shape)
        self.assertGreater(logo[y == 1].mean(), logo[y == 0].mean())
        out = platt(raw, y, None, np.array([0.9, 0.1]))
        self.assertGreater(out[0], out[1])

    def test_paired_ci_contains_point(self):
        diff = np.array([0.1, 0.2, 0.0, 0.1])
        point, lo, hi = paired_mean_ci(diff, np.array(["a", "a", "b", "b"]), 200)
        self.assertAlmostEqual(point, 0.1)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)

    def test_evaluated_system_is_never_evidence(self):
        self.assertNotIn("sat3lsm", EVIDENCE)
        self.assertEqual(MODELS["sat3lsm"], "sat-3l-sm")
        self.assertEqual(set(A9), {"gum", "gumdev", "gentle", "lines", "ewt"})


if __name__ == "__main__":
    unittest.main()
