import unittest

import numpy as np

from src.a6_data import SYSTEMS
from src.a7_posthoc import hard_reference_scores, unstratified
from src.quality_agreement import Items


class Doc:
    def __init__(self, doc_id, offsets):
        self.doc_id, self.candidate_offsets = doc_id, offsets


class HardReferenceTests(unittest.TestCase):
    def test_f1_against_thresholded_evidence(self):
        docs = [Doc("a", [3, 7, 11, 15])]
        p = np.array([0.9, 0.2, 0.6, 0.1])                 # reference = {3, 11}
        preds = {("a", s): set() for s in SYSTEMS}
        preds[("a", SYSTEMS[0])] = {3, 11}                  # exact
        preds[("a", SYSTEMS[1])] = {3, 7, 99}               # 99 = non-gap position, ignored
        out = hard_reference_scores(docs, preds, p)
        self.assertEqual(len(out), len(SYSTEMS))
        self.assertAlmostEqual(out[0], 1.0)
        self.assertAlmostEqual(out[1], 0.5)                 # tp 1 of 2 vs 2
        self.assertAlmostEqual(out[2], 0.0)

    def test_both_empty_scores_one(self):
        docs = [Doc("a", [3, 7])]
        preds = {("a", s): set() for s in SYSTEMS}
        self.assertTrue((hard_reference_scores(docs, preds, np.array([0.1, 0.2])) == 1.0).all())


class UnstratifiedTests(unittest.TestCase):
    def test_single_stratum_keeps_everything_else(self):
        items = Items(np.array(["d1", "d2"]), np.array(["x", "y"]), np.array(["s", "s"]), np.array([0.5, 0.9]),
                      {"a": np.array([1.0, 2.0])})
        flat = unstratified(items)
        self.assertEqual(set(flat.domain.tolist()), {"all"})
        np.testing.assert_array_equal(flat.gold, items.gold)
        self.assertEqual(list(items.domain), ["x", "y"])     # original untouched


if __name__ == "__main__":
    unittest.main()
