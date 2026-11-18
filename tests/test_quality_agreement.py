import unittest

import numpy as np

from src.quality_agreement import build_items, pairwise_bins, statistics
from src.selection import ScoredSegmentation


class QualityAgreementTests(unittest.TestCase):
    def test_pairwise_bins_counts_ties_and_skips_gold_ties(self):
        gold = np.array([0.9, 0.5, 0.5, 0.95])
        score = np.array([0.8, 0.3, 0.6, 0.8])
        bins = pairwise_bins(score, gold)
        # pairs with gold ties: (1,2) skipped -> 5 valid pairs
        self.assertEqual(bins["all"]["pairs"], 5)
        # (0,3): score tie -> 0.5; (0,1) ok; (0,2) ok; (1,3) ok; (2,3) ok
        self.assertAlmostEqual(bins["all"]["accuracy"], 4.5 / 5)
        self.assertEqual(bins["large"]["pairs"], 4)        # all except |0.9-0.95|

    def test_within_document_restriction(self):
        gold = np.array([0.9, 0.5, 0.7, 0.6])
        score = np.array([0.1, 0.9, 0.9, 0.1])
        same = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
        self.assertEqual(pairwise_bins(score, gold, same)["all"]["accuracy"], 0.5)   # (0,1) wrong, (2,3) right

    def test_normalization_changes_cross_doc_but_not_within_doc(self):
        records = [
            ScoredSegmentation("a", "x", (1, 0, 1), (1, 0, 1), (0.9, 0.1, 0.9)),
            ScoredSegmentation("a", "y", (1, 1, 1), (1, 0, 1), (0.9, 0.1, 0.9)),
            ScoredSegmentation("b", "x", (1, 0, 0), (1, 0, 1), (0.6, 0.1, 0.3)),
            ScoredSegmentation("b", "y", (1, 0, 1), (1, 0, 1), (0.6, 0.1, 0.3)),
        ]
        items = build_items(records, {"a": "d1", "b": "d2"})
        stats = statistics(items)
        self.assertEqual(stats["soft_f1"]["within_doc_pairwise"], stats["normalized_soft_f1"]["within_doc_pairwise"])
        self.assertGreaterEqual(items.scores["normalized_soft_f1"].max(), items.scores["soft_f1"].max())


if __name__ == "__main__":
    unittest.main()
