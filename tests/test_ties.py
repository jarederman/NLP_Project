from __future__ import annotations

import unittest

import numpy as np

from src.quality_agreement import Items, statistics, tie_exact


class TieExactTests(unittest.TestCase):
    def test_noisy_equal_scores_tie_like_exact_ones(self):
        # items 0/1 and 2/3: identical candidates (same Gold F1); scores equal up to float noise
        gold = np.array([0.9, 0.9, 0.7, 0.7, 0.8])
        exact = np.array([0.3, 0.3, 0.1, 0.1, 0.2])
        noisy = exact + np.array([0.0, 1e-16, 0.0, -1e-16, 0.0]) * 0 + np.array([0.0, 0.1 + 0.2 - 0.3, 0.0, 0.0, 0.0])
        items = lambda s: Items(np.array(list("aabbc")), np.array(["x"] * 5), np.array(list("pqpqp")), gold,
                                {"s": s})
        self.assertNotEqual(noisy[0], noisy[1])
        a = statistics(items(exact))["s"]["global_spearman"]
        b = statistics(items(noisy))["s"]["global_spearman"]
        self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(a, 1.0, places=12)

    def test_rounding_keeps_distinct_values(self):
        self.assertEqual(len(set(tie_exact([0.1, 0.1 + 1e-9, 0.1 + 2e-16]).tolist())), 2)


if __name__ == "__main__":
    unittest.main()
