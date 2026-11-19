from __future__ import annotations

import unittest

import numpy as np

from src.a11_analysis import soft_f1_end
from src.a12_test import EVALUATORS, FINAL_OF, LADDER, q_scores
from src.a8_supplement import hard_f1, soft_f1
from src.meta_eval import max_soft_f1


class A12Tests(unittest.TestCase):
    def test_q_scores_kinds(self):
        q = np.array([0.9, 0.2, 0.7, 0.05])
        rows = np.array([[1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], bool)
        np.testing.assert_allclose(q_scores(rows, q, "Q"), soft_f1(rows, q) / max_soft_f1(tuple(q)))
        np.testing.assert_allclose(q_scores(rows, q, "Q_end"), soft_f1_end(rows, q))
        np.testing.assert_allclose(q_scores(rows, q, "hard"), hard_f1(rows, q >= 0.5))

    def test_ladder_names_are_defined(self):
        for scen, names in LADDER.items():
            for n in names:
                self.assertTrue(n == "Probe" or n in EVALUATORS, n)
            self.assertIn(FINAL_OF[scen], names)


if __name__ == "__main__":
    unittest.main()
