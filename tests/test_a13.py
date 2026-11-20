from __future__ import annotations

import unittest

import numpy as np

from src.a13_baselines import REPS, control_rows
from src.a8_supplement import hard_f1


class A13Tests(unittest.TestCase):
    def test_control_is_quality_neutral(self):
        gold = np.zeros(40, bool)
        gold[[5, 12, 30]] = True
        base, moved = control_rows(gold, 3)
        self.assertEqual(len(base), REPS)
        np.testing.assert_allclose(hard_f1(base, gold), hard_f1(moved, gold))       # true F1 unchanged
        self.assertTrue(((base ^ gold).sum(1) == 1).all() and ((moved ^ gold).sum(1) == 1).all())
        self.assertFalse((base & ~gold & moved).any())                                 # the false boundary moved


if __name__ == "__main__":
    unittest.main()
