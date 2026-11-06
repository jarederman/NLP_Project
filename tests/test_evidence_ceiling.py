import math
import unittest

from src.evidence_ceiling import disagreement_accuracy, oracle_on, shift_evidence


class EvidenceCeilingTest(unittest.TestCase):
    def test_shift_moves_toward_gold(self):
        evidence, gold = [0.3, 0.3], [1, 0]
        self.assertAlmostEqual(shift_evidence(evidence, gold, 0.0)[0], 0.3)
        shifted = shift_evidence(evidence, gold, 1.0)
        self.assertGreater(shifted[0], 0.3)
        self.assertLess(shifted[1], 0.3)
        self.assertEqual(shift_evidence(evidence, gold, math.inf), [1.0, 0.0])

    def test_oracle_only_on_target_categories(self):
        out = oracle_on([0.2, 0.7], [1, 0], ["terminal", "dash"], frozenset({"terminal"}))
        self.assertEqual(out, [1.0, 0.7])
        gold_only = oracle_on([0.2, 0.7, 0.4], [1, 0, 0], ["dash", "dash", "dash"], frozenset({"dash"}),
                              labels=frozenset({1}))
        self.assertEqual(gold_only, [1.0, 0.7, 0.4])

    def test_disagreement_accuracy_ignores_unanimous_gaps(self):
        docs = {"d": {"gold": (1, 0, 1), "candidates": {"a": (1, 0, 1), "b": (1, 1, 0)}}}
        # gaps 1 and 2 disagree; evidence right on gap 1 (0.1 -> no boundary), wrong on gap 2
        self.assertEqual(disagreement_accuracy(docs, {"d": [0.0, 0.1, 0.2]}, 0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
