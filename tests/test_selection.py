from __future__ import annotations

import unittest

from src.selection import ScoredSegmentation, selection_report, threshold_evidence


class SelectionTests(unittest.TestCase):
    @staticmethod
    def records() -> list[ScoredSegmentation]:
        # Both systems are correct on d1.  Punkt is better on d2, so a correct
        # selector beats the fixed SaT policy without changing any boundaries.
        return [
            ScoredSegmentation("d1", "punkt", (0, 1, 0, 1), (0, 1, 0, 1), (0.1, 0.9, 0.1, 0.9)),
            ScoredSegmentation("d1", "sat", (0, 1, 0, 1), (0, 1, 0, 1), (0.1, 0.9, 0.1, 0.9)),
            ScoredSegmentation("d2", "punkt", (1, 0, 1, 0), (1, 0, 1, 0), (0.8, 0.1, 0.8, 0.1)),
            ScoredSegmentation("d2", "sat", (1, 0, 0, 0), (1, 0, 1, 0), (0.8, 0.1, 0.8, 0.1)),
        ]

    def test_macro_selection_and_fixed_system_headroom(self):
        report = selection_report(self.records(), fixed_system="sat")
        self.assertEqual(report["macro"]["selected"], "punkt")
        self.assertEqual(report["macro"]["oracle"], "punkt")
        self.assertAlmostEqual(report["macro"]["selection_regret"], 0.0)
        self.assertGreater(report["macro"]["oracle_headroom_over_fixed"], 0.0)

    def test_macro_and_micro_are_reported_separately(self):
        report = selection_report(self.records(), fixed_system="sat")
        punkt = report["systems"]["punkt"]
        self.assertIn("macro_quality", punkt)
        self.assertIn("micro_quality", punkt)
        self.assertIn("macro_gold_f1", punkt)
        self.assertIn("micro_gold_f1", punkt)

    def test_pairwise_excludes_equal_gold_quality(self):
        report = selection_report(self.records(), fixed_system="sat")
        pairwise = report["within_document_pairwise"]
        self.assertEqual(pairwise["pairs"], 1.0)
        self.assertAlmostEqual(pairwise["accuracy"], 1.0)

    def test_pairwise_metric_tie_receives_half_credit(self):
        records = [
            ScoredSegmentation("d", "punkt", (1, 0), (1, 0), (0.5, 0.5)),
            ScoredSegmentation("d", "sat", (0, 1), (1, 0), (0.5, 0.5)),
        ]
        pairwise = selection_report(records, fixed_system="sat")["within_document_pairwise"]
        self.assertAlmostEqual(pairwise["accuracy"], 0.5)
        self.assertAlmostEqual(pairwise["strict_accuracy"], 0.0)
        self.assertAlmostEqual(pairwise["tie_rate"], 1.0)

    def test_direct_internal_threshold(self):
        self.assertEqual(threshold_evidence((0.1, 0.5, 0.9), 0.5), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
