from __future__ import annotations

import unittest

from src.selection import (
    ScoredSegmentation,
    direct_internal_summary,
    selection_report,
    threshold_evidence,
)


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

    def test_non_gap_boundaries_are_gold_false_positives_only(self):
        args = ("d1", "wtp", (0, 1, 0, 1), (0, 1, 0, 1), (0.1, 0.9, 0.1, 0.9))
        clean = ScoredSegmentation(*args)
        non_gap_split = ScoredSegmentation(*args, non_gap_fp=2)
        self.assertEqual(non_gap_split.gold_counts.fp, clean.gold_counts.fp + 2)
        self.assertAlmostEqual(non_gap_split.gold_f1, 2 * 2 / (2 * 2 + 2))
        # The gold-free evaluator has no evidence inside tokens: Soft F1 is unchanged.
        self.assertEqual(non_gap_split.quality, clean.quality)
        with self.assertRaises(ValueError):
            ScoredSegmentation(*args, non_gap_fp=-1)

    def test_selected_minus_fixed_is_reported(self):
        report = selection_report(self.records(), fixed_system="sat")
        macro = report["macro"]
        self.assertAlmostEqual(macro["selected_minus_fixed"], macro["oracle_headroom_over_fixed"])
        self.assertGreater(macro["selected_minus_fixed"], 0.0)
        self.assertEqual(macro["fixed_system"], "sat")
        self.assertIn("selected_minus_fixed", report["micro_sensitivity"])

    def test_direct_internal_uses_one_evidence_vector_per_document(self):
        direct = direct_internal_summary(self.records(), threshold=0.5)
        # Thresholding the evidence reproduces Gold exactly on both documents.
        self.assertEqual(direct["documents"], 2.0)
        self.assertAlmostEqual(direct["macro_gold_f1"], 1.0)
        self.assertAlmostEqual(direct["micro_gold_f1"], 1.0)
        strict = direct_internal_summary(self.records(), threshold=0.85)
        self.assertLess(strict["macro_gold_f1"], 1.0)

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


class NonGapGoldTests(unittest.TestCase):
    """Amendment 4a: Gold boundaries between tokens with no whitespace count exactly in Gold F1."""

    def test_non_gap_counts(self):
        from src.selection import non_gap_counts
        # system splits at 10 (a non-gap Gold boundary) and twice at 30 (not Gold); Gold also at 20
        self.assertEqual(non_gap_counts([10, 30, 30], {10, 20}), (1, 2, 1))
        self.assertEqual(non_gap_counts([], set()), (0, 0, 0))
        self.assertEqual(non_gap_counts([5], set()), (0, 1, 0))     # GUM behaviour: every non-gap split is a FP

    def test_gold_counts_include_non_gap(self):
        record = ScoredSegmentation("d", "s", (1, 0), (1, 1), (0.9, 0.1), non_gap_fp=1, non_gap_tp=1, non_gap_fn=2)
        counts = record.gold_counts
        self.assertEqual((counts.tp, counts.fp, counts.fn), (2, 1, 3))
        with self.assertRaises(ValueError):
            ScoredSegmentation("d", "s", (1,), (1,), (0.5,), non_gap_fn=-1)
