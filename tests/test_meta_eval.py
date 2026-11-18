from __future__ import annotations

import itertools
import unittest

from src.meta_eval import (
    auxiliary_quality,
    evaluate,
    kendall_tau_b,
    max_soft_f1,
)
from src.selection import ScoredSegmentation, selection_report


def records() -> list[ScoredSegmentation]:
    ev1, ev2, ev3 = (0.1, 0.9, 0.1, 0.9), (0.8, 0.1, 0.8, 0.1), (0.6, 0.2, 0.7, 0.3)
    return [
        ScoredSegmentation("d1", "punkt", (0, 1, 0, 1), (0, 1, 0, 1), ev1),
        ScoredSegmentation("d1", "sat", (0, 1, 0, 0), (0, 1, 0, 1), ev1),
        ScoredSegmentation("d2", "punkt", (1, 0, 1, 0), (1, 0, 1, 0), ev2),
        ScoredSegmentation("d2", "sat", (1, 0, 0, 0), (1, 0, 1, 0), ev2),
        ScoredSegmentation("d3", "punkt", (1, 0, 0, 0), (1, 0, 1, 0), ev3),
        ScoredSegmentation("d3", "sat", (1, 0, 1, 1), (1, 0, 1, 0), ev3, non_gap_fp=1),
    ]


DOMAINS = {"d1": "news", "d2": "news", "d3": "fiction"}


class MetaEvalTests(unittest.TestCase):
    def test_corpus_numbers_match_selection_report(self):
        result = evaluate(records(), DOMAINS, "sat", 0.5, n_resamples=10)
        reference = selection_report(records(), fixed_system="sat")
        macro = result["point"]["scorers"]["soft_f1"]["corpus_macro"]
        self.assertEqual(macro["selected"], reference["macro"]["selected"])
        self.assertAlmostEqual(macro["regret"], reference["macro"]["selection_regret"])
        self.assertAlmostEqual(macro["selected_minus_fixed"], reference["macro"]["selected_minus_fixed"])
        micro = result["point"]["scorers"]["soft_f1"]["corpus_micro"]
        self.assertAlmostEqual(micro["regret"], reference["micro_sensitivity"]["selection_regret"])
        self.assertAlmostEqual(
            result["point"]["scorers"]["soft_f1"]["pairwise"]["accuracy"],
            reference["within_document_pairwise"]["accuracy"],
        )

    def test_domains_are_selected_separately(self):
        domains = evaluate(records(), DOMAINS, "sat", 0.5, n_resamples=10)["point"]["scorers"]["soft_f1"]["domains"]
        self.assertEqual(set(domains), {"news", "fiction"})
        self.assertEqual(domains["news"]["documents"], 2)

    def test_gold_win_rate_counts_documents(self):
        point = evaluate(records(), DOMAINS, "sat", 0.5, n_resamples=10)["point"]
        # punkt equals Gold on d1 and d2, so Gold ties for the top there.
        self.assertGreaterEqual(point["gold_win_rate"], 2 / 3)

    def test_bootstrap_interval_contains_point(self):
        intervals = evaluate(records(), DOMAINS, "sat", 0.5, n_resamples=200, seed=1)["bootstrap"]
        for name, value in intervals.items():
            if value["low"] == value["low"]:
                with self.subTest(name=name):
                    self.assertLessEqual(value["low"], value["high"])

    def test_kendall_tau_b(self):
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertNotEqual(kendall_tau_b([1, 1, 1], [1, 2, 3]), kendall_tau_b([1, 1, 1], [1, 2, 3]))  # NaN

    def test_max_soft_f1_is_exact(self):
        evidence = (0.9, 0.2, 0.6, 0.05)
        best = max(
            ScoredSegmentation("d", "x", c, (0,) * 4, evidence).quality
            for c in itertools.product((0, 1), repeat=4) if any(c)
        )
        self.assertAlmostEqual(max_soft_f1(evidence), best)
        record = ScoredSegmentation("d", "x", (1, 0, 1, 0), (1, 0, 1, 0), evidence)
        self.assertLessEqual(auxiliary_quality(record)["normalized_soft_f1"], 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
