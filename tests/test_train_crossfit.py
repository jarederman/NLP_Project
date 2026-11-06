from __future__ import annotations

import unittest
from collections import Counter

from src.train_crossfit import assign_folds, fit_fold_calibrator


class CrossFitTests(unittest.TestCase):
    def test_folds_are_deterministic_and_stratified(self):
        domains = {f"{g}_{i}": g for g in ("news", "bio", "vlog") for i in range(10)}
        first = assign_folds(domains, 5, seed=7)
        self.assertEqual(first, assign_folds(dict(reversed(list(domains.items()))), 5, seed=7))
        for genre in ("news", "bio", "vlog"):
            counts = Counter(fold for doc, fold in first.items() if domains[doc] == genre)
            self.assertEqual(sorted(counts.values()), [2, 2, 2, 2, 2])

    def test_calibrator_uses_only_documents_passed_in(self):
        raw = {"a": [0.9, 0.1], "b": [0.8, 0.2], "held_out": [0.5, 0.5]}
        labels = {"a": [1, 0], "b": [1, 0]}
        fitted = fit_fold_calibrator(raw, labels).to_dict()
        # Adding a held-out raw vector (without labels) must not change the fit.
        self.assertEqual(fitted, fit_fold_calibrator({"a": raw["a"], "b": raw["b"]}, labels).to_dict())


if __name__ == "__main__":
    unittest.main()
