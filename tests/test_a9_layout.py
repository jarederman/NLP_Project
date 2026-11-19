import unittest

import numpy as np

from src.a6_data import SYSTEMS
from src.a9_layout import CELLS, bootstrap, contrast, fast_stats, newline_text
from src.quality_agreement import Items, statistics


class NewlineTextTests(unittest.TestCase):
    def test_replaces_only_the_break_spaces(self):
        text = "Title Body text here. Next"
        out = newline_text(text, [5, 21])
        self.assertEqual(out, "Title\nBody text here.\nNext")
        self.assertEqual(len(out), len(text))

    def test_refuses_non_space(self):
        with self.assertRaises(ValueError):
            newline_text("ab cd", [1])


class FastStatsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n_docs = 12
        self.gold = np.round(rng.random(n_docs * len(SYSTEMS)), 2)
        self.scores = {"a": self.gold + rng.normal(0, 0.2, len(self.gold)), "b": rng.random(len(self.gold))}
        self.scores["a"][:4] = self.scores["a"][0]                     # ties inside one document
        self.docs = np.repeat([f"d{i}" for i in range(n_docs)], len(SYSTEMS))
        self.genres = np.array(["x"] * 6 + ["y"] * 6)

    def test_matches_quality_agreement(self):
        items = Items(self.docs, np.repeat(self.genres, len(SYSTEMS)), np.tile(SYSTEMS, 12), self.gold,
                      {k: v for k, v in self.scores.items()})
        ref = statistics(items)
        fast = fast_stats(np.array([self.scores["a"], self.scores["b"]]), self.gold)
        for k, name in enumerate(("a", "b")):
            self.assertAlmostEqual(fast[k, 0], ref[name]["global_spearman"], places=10)
            self.assertAlmostEqual(fast[k, 1], ref[name]["within_doc_pairwise"], places=10)

    def test_bootstrap_and_contrast(self):
        point, draws, names = bootstrap(self.scores, self.gold, self.genres, 50)
        self.assertEqual(draws.shape, (50, 2, 2))
        c = contrast(draws, names, point, ["a"], ["b"])
        self.assertAlmostEqual(c[0], point["a"][0] - point["b"][0])
        self.assertLessEqual(c[1], c[2])

    def test_cells_are_ordered_simplest_first(self):
        self.assertEqual(CELLS[0], ("lower", "flat"))
        self.assertEqual(CELLS[-1], ("cased", "nl"))


if __name__ == "__main__":
    unittest.main()
