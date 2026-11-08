import inspect
import unittest

import numpy as np

from src.gap_features import _abbreviation_core, document_features, vote_features, word_shape


class GapFeatureTests(unittest.TestCase):
    def test_word_shape(self):
        self.assertEqual(word_shape("Hello"), "cap")
        self.assertEqual(word_shape("“NASA,"), "allcap")
        self.assertEqual(word_shape("world."), "lower")
        self.assertEqual(word_shape("1990s"), "digit")
        self.assertEqual(word_shape("--"), "other")

    def test_abbreviation_core(self):
        self.assertEqual(_abbreviation_core("Dr."), "dr")
        self.assertEqual(_abbreviation_core("U.S.)"), "u.s")
        self.assertIsNone(_abbreviation_core("wait..."))
        self.assertIsNone(_abbreviation_core("word"))

    def test_features_are_gold_free_and_aligned(self):
        params = set(inspect.signature(document_features).parameters)
        self.assertEqual(params, {"text", "offsets", "raw_probability", "mask_modes"})
        text = "Intro Heading Dr. Smith came. He left - then Dr went"
        offsets = [m.end() for m in __import__("re").finditer(r"\S+", text)][:-1]
        n = len(offsets)
        groups = document_features(text, offsets, [0.1] * n, ["insert_at_gap"] * n)
        self.assertEqual({g: a.shape[0] for g, a in groups.items()}, {g: n for g in "BCKALO"})
        dr = offsets[2]                      # gap after "Dr."
        abbreviation = groups["A"][offsets.index(dr)]
        self.assertEqual(abbreviation[0], 1.0)
        self.assertGreater(abbreviation[4], -1.0)   # "Dr" also occurs without a period
        came = offsets.index(text.index("came.") + 5)
        layout = groups["L"]
        self.assertAlmostEqual(layout[came, 0], np.log1p(2))   # "Dr." is the previous terminal unit
        self.assertTrue(np.all(np.isfinite(np.hstack(list(groups.values())))))

    def test_vote_features(self):
        votes = vote_features([1, 2, 3], {"a": (1, 0, 1), "b": (1, 1, 0)})
        self.assertEqual(votes[:, 0].tolist(), [1.0, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()


class OrthographicTests(unittest.TestCase):
    def test_capitalised_common_word_after_heading(self):
        import re
        text = "Results The model works . the end . The cat"
        offsets = [m.end() for m in re.finditer(r"\S+", text)][:-1]
        n = len(offsets)
        ortho = document_features(text, offsets, [0.1] * n, ["insert_at_gap"] * n)["O"]
        heading_gap = offsets.index(text.index("Results") + 7)
        # "The" is capitalised here, appears lower-case elsewhere, never capitalised mid-sentence.
        self.assertEqual(ortho[heading_gap, 2], 1.0)
        self.assertEqual(ortho[heading_gap, 3], 1.0)
        after_terminal = offsets.index(text.index("end .") + 5)
        self.assertEqual(ortho[after_terminal, 3], 0.0)   # terminal gaps are excluded from the cross

    def test_lexical_ids(self):
        from src.gap_features import lexical_ids
        left, right = lexical_ids("Hello, “World” again", [6, 14])
        self.assertEqual(left, ["hello", "world"])
        self.assertEqual(right, ["world", "again"])
