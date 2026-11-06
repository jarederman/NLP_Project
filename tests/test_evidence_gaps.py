import unittest

from src.analyze_evidence_gaps import left_context_category, right_is_capitalized, summarize


class LeftContextCategoryTest(unittest.TestCase):
    def test_categories(self):
        text = "Done. [3] Next Heading Intro: then - more, end; x"
        self.assertEqual(left_context_category(text, text.index("[3]") + 3), "terminal")
        self.assertEqual(left_context_category(text, text.index(" Intro")), "no_punct_alnum")
        self.assertEqual(left_context_category(text, text.index(" then")), "colon")
        self.assertEqual(left_context_category(text, text.index(" more")), "dash")
        self.assertEqual(left_context_category(text, text.index(" end")), "comma")
        self.assertEqual(left_context_category(text, text.index(" x")), "semicolon")
        self.assertTrue(right_is_capitalized(text, text.index(" Next")))
        self.assertFalse(right_is_capitalized(text, text.index(" then")))

    def test_summary_splits_gold_and_non_gold(self):
        rows = [
            {"category": "terminal", "gold": True, "p": 0.9, "right_cap": True},
            {"category": "terminal", "gold": True, "p": 0.3, "right_cap": True},
            {"category": "terminal", "gold": False, "p": 0.6, "right_cap": False},
            {"category": "no_punct_alnum", "gold": False, "p": 0.0, "right_cap": False},
        ]
        result = summarize(rows)
        terminal = result["categories"]["terminal"]
        self.assertEqual(result["total_gold_boundaries"], 2)
        self.assertAlmostEqual(terminal["gold"]["mean_p"], 0.6)
        self.assertAlmostEqual(terminal["gold"]["share_p_below_0.5"], 0.5)
        self.assertAlmostEqual(terminal["gold_base_rate"], 2 / 3)
        self.assertEqual(result["categories"]["no_punct_alnum"]["gold"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
