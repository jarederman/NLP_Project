from __future__ import annotations

import unittest

from src.bert_surface import mask_anchor


class MaskAnchorTest(unittest.TestCase):
    def test_existing_marks_are_replaced(self):
        cases = {
            "Mr. Smith": ("replace_terminal_punctuation", 2),
            "U.S. government": ("replace_terminal_punctuation", 3),
            'He left." Then': ("replace_terminal_punctuation", 7),
            "Is it? Yes": ("replace_terminal_punctuation", 5),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(mask_anchor(text, text.rindex(" ")), expected)

    def test_plain_gap_inserts_mask(self):
        text = "value 3.14 remains"
        first = text.index(" ")
        second = text.index(" ", first + 1)
        self.assertEqual(mask_anchor(text, first), ("insert_at_gap", first))
        self.assertEqual(mask_anchor(text, second), ("insert_at_gap", second))


if __name__ == "__main__":
    unittest.main()
