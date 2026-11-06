from __future__ import annotations

import unittest

from src.bert_surface import mask_anchor, mask_span

V11 = "surface_v1.1_terminal_unit_incl_reference_markers_else_insert_gap"


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

    def test_terminal_punctuation_run_is_one_mask_query(self):
        text = 'Really?!" Next'
        boundary = text.rindex(" ")
        self.assertEqual(mask_span(text, boundary), ("replace_terminal_punctuation", 6, 8))


class TerminalUnitV11Test(unittest.TestCase):
    def test_reference_marker_after_terminal_owns_the_evidence(self):
        text = "in Dulwich. [17] Placed under care"
        after_marker = text.index(" Placed")
        inside_unit = text.index(" [17]")
        self.assertEqual(mask_span(text, after_marker, V11), ("replace_terminal_punctuation", 10, 11))
        # The gap inside the unit must not reuse the same terminal mark.
        self.assertEqual(mask_anchor(text, inside_unit, V11), ("insert_at_gap", inside_unit))
        # v1 behaviour is preserved for reproducibility.
        self.assertEqual(mask_anchor(text, after_marker), ("insert_at_gap", after_marker))
        self.assertEqual(mask_anchor(text, inside_unit), ("replace_terminal_punctuation", 10))

    def test_multiple_markers_and_closers(self):
        text = 'feelings." [20][21] Byron returned'
        self.assertEqual(
            mask_span(text, text.index(" Byron"), V11), ("replace_terminal_punctuation", 8, 9)
        )
        self.assertEqual(mask_anchor(text, text.index(" [20]"), V11)[0], "insert_at_gap")

    def test_v11_matches_v1_without_reference_markers(self):
        for text in ["Mr. Smith", 'He left." Then', "value 3.14 remains", "Is it? Yes", "Results [3] show"]:
            for gap in [i for i, ch in enumerate(text) if ch == " "]:
                with self.subTest(text=text, gap=gap):
                    self.assertEqual(mask_span(text, gap, V11), mask_span(text, gap))

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            mask_anchor("A. B", 2, "surface_v9")


if __name__ == "__main__":
    unittest.main()
