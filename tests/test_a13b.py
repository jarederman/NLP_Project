from __future__ import annotations

import unittest

import numpy as np

from src.a13b_classic import render


class A13bTests(unittest.TestCase):
    def test_render_one_sentence_per_line(self):
        text = "Hello there. How are you? Fine"
        offsets = [5, 12, 16, 20, 25]            # whitespace gaps after each word
        self.assertEqual(render(text, offsets, np.array([0, 1, 0, 0, 1])), "Hello there.\nHow are you?\nFine")
        self.assertEqual(render(text, offsets, np.zeros(5, bool)), text)


if __name__ == "__main__":
    unittest.main()
