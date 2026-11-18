from __future__ import annotations

import unittest

from src.contracts import RawDocument
from src.real_splitters import canonicalize_offsets, segment_offsets


class RealSplitterAlignmentTest(unittest.TestCase):
    def test_text_preserving_segments_map_to_offsets(self):
        text = "First sentence. Second sentence. Third."
        segments = ["First sentence. ", "Second sentence. ", "Third."]
        self.assertEqual(segment_offsets(text, segments), (15, 32))

    def test_normalized_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not reconstruct input exactly"):
            segment_offsets("One  two.", ["One ", "two."])

    def test_noncanonical_boundary_is_logged_not_silently_moved(self):
        document = RawDocument.create(
            doc_id="d", corpus="toy", split="dev", genre="news",
            text="First sentence. Second.", source_ref="toy",
        )
        offsets, failures = canonicalize_offsets(document, (5, 15), {15}, "toy")
        self.assertEqual(offsets, (15,))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].offset, 5)


if __name__ == "__main__":
    unittest.main()
