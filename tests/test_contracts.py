from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.contracts import (
    CandidateSegmentation,
    GoldBoundaries,
    RawDocument,
    read_raw_documents,
    write_jsonl,
)


class ContractTests(unittest.TestCase):
    def test_gold_and_candidate_validate_against_exact_text_snapshot(self):
        document = RawDocument.create(
            doc_id="d1", corpus="toy", split="dev", genre="news",
            text="First sentence. Second sentence.", source_ref="toy-v1",
        )
        gold = GoldBoundaries("d1", document.sha256, (15,), "human")
        candidate = CandidateSegmentation("d1", document.sha256, "sat", (15,), "1.0", "default")
        gold.validate_against(document)
        candidate.validate_against(document)

    def test_checksum_blocks_offsets_from_another_text(self):
        document = RawDocument.create(
            doc_id="d1", corpus="toy", split="dev", genre="news",
            text="First. Second.", source_ref="toy-v1",
        )
        wrong = GoldBoundaries("d1", "0" * 64, (6,), "human")
        with self.assertRaisesRegex(ValueError, "different text snapshot"):
            wrong.validate_against(document)

    def test_raw_document_jsonl_round_trip(self):
        document = RawDocument.create(
            doc_id="d1", corpus="toy", split="dev", genre="news",
            text="First. Second.", source_ref="toy-v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            write_jsonl([document], path)
            self.assertEqual(read_raw_documents(path), [document])


if __name__ == "__main__":
    unittest.main()
