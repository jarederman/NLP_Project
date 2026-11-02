"""Gold-isolated, character-offset data contracts for the real-system study.

Splitter and scorer code should consume :class:`RawDocument` records only.  Gold
boundaries live in a separate file that is opened by evaluation code after
predictions have been frozen.  Keeping the two artifacts separate does not by
itself enforce process isolation, but it makes accidental leakage auditable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TypeVar


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_offsets(offsets: tuple[int, ...], text_length: int) -> None:
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError("Boundary offsets must be sorted and unique")
    # The mandatory document-final boundary is deliberately excluded.
    if any(offset <= 0 or offset >= text_length for offset in offsets):
        raise ValueError("Boundary offsets must be inside the document")


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    corpus: str
    split: str
    genre: str
    text: str
    source_ref: str
    sha256: str

    @classmethod
    def create(cls, *, doc_id: str, corpus: str, split: str, genre: str,
               text: str, source_ref: str) -> "RawDocument":
        if not doc_id or not text:
            raise ValueError("Raw documents require a non-empty id and text")
        return cls(doc_id, corpus, split, genre, text, source_ref, text_sha256(text))

    def validate(self) -> None:
        if self.sha256 != text_sha256(self.text):
            raise ValueError(f"Text checksum mismatch for {self.doc_id}")


@dataclass(frozen=True)
class GoldBoundaries:
    doc_id: str
    text_sha256: str
    offsets: tuple[int, ...]
    annotation_source: str

    def validate_against(self, document: RawDocument) -> None:
        document.validate()
        if self.doc_id != document.doc_id:
            raise ValueError("Gold/document ids do not match")
        if self.text_sha256 != document.sha256:
            raise ValueError("Gold offsets refer to a different text snapshot")
        _validate_offsets(self.offsets, len(document.text))


@dataclass(frozen=True)
class CandidateSegmentation:
    doc_id: str
    text_sha256: str
    system: str
    offsets: tuple[int, ...]
    system_version: str
    configuration: str

    def validate_against(self, document: RawDocument) -> None:
        document.validate()
        if self.doc_id != document.doc_id:
            raise ValueError("Candidate/document ids do not match")
        if self.text_sha256 != document.sha256:
            raise ValueError("Candidate offsets refer to a different text snapshot")
        _validate_offsets(self.offsets, len(document.text))


Record = TypeVar("Record", RawDocument, GoldBoundaries, CandidateSegmentation)


def write_jsonl(records: Iterable[Record], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record)
            if "offsets" in payload:
                payload["offsets"] = list(payload["offsets"])
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: str | Path, record_type: type[Record]) -> list[Record]:
    records: list[Record] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if "offsets" in payload:
                payload["offsets"] = tuple(payload["offsets"])
            try:
                records.append(record_type(**payload))
            except TypeError as error:
                raise ValueError(f"Invalid record at {path}:{line_number}: {error}") from error
    return records


def read_raw_documents(path: str | Path) -> list[RawDocument]:
    records = _read_jsonl(path, RawDocument)
    for record in records:
        record.validate()
    return records


def read_gold_boundaries(path: str | Path) -> list[GoldBoundaries]:
    return _read_jsonl(path, GoldBoundaries)


def read_candidate_segmentations(path: str | Path) -> list[CandidateSegmentation]:
    return _read_jsonl(path, CandidateSegmentation)
