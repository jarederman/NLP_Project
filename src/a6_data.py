"""Shared helpers for amendment 6 (strong evidence + other signals).

Corpora: ``gum`` (GUM train, the development set, LOGO cross-fitting) and the one-shot test
corpora ``gentle`` / ``lines``. Every evidence source works on the lower-cased raw text; a
document's candidate gaps are its whitespace characters, so "word" k is the k-th whitespace
chunk and gap k is the whitespace right after it.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import read_gold_boundaries, read_raw_documents
from .surface_data import load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("punkt", "sat", "stanza", "wtp")
SEED = 20261102


@dataclass(frozen=True)
class Corpus:
    name: str
    data_root: Path
    split: str
    predictions: Path
    failures: Path
    a6_root: Path            # amendment-6 artifacts for this corpus

    def path(self, *parts: str) -> Path:
        return self.a6_root.joinpath(*parts)


CORPORA = {
    "gum": Corpus("gum", ROOT / "data/real_system", "train", ROOT / "data/real_system/train_cv/predictions",
                  ROOT / "data/real_system/train_cv/alignment_failures.jsonl", ROOT / "data/real_system/train_cv/a6"),
    "gentle": Corpus("gentle", ROOT / "data/real_system_gentle", "test", ROOT / "data/real_system_gentle/predictions",
                     ROOT / "data/real_system_gentle/alignment_failures.jsonl", ROOT / "data/real_system_gentle/a6"),
    "lines": Corpus("lines", ROOT / "data/real_system_lines", "test", ROOT / "data/real_system_lines/predictions",
                    ROOT / "data/real_system_lines/alignment_failures.jsonl", ROOT / "data/real_system_lines/a6"),
}
TEST_SOURCES = {
    "gentle": ("UD_English-GENTLE", "en_gentle-ud-test.conllu"),
    "lines": ("UD_English-LinES", "en_lines-ud-test.conllu"),
}
UD_RELEASE = "r2.18"


def safe_lower(text: str) -> str:
    """Lower-case without changing the length (characters whose lower form is longer stay)."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def surfaces(corpus: Corpus):
    docs = sorted(load_surface_jsonl(corpus.data_root / f"surface_{corpus.split}.jsonl"), key=lambda d: d.doc_id)
    raw = {d.doc_id: d for d in read_raw_documents(corpus.data_root / f"raw_{corpus.split}.jsonl")}
    for d in docs:
        if raw[d.doc_id].sha256 != d.text_sha256:
            raise RuntimeError(f"Raw/surface checksum mismatch for {d.doc_id}")
        d.genre = raw[d.doc_id].genre
        spans = word_spans(d.text)
        if [e for _, e in spans[:-1]] != list(d.candidate_offsets):
            raise RuntimeError(f"Candidate gaps are not the whitespace chunks in {d.doc_id}")
    return docs


def gold_sets(corpus: Corpus) -> dict[str, set[int]]:
    raw = {d.doc_id: d for d in read_raw_documents(corpus.data_root / f"raw_{corpus.split}.jsonl")}
    out = {}
    for record in read_gold_boundaries(corpus.data_root / f"gold_{corpus.split}.jsonl"):
        record.validate_against(raw[record.doc_id])
        out[record.doc_id] = set(record.offsets)
    return out


def gap_labels(docs, gold: dict[str, set[int]]) -> np.ndarray:
    return np.concatenate([[int(o in gold[d.doc_id]) for o in d.candidate_offsets] for d in docs]).astype(float)


def doc_slices(docs) -> list[slice]:
    out, start = [], 0
    for d in docs:
        n = len(d.candidate_offsets)
        out.append(slice(start, start + n))
        start += n
    return out


def logo_folds(docs) -> np.ndarray:
    """Per-gap genre index (the LOGO fold) for GUM train."""
    genres = sorted({d.genre for d in docs})
    return np.concatenate([[genres.index(d.genre)] * len(d.candidate_offsets) for d in docs]).astype(int)


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_temporary(path: Path) -> Path:
    """Per-writer temporary file next to ``path`` (concurrent jobs writing the same destination must not share it)."""
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = unique_temporary(path)
    with temporary.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_gap_array(corpus: Corpus, name: str, docs, values: np.ndarray) -> Path:
    """Per-gap array (n_gaps,) or (n_gaps, k) saved with the doc order and gap counts."""
    path = corpus.path("gaps", f"{name}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=np.asarray(values), doc_ids=np.array([d.doc_id for d in docs]),
                        gaps=np.array([len(d.candidate_offsets) for d in docs]))
    return path


def load_gap_array(corpus: Corpus, name: str, docs) -> np.ndarray:
    saved = np.load(corpus.path("gaps", f"{name}.npz"))
    if list(saved["doc_ids"]) != [d.doc_id for d in docs] or \
            list(saved["gaps"]) != [len(d.candidate_offsets) for d in docs]:
        raise RuntimeError(f"Gap array {name} does not match the documents of {corpus.name}")
    return saved["values"]


def write_evidence(corpus: Corpus, name: str, docs, p: np.ndarray, system_rows: dict | None = None) -> Path:
    """Evidence JSONL readable by ``quality_agreement.load_items``. ``system_rows`` maps a system to
    its own per-gap array (system-specific evidence, e.g. leave-one-out Dawid-Skene posteriors)."""
    rows = []
    for d, sl in zip(docs, doc_slices(docs)):
        base = {"doc_id": d.doc_id, "text_sha256": d.text_sha256, "candidate_offsets": d.candidate_offsets}
        if system_rows is None:
            rows.append({**base, "calibrated_boundary_probability": np.asarray(p[sl], float).tolist()})
        else:
            for system, values in system_rows.items():
                rows.append({**base, "system": system,
                             "calibrated_boundary_probability": np.asarray(values[sl], float).tolist()})
    path = corpus.path("evidence", f"{name}.jsonl")
    write_jsonl(rows, path)
    return path


def predictions(corpus: Corpus) -> dict[tuple[str, str], set[int]]:
    """Whitespace-gap boundaries of every system (non-gap ones are in the failures file)."""
    raw = {d.doc_id: d for d in read_raw_documents(corpus.data_root / f"raw_{corpus.split}.jsonl")}
    out = {}
    for system in SYSTEMS:
        for row in jsonl(corpus.predictions / f"{system}.jsonl"):
            if row["text_sha256"] != raw[row["doc_id"]].sha256:
                raise RuntimeError(f"Prediction checksum mismatch for {row['doc_id']}")
            out[(row["doc_id"], system)] = set(row["offsets"])
    return out


def vote_matrix(docs, preds: dict) -> np.ndarray:
    """(n_systems, n_gaps) 0/1 votes in SYSTEMS order."""
    return np.array([np.concatenate([[int(o in preds[(d.doc_id, s)]) for o in d.candidate_offsets] for d in docs])
                     for s in SYSTEMS], dtype=float)
