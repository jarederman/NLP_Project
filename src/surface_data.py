from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SurfaceDocument:
    """Gold-free document surface and its canonical whitespace gaps."""

    doc_id: str
    genre: str
    split: str
    text: str
    text_sha256: str
    tokens: list[str]
    token_ends: list[int]
    candidate_offsets: list[int]
    left_units: list[str]
    right_units: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GoldBoundaries:
    """Separate evaluation-only artifact; never required for candidate scoring."""

    doc_id: str
    split: str
    boundary_offsets: list[int]
    labels: list[int]
    sentence_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _SurfaceToken:
    form: str
    space_after: bool


def _genre(doc_id: str) -> str:
    parts = doc_id.split("_")
    return parts[1] if len(parts) > 2 else "unknown"


def _token_row(line: str) -> tuple[str, str, str] | None:
    if not line or line.startswith("#"):
        return None
    cols = line.split("\t")
    if len(cols) != 10:
        return None
    return cols[0], cols[1], cols[9]


def _sentence_surface_tokens(lines: Iterable[str]) -> list[_SurfaceToken]:
    """Return CoNLL-U surface tokens, preserving MWT forms exactly once."""
    rows = [row for line in lines if (row := _token_row(line)) is not None]
    covered: set[int] = set()
    output: list[_SurfaceToken] = []
    for token_id, form, misc in rows:
        if "-" in token_id:
            start, end = token_id.split("-", 1)
            if start.isdigit() and end.isdigit():
                covered.update(range(int(start), int(end) + 1))
                output.append(_SurfaceToken(form, "SpaceAfter=No" not in misc))
            continue
        if not token_id.isdigit() or int(token_id) in covered:
            continue
        output.append(_SurfaceToken(form, "SpaceAfter=No" not in misc))
    return output


def _iter_documents_and_sentences(path: str | Path):
    """Yield sentence token blocks grouped by document.

    Blank lines are used only to reset CoNLL-U token IDs and MWT coverage. Surface
    whitespace is determined exclusively by ``SpaceAfter=No``.
    """
    current_id: str | None = None
    sentences: list[list[_SurfaceToken]] = []
    sentence_lines: list[str] = []

    def flush_sentence() -> None:
        nonlocal sentence_lines
        tokens = _sentence_surface_tokens(sentence_lines)
        if tokens:
            sentences.append(tokens)
        sentence_lines = []

    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("# newdoc id ="):
                flush_sentence()
                if current_id is not None:
                    yield current_id, sentences
                current_id = line.split("=", 1)[1].strip()
                sentences = []
            elif not line:
                flush_sentence()
            else:
                sentence_lines.append(line)
    flush_sentence()
    if current_id is not None:
        yield current_id, sentences


def _iter_surface_documents(path: str | Path):
    """Yield flat surface tokens without consulting sentence separators.

    Token-ID resets are needed only to scope CoNLL-U MWT ranges; neither blank
    lines nor sentence comments are read. They therefore cannot add whitespace
    or otherwise change the splitter input.
    """
    current_id: str | None = None
    tokens: list[_SurfaceToken] = []
    covered: set[int] = set()
    last_integer_id: int | None = None
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("# newdoc id ="):
                if current_id is not None:
                    yield current_id, tokens
                current_id = line.split("=", 1)[1].strip()
                tokens = []
                covered = set()
                last_integer_id = None
                continue
            row = _token_row(line)
            if row is None:
                continue
            token_id, form, misc = row
            if "-" in token_id:
                start, end = token_id.split("-", 1)
                if not (start.isdigit() and end.isdigit()):
                    continue
                start_id, end_id = int(start), int(end)
                if last_integer_id is not None and start_id <= last_integer_id:
                    covered = set()
                    last_integer_id = None
                covered.update(range(start_id, end_id + 1))
                tokens.append(_SurfaceToken(form, "SpaceAfter=No" not in misc))
                continue
            if not token_id.isdigit():
                continue
            integer_id = int(token_id)
            if last_integer_id is not None and integer_id <= last_integer_id:
                covered = set()
            last_integer_id = integer_id
            if integer_id in covered:
                continue
            tokens.append(_SurfaceToken(form, "SpaceAfter=No" not in misc))
    if current_id is not None:
        yield current_id, tokens


def load_surface_documents(path: str | Path, split: str) -> list[SurfaceDocument]:
    """Reconstruct document text without using sentence boundaries for spacing."""
    documents: list[SurfaceDocument] = []
    for doc_id, flat in _iter_surface_documents(path):
        if not flat:
            continue
        parts: list[str] = []
        token_ends: list[int] = []
        char_pos = 0
        for i, token in enumerate(flat):
            parts.append(token.form)
            char_pos += len(token.form)
            token_ends.append(char_pos)
            if token.space_after and i + 1 < len(flat):
                parts.append(" ")
                char_pos += 1
        text = "".join(parts)
        candidate_token_indices = [
            i for i, token in enumerate(flat[:-1]) if token.space_after
        ]
        candidate_offsets = [token_ends[i] for i in candidate_token_indices]
        documents.append(SurfaceDocument(
            doc_id=doc_id,
            genre=_genre(doc_id),
            split=split,
            text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            tokens=[token.form for token in flat],
            token_ends=token_ends,
            candidate_offsets=candidate_offsets,
            left_units=[flat[i].form for i in candidate_token_indices],
            right_units=[flat[i + 1].form for i in candidate_token_indices],
        ))
    return documents


def extract_gold_boundaries(
    path: str | Path, split: str, surface_docs: Iterable[SurfaceDocument],
    *, allow_source_superset: bool = False, allow_non_gap: bool = False,
) -> list[GoldBoundaries]:
    """Create evaluation-only labels by aligning gold sentence ends to surfaces.

    ``allow_non_gap`` keeps Gold boundaries that sit between tokens with no whitespace (absent
    from GUM, present in EWT): they stay in ``boundary_offsets`` for Gold F1 but have no
    candidate gap, so ``labels`` (per gap) cannot mark them."""
    surfaces = {doc.doc_id: doc for doc in surface_docs}
    output: list[GoldBoundaries] = []
    seen: set[str] = set()
    for doc_id, sentences in _iter_documents_and_sentences(path):
        if doc_id not in surfaces:
            if allow_source_superset:
                continue
            raise ValueError(f"Gold document {doc_id} has no reconstructed surface")
        doc = surfaces[doc_id]
        flattened = [token.form for sentence in sentences for token in sentence]
        if flattened != doc.tokens:
            raise ValueError(f"Gold/surface token mismatch in {doc_id}")
        sentence_last_token_indices: list[int] = []
        cumulative = 0
        for sentence in sentences:
            cumulative += len(sentence)
            sentence_last_token_indices.append(cumulative - 1)
        boundary_offsets = [
            doc.token_ends[index] for index in sentence_last_token_indices[:-1]
        ]
        candidates = set(doc.candidate_offsets)
        missing = [offset for offset in boundary_offsets if offset not in candidates]
        if missing and not allow_non_gap:
            raise ValueError(
                f"{doc_id} has {len(missing)} gold boundaries outside canonical whitespace gaps: {missing[:5]}"
            )
        boundary_set = set(boundary_offsets)
        output.append(GoldBoundaries(
            doc_id=doc_id,
            split=split,
            boundary_offsets=boundary_offsets,
            labels=[int(offset in boundary_set) for offset in doc.candidate_offsets],
            sentence_count=len(sentences),
        ))
        seen.add(doc_id)
    missing_docs = sorted(set(surfaces) - seen)
    if missing_docs:
        raise ValueError(f"Surface documents missing from gold extraction: {missing_docs[:5]}")
    return output


def save_jsonl(records: Iterable[SurfaceDocument | GoldBoundaries | dict], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as out:
        for record in records:
            payload = record if isinstance(record, dict) else record.to_dict()
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_surface_jsonl(path: str | Path) -> list[SurfaceDocument]:
    with Path(path).open(encoding="utf-8") as handle:
        return [SurfaceDocument(**json.loads(line)) for line in handle]


def load_gold_jsonl(path: str | Path) -> list[GoldBoundaries]:
    with Path(path).open(encoding="utf-8") as handle:
        return [GoldBoundaries(**json.loads(line)) for line in handle]
