from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


TERMINAL_ONLY = re.compile(r"^[.!?]+$")
WORD_ID = re.compile(r"^[0-9]+$")


@dataclass
class Unit:
    form: str
    space_after: bool
    upos: str
    sent_index: int


@dataclass
class Document:
    doc_id: str
    genre: str
    split: str
    text: str
    units: list[str]
    next_units: list[str]
    char_offsets: list[int]
    gold: list[int]
    sentence_count: int
    removed_terminal_marks: int

    def to_dict(self) -> dict:
        return asdict(self)


def _genre(doc_id: str) -> str:
    # GUM document ids are normally GUM_genre_name.
    parts = doc_id.split("_")
    return parts[1] if len(parts) > 2 else "unknown"


def _parse_sentence(lines: list[str], sent_index: int) -> list[Unit]:
    units: list[Unit] = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10 or not WORD_ID.match(cols[0]):
            continue
        misc = cols[9]
        units.append(Unit(cols[1], "SpaceAfter=No" not in misc, cols[3], sent_index))
    return units


def _remove_gold_terminal_punctuation(units: list[Unit]) -> tuple[list[Unit], int]:
    """Remove only standalone sentence-final .?! tokens, preserving internal punctuation.

    Quotes/brackets remain. We scan through trailing quote/bracket tokens and remove
    terminal marks immediately before them as well. Lexical abbreviations such as
    ``U.S.`` and decimals such as ``3.14`` are never rewritten.
    """
    if not units:
        return units, 0
    removable: set[int] = set()
    trailing_closers = {'"', "'", "”", "’", ")", "]", "}"}
    i = len(units) - 1
    while i >= 0 and units[i].form in trailing_closers:
        i -= 1
    while i >= 0 and units[i].upos == "PUNCT" and TERMINAL_ONLY.fullmatch(units[i].form):
        removable.add(i)
        i -= 1
    kept = [u for j, u in enumerate(units) if j not in removable]
    return kept, len(removable)


def parse_conllu(path: str | Path, split: str) -> list[Document]:
    """Read CoNLL-U and retain official document boundaries from ``# newdoc id``."""
    path = Path(path)
    docs: list[Document] = []
    current_id: str | None = None
    sentences: list[list[Unit]] = []
    sent_lines: list[str] = []

    def flush_sentence() -> None:
        nonlocal sent_lines
        if sent_lines:
            parsed = _parse_sentence(sent_lines, len(sentences))
            if parsed:
                sentences.append(parsed)
        sent_lines = []

    def flush_doc() -> None:
        nonlocal sentences, current_id
        flush_sentence()
        if current_id is None or not sentences:
            sentences = []
            return
        flattened: list[Unit] = []
        removed = 0
        sentence_last_indices: list[int] = []
        for sentence in sentences:
            clean, n_removed = _remove_gold_terminal_punctuation(sentence)
            removed += n_removed
            if clean:
                flattened.extend(clean)
                sentence_last_indices.append(len(flattened) - 1)
        # Build controlled document text. Every sentence join has one space, even if
        # source metadata marks no space, and candidate indices are whitespace gaps.
        text_parts: list[str] = []
        offsets: list[int] = []
        forms: list[str] = []
        char_pos = 0
        candidate_unit_indices: list[int] = []
        for idx, unit in enumerate(flattened):
            forms.append(unit.form)
            text_parts.append(unit.form)
            char_pos += len(unit.form)
            next_unit = flattened[idx + 1] if idx + 1 < len(flattened) else None
            sentence_break = next_unit is not None and next_unit.sent_index != unit.sent_index
            has_space = next_unit is not None and (unit.space_after or sentence_break)
            if has_space:
                offsets.append(char_pos)
                candidate_unit_indices.append(idx)
                text_parts.append(" ")
                char_pos += 1
        gold_unit_set = set(sentence_last_indices[:-1])
        gold = [1 if idx in gold_unit_set else 0 for idx in candidate_unit_indices]
        candidate_forms = [forms[idx] for idx in candidate_unit_indices]
        next_forms = [forms[idx + 1] for idx in candidate_unit_indices]
        docs.append(Document(
            doc_id=current_id,
            genre=_genre(current_id),
            split=split,
            text="".join(text_parts),
            units=candidate_forms,
            next_units=next_forms,
            char_offsets=offsets,
            gold=gold,
            sentence_count=len(sentences),
            removed_terminal_marks=removed,
        ))
        sentences = []

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("# newdoc id ="):
                flush_doc()
                current_id = line.split("=", 1)[1].strip()
                sent_lines = [line]
            elif not line:
                flush_sentence()
            else:
                sent_lines.append(line)
    flush_doc()
    return docs


def save_documents(docs: Iterable[Document], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for doc in docs:
            out.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def load_documents(path: str | Path) -> list[Document]:
    docs = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            docs.append(Document(**json.loads(line)))
    return docs
