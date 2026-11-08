"""Paragraph-break layout artifact for the controlled layout experiment (amendment 3, item 4).

GUM's CoNLL-U marks the start of each paragraph (heading, list item, paragraph, utterance)
with a ``# newpar`` comment. The reconstructed raw text deliberately drops that layout;
this module recovers it as a *separate* artifact: per document, the character offsets of the
whitespace gaps that are paragraph breaks.

Caveat recorded in PREREGISTRATION.md: in GUM a paragraph break is a sentence boundary by
construction, so this artifact hands part of the boundary information to the evidence. It is
used only for the labelled layout control, never by the default protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .surface_data import _sentence_surface_tokens, load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]


def iter_paragraph_flags(path: str | Path):
    """Yield ``(doc_id, [(n_surface_tokens, starts_paragraph), ...])`` per document."""
    current_id: str | None = None
    sentences: list[tuple[int, bool]] = []
    lines: list[str] = []
    newpar = False

    def flush() -> None:
        nonlocal lines, newpar
        tokens = _sentence_surface_tokens(lines)
        if tokens:
            sentences.append((len(tokens), newpar))
            newpar = False
        lines = []

    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("# newdoc id ="):
                flush()
                if current_id is not None:
                    yield current_id, sentences
                current_id = line.split("=", 1)[1].strip()
                sentences = []
                newpar = False
            elif not line:
                flush()
            else:
                if line == "# newpar" or line.startswith("# newpar ") or line.startswith("# newpar_block"):
                    newpar = True
                lines.append(line)
    flush()
    if current_id is not None:
        yield current_id, sentences


def paragraph_break_offsets(conllu: str | Path, surfaces) -> dict[str, list[int]]:
    """Gap offsets (as in ``candidate_offsets``) that are paragraph breaks, per document."""
    by_id = {doc.doc_id: doc for doc in surfaces}
    output: dict[str, list[int]] = {}
    for doc_id, sentences in iter_paragraph_flags(conllu):
        if doc_id not in by_id:
            continue
        doc = by_id[doc_id]
        if sum(n for n, _ in sentences) != len(doc.tokens):
            raise ValueError(f"Layout/surface token mismatch in {doc_id}")
        gaps = set(doc.candidate_offsets)
        breaks, last_token = [], -1
        for i, (n, starts_paragraph) in enumerate(sentences):
            if i > 0 and starts_paragraph:
                offset = doc.token_ends[last_token]
                if offset not in gaps:
                    raise ValueError(f"Paragraph break outside a whitespace gap in {doc_id} at {offset}")
                breaks.append(offset)
            last_token += n
        output[doc_id] = breaks
    missing = sorted(set(by_id) - set(output))
    if missing:
        raise ValueError(f"Documents without layout: {missing[:5]}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conllu", type=Path, default=ROOT / "data/UD_English-GUM/en_gum-ud-train.conllu")
    parser.add_argument("--surface", type=Path, default=ROOT / "data/real_system/surface_train.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/real_system/layout_train.jsonl")
    args = parser.parse_args()
    surfaces = load_surface_jsonl(args.surface)
    breaks = paragraph_break_offsets(args.conllu, surfaces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for doc in sorted(surfaces, key=lambda d: d.doc_id):
            out.write(json.dumps({"doc_id": doc.doc_id, "text_sha256": doc.text_sha256,
                                  "paragraph_break_offsets": breaks[doc.doc_id]}) + "\n")
    print(json.dumps({"status": "LAYOUT_COMPLETE", "documents": len(surfaces),
                      "paragraph_breaks": sum(map(len, breaks.values()))}), flush=True)


if __name__ == "__main__":
    main()
