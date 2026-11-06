from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


PUNCTUATION = (".", "?", "!")
CLOSERS = r"[\"'”’»\)\]\}]"
# Bracketed reference/annotation marker after a sentence-final mark, e.g. "[17]",
# "[a]", "[citation needed]". Such markers conventionally belong to the sentence
# they follow, so the terminal unit ends after them.
REFERENCE_MARKER = r"\[[^\[\]\n]{1,30}\]"
# surface_v1: terminal mark plus directly attached closers.
TRAILING_PUNCTUATION = re.compile(rf"([.!?]+)({CLOSERS}+)?$")
# surface_v1.1: additionally allow whitespace-separated reference markers.
TRAILING_PUNCTUATION_V11 = re.compile(rf"([.!?]+)({CLOSERS}*)((?:\s*{REFERENCE_MARKER})*)$")
LEADING_REFERENCE_MARKER = re.compile(rf"^\s*{REFERENCE_MARKER}")
PROTOCOLS = {
    "surface_v1_replace_visible_terminal_else_insert_gap": "v1",
    "surface_v1.1_terminal_unit_incl_reference_markers_else_insert_gap": "v1.1",
}
DEFAULT_PROTOCOL = "surface_v1_replace_visible_terminal_else_insert_gap"


def _terminal_match(text: str, boundary_offset: int, protocol: str):
    """Return the terminal-mark match that owns this gap, or None.

    v1.1 treats "terminal mark + closers + reference markers" as one sentence-final
    unit. Only the gap after the complete unit owns the terminal-mark evidence; a
    gap inside the unit (e.g. between "." and "[17]") is queried as a plain gap,
    so one visible mark is never counted at two candidate positions.
    """
    version = PROTOCOLS.get(protocol)
    if version is None:
        raise ValueError(f"Unknown mask protocol: {protocol}")
    left = text[:boundary_offset]
    if version == "v1":
        return TRAILING_PUNCTUATION.search(left)
    if LEADING_REFERENCE_MARKER.match(text[boundary_offset:]):
        return None
    return TRAILING_PUNCTUATION_V11.search(left)


def mask_anchor(
    text: str, boundary_offset: int, protocol: str = DEFAULT_PROTOCOL
) -> tuple[str, int]:
    """Choose an MLM query without asking for a second terminal mark.

    If a terminal mark already ends the text before the gap (possibly followed by
    closing punctuation, and in v1.1 by reference markers), replace that mark.
    Otherwise insert a mask at the gap. The decision depends only on raw text,
    never on a gold boundary.
    """
    if not (0 < boundary_offset < len(text)):
        raise ValueError(f"Boundary offset outside document interior: {boundary_offset}")
    match = _terminal_match(text, boundary_offset, protocol)
    if match:
        return "replace_terminal_punctuation", match.start(1)
    return "insert_at_gap", boundary_offset


def mask_span(
    text: str, boundary_offset: int, protocol: str = DEFAULT_PROTOCOL
) -> tuple[str, int, int]:
    mode, start = mask_anchor(text, boundary_offset, protocol)
    if mode == "insert_at_gap":
        return mode, start, start
    match = _terminal_match(text, boundary_offset, protocol)
    assert match is not None
    return mode, match.start(1), match.end(1)


def load_surface_bert(model_name: str, cache_dir: str):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, cache_dir=cache_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    if not hasattr(model, "bert") or not hasattr(model, "cls"):
        raise TypeError("The frozen extractor requires a BertForMaskedLM-compatible model")
    punct_ids: list[int] = []
    mapping: dict[str, int] = {}
    for mark in PUNCTUATION:
        ids = tokenizer(mark, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Punctuation {mark!r} is not one BERT token: {ids}")
        punct_ids.append(ids[0])
        mapping[mark] = ids[0]
    return model, tokenizer, punct_ids, mapping


def extract_surface_probabilities(
    model, tokenizer, punct_ids: list[int], doc, max_length: int, batch_size: int,
    protocol: str = DEFAULT_PROTOCOL,
) -> dict:
    """Score all canonical raw-text gaps with balanced bidirectional context."""
    import torch

    encoded = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoded["input_ids"]
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    start_to_token = {start: i for i, (start, end) in enumerate(offsets) if end > start}
    span_to_token = {pair: i for i, pair in enumerate(offsets) if pair[1] > pair[0]}
    plans: list[tuple[str, int, int, int]] = []
    for boundary_offset in doc.candidate_offsets:
        mode, anchor_start, anchor_end = mask_span(doc.text, boundary_offset, protocol)
        if mode == "insert_at_gap":
            right_index = start_to_token.get(boundary_offset + 1)
            if right_index is None:
                raise ValueError(
                    f"Cannot align right context in {doc.doc_id} at {boundary_offset}"
                )
            plans.append((mode, anchor_start, right_index, right_index))
        else:
            token_indices = [span_to_token.get((char, char + 1)) for char in range(anchor_start, anchor_end)]
            if any(index is None for index in token_indices):
                containing = [
                    (i, span) for i, span in enumerate(offsets)
                    if span[0] < anchor_end and span[1] > anchor_start
                ]
                raise ValueError(
                    f"Terminal mark is not exact BERT token(s) in {doc.doc_id} "
                    f"at {anchor_start}:{anchor_end}: {containing[:4]}"
                )
            indices = [int(index) for index in token_indices]
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(f"Non-contiguous punctuation tokens in {doc.doc_id}: {indices}")
            plans.append((mode, anchor_start, indices[0], indices[-1] + 1))

    left_budget = (max_length - 3) // 2
    right_budget = max_length - 3 - left_budget
    probabilities: list[float] = []
    for batch_start in range(0, len(plans), batch_size):
        batch = plans[batch_start:batch_start + batch_size]
        sequences: list[list[int]] = []
        mask_positions: list[int] = []
        for mode, _, left_stop, right_start in batch:
            if mode == "replace_terminal_punctuation":
                left = full_ids[max(0, left_stop - left_budget):left_stop]
                right = full_ids[right_start:right_start + right_budget]
            else:
                left = full_ids[max(0, left_stop - left_budget):left_stop]
                right = full_ids[right_start:right_start + right_budget]
            sequence = [tokenizer.cls_token_id, *left, tokenizer.mask_token_id, *right, tokenizer.sep_token_id]
            sequences.append(sequence)
            mask_positions.append(1 + len(left))
        width = max(map(len, sequences))
        ids = torch.full(
            (len(sequences), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda"
        )
        attention = torch.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device="cuda")
            attention[row, :len(sequence)] = 1
        mask_index = torch.tensor(mask_positions, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            hidden = model.bert(
                input_ids=ids, attention_mask=attention, return_dict=True
            ).last_hidden_state
            masked_hidden = hidden[torch.arange(len(sequences), device="cuda"), mask_index]
            logits = model.cls(masked_hidden).float()
            log_denom = torch.logsumexp(logits, dim=-1)
            log_num = torch.logsumexp(logits[:, punct_ids], dim=-1)
            probabilities.extend(torch.exp(log_num - log_denom).cpu().tolist())
        del ids, attention, hidden, masked_hidden, logits

    values = np.asarray(probabilities, dtype=float)
    if len(values) != len(doc.candidate_offsets) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid surface BERT output for {doc.doc_id}")
    return {
        "doc_id": doc.doc_id,
        "text_sha256": doc.text_sha256,
        "candidate_offsets": doc.candidate_offsets,
        "raw_punctuation_probability": values.tolist(),
        "mask_modes": [plan[0] for plan in plans],
        "mask_anchors": [plan[1] for plan in plans],
        "token_count": len(full_ids),
        "protocol": protocol,
    }


def write_cache(records: list[dict], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def read_cache(path: str | Path) -> dict[str, dict]:
    source = Path(path)
    if not source.exists():
        return {}
    with source.open(encoding="utf-8") as handle:
        return {record["doc_id"]: record for line in handle if (record := json.loads(line))}
