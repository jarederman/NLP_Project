from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


PUNCTUATION = (".", "?", "!")
TRAILING_PUNCTUATION = re.compile(r"([.!?])([\"'”’»\)\]\}]+)?$")


def mask_anchor(text: str, boundary_offset: int) -> tuple[str, int]:
    """Choose an MLM query without asking for a second terminal mark.

    If a terminal mark already occurs immediately before the gap (possibly
    inside closing punctuation), replace that mark. Otherwise insert a mask at
    the gap. The decision depends only on raw text, never on a gold boundary.
    """
    if not (0 < boundary_offset < len(text)):
        raise ValueError(f"Boundary offset outside document interior: {boundary_offset}")
    match = TRAILING_PUNCTUATION.search(text[:boundary_offset])
    if match:
        return "replace_terminal_punctuation", match.start(1)
    return "insert_at_gap", boundary_offset


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
    model, tokenizer, punct_ids: list[int], doc, max_length: int, batch_size: int
) -> dict:
    """Score all canonical raw-text gaps with balanced bidirectional context."""
    import torch

    encoded = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoded["input_ids"]
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    start_to_token = {start: i for i, (start, end) in enumerate(offsets) if end > start}
    span_to_token = {pair: i for i, pair in enumerate(offsets) if pair[1] > pair[0]}
    plans: list[tuple[str, int, int]] = []
    for boundary_offset in doc.candidate_offsets:
        mode, anchor = mask_anchor(doc.text, boundary_offset)
        if mode == "insert_at_gap":
            right_index = start_to_token.get(boundary_offset + 1)
            if right_index is None:
                raise ValueError(
                    f"Cannot align right context in {doc.doc_id} at {boundary_offset}"
                )
            plans.append((mode, anchor, right_index))
        else:
            token_index = span_to_token.get((anchor, anchor + 1))
            if token_index is None:
                containing = [
                    (i, span) for i, span in enumerate(offsets)
                    if span[0] <= anchor < span[1]
                ]
                raise ValueError(
                    f"Terminal mark is not an exact BERT token in {doc.doc_id} at {anchor}: {containing[:2]}"
                )
            plans.append((mode, anchor, token_index))

    left_budget = (max_length - 3) // 2
    right_budget = max_length - 3 - left_budget
    probabilities: list[float] = []
    for batch_start in range(0, len(plans), batch_size):
        batch = plans[batch_start:batch_start + batch_size]
        sequences: list[list[int]] = []
        mask_positions: list[int] = []
        for mode, _, token_index in batch:
            if mode == "replace_terminal_punctuation":
                left = full_ids[max(0, token_index - left_budget):token_index]
                right = full_ids[token_index + 1:token_index + 1 + right_budget]
            else:
                left = full_ids[max(0, token_index - left_budget):token_index]
                right = full_ids[token_index:token_index + right_budget]
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
        "protocol": "surface_v1_replace_visible_terminal_else_insert_gap",
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
