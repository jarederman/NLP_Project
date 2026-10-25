from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PUNCTUATION = (".", "?", "!")


def load_bert(model_name: str, cache_dir: str):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, cache_dir=cache_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    if not hasattr(model, "bert") or not hasattr(model, "cls"):
        raise TypeError("This minimal extractor requires a BertForMaskedLM-compatible model")
    punct_ids = []
    mapping = {}
    for mark in PUNCTUATION:
        ids = tokenizer(mark, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Punctuation {mark!r} is not one BERT token: {ids}")
        punct_ids.append(ids[0])
        mapping[mark] = ids[0]
    return model, tokenizer, punct_ids, mapping


def _candidate_token_boundaries(offset_mapping, char_offsets: list[int]) -> list[int]:
    # Return the first token on the right of each whitespace candidate.
    starts = {}
    for i, (start, end) in enumerate(offset_mapping):
        if end > start:
            starts[start] = i
    missing = [offset for offset in char_offsets if offset + 1 not in starts]
    if missing:
        raise ValueError(f"Cannot align {len(missing)} BERT right-context offsets; first={missing[:5]}")
    return [starts[offset + 1] for offset in char_offsets]


def extract_document_probabilities(model, tokenizer, punct_ids: list[int], doc, max_length: int, batch_size: int) -> dict:
    """Mask each candidate gap and score punctuation using both left and right context.

    We call the BERT encoder normally but apply the MLM head only to masked hidden
    states, avoiding a full [batch, sequence, vocabulary] logits tensor.
    """
    import torch

    encoded = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoded["input_ids"]
    right_starts = _candidate_token_boundaries(encoded["offset_mapping"], doc.char_offsets)
    left_budget = (max_length - 3) // 2
    right_budget = max_length - 3 - left_budget
    probabilities = []
    for batch_start in range(0, len(right_starts), batch_size):
        batch_positions = right_starts[batch_start:batch_start + batch_size]
        sequences, mask_positions = [], []
        for right_start in batch_positions:
            left = full_ids[max(0, right_start - left_budget):right_start]
            right = full_ids[right_start:right_start + right_budget]
            sequence = [tokenizer.cls_token_id, *left, tokenizer.mask_token_id, *right, tokenizer.sep_token_id]
            mask_positions.append(1 + len(left))
            sequences.append(sequence)
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
        attention = torch.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device="cuda")
            attention[row, :len(sequence)] = 1
        mask_index = torch.tensor(mask_positions, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            hidden = model.bert(input_ids=ids, attention_mask=attention, return_dict=True).last_hidden_state
            masked_hidden = hidden[torch.arange(len(sequences), device="cuda"), mask_index]
            logits = model.cls(masked_hidden).float()
            log_denom = torch.logsumexp(logits, dim=-1)
            log_num = torch.logsumexp(logits[:, punct_ids], dim=-1)
            probabilities.extend(torch.exp(log_num - log_denom).cpu().tolist())
        del ids, attention, hidden, masked_hidden, logits
    values = np.asarray(probabilities, dtype=float)
    if len(values) != len(doc.gold) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid BERT output for {doc.doc_id}")
    return {"doc_id": doc.doc_id, "raw_punctuation_probability": values.tolist(),
            "token_count": len(full_ids), "punctuation_token_mapping": dict(zip(PUNCTUATION, punct_ids))}


def append_cache(record: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_cache(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["doc_id"]] = record
    return records
