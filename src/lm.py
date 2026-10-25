from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np


PUNCTUATION_STRINGS = [".", "?", "!", "。", "？", "！"]


def resolve_punctuation_ids(tokenizer) -> tuple[list[int], dict[str, list[int]]]:
    mapping: dict[str, list[int]] = {}
    ids: set[int] = set()
    for mark in PUNCTUATION_STRINGS:
        encoded = tokenizer(mark, add_special_tokens=False).input_ids
        # Only exact one-token representations are valid next-token alternatives.
        if len(encoded) == 1 and tokenizer.decode(encoded, clean_up_tokenization_spaces=False) == mark:
            ids.add(encoded[0])
            mapping[mark] = encoded
    if not ids:
        raise RuntimeError("Tokenizer exposed no exact one-token terminal punctuation forms")
    return sorted(ids), mapping


def _map_offsets(token_offsets: list[tuple[int, int]], candidate_offsets: list[int]) -> list[int]:
    end_to_token: dict[int, int] = {}
    for idx, (start, end) in enumerate(token_offsets):
        if end > start:
            end_to_token[end] = idx
    missing = [off for off in candidate_offsets if off not in end_to_token]
    if missing:
        raise ValueError(f"Cannot align {len(missing)} candidate character offsets; first={missing[:5]}")
    return [end_to_token[off] for off in candidate_offsets]


def load_model_and_tokenizer(model_names: list[str], cache_dir: str):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    errors = []
    for name in model_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, use_fast=True)
            model = Qwen3_5ForCausalLM.from_pretrained(
                name,
                cache_dir=cache_dir,
                dtype=torch.bfloat16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
            )
            model.eval()
            return name, model, tokenizer
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("All model candidates failed:\n" + "\n".join(errors))


def extract_document_probabilities(model, tokenizer, doc, max_context: int, overlap: int) -> dict:
    """One batched forward per context window, scoring all aligned gaps together."""
    import torch

    encoded = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    positions = _map_offsets(encoded["offset_mapping"], doc.char_offsets)
    punct_ids, mapping = resolve_punctuation_ids(tokenizer)
    result = np.full(len(positions), np.nan, dtype=np.float64)
    position_to_candidates: dict[int, list[int]] = {}
    for candidate_idx, token_idx in enumerate(positions):
        position_to_candidates.setdefault(token_idx, []).append(candidate_idx)

    # Each target region is disjoint. A preceding overlap supplies left context.
    target_width = max_context - overlap
    for target_start in range(0, len(input_ids), target_width):
        target_end = min(len(input_ids), target_start + target_width)
        selected_global = [p for p in position_to_candidates if target_start <= p < target_end]
        if not selected_global:
            continue
        window_start = max(0, target_start - overlap)
        window_end = target_end
        local_positions = torch.tensor([p - window_start for p in selected_global], device="cuda", dtype=torch.long)
        ids = torch.tensor([input_ids[window_start:window_end]], device="cuda", dtype=torch.long)
        with torch.inference_mode():
            outputs = model(input_ids=ids, use_cache=False, logits_to_keep=local_positions)
            logits = outputs.logits[0].float()
            log_denom = torch.logsumexp(logits, dim=-1)
            log_num = torch.logsumexp(logits[:, punct_ids], dim=-1)
            probs = torch.exp(log_num - log_denom).cpu().numpy()
        for global_pos, prob in zip(selected_global, probs.tolist()):
            for candidate_idx in position_to_candidates[global_pos]:
                result[candidate_idx] = prob
        del outputs, logits, ids, local_positions
    if np.isnan(result).any():
        raise RuntimeError(f"Unscored positions in {doc.doc_id}: {np.where(np.isnan(result))[0][:10]}")
    return {
        "doc_id": doc.doc_id,
        "raw_punctuation_probability": result.tolist(),
        "token_count": len(input_ids),
        "punctuation_token_ids": punct_ids,
        "punctuation_token_mapping": mapping,
    }


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
