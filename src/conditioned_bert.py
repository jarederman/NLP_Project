"""Candidate-conditioned BERT evidence: what context should the [MASK] query see?

The query at each gap is identical in every condition (the gap's terminal mark,
if any, is replaced by one [MASK]; otherwise [MASK] is inserted). Only the
*context* — every other gap — changes:

  raw                  original text (the current protocol; shared by all systems)
  raw_plus_candidate   original text, plus "." at the candidate's boundaries that
                       have no terminal mark (e.g. headings it splits off)
  candidate_only       terminal marks follow the candidate only: kept or added
                       where it splits, removed where it does not (e.g. "Dr." -> "Dr")

Non-terminal punctuation (commas, colons, quotes, brackets) is always kept.
Each candidate is therefore scored on evidence computed in its own rendering.
Train split only (P1/P2 development set); GUM dev and test are not read.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bert_surface import (
    PROTOCOLS, load_surface_bert, masked_punctuation_probabilities, plan_queries, read_cache, write_cache,
)
from .surface_data import load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]
MODES = ("raw", "raw_plus_candidate", "candidate_only")


def render_candidate(full_ids: list[int], plans: list[tuple[str, int, int, int]], vector: tuple[int, ...] | None,
                     mode: str, period_id: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Rebuild the token sequence with every gap slot rendered for ``mode`` and return
    the new ids plus each gap's slot (the range one [MASK] will replace)."""
    if mode not in MODES:
        raise ValueError(mode)
    if mode != "raw" and (vector is None or len(vector) != len(plans)):
        raise ValueError("candidate-conditioned modes need one 0/1 decision per gap")
    order = sorted(range(len(plans)), key=lambda g: (plans[g][2], plans[g][3]))
    ids: list[int] = []
    slots: list[tuple[int, int] | None] = [None] * len(plans)
    cursor = 0
    for g in order:
        start, end = plans[g][2], plans[g][3]
        if start < cursor:
            raise ValueError("gap slots overlap")
        ids.extend(full_ids[cursor:start])
        original = full_ids[start:end]
        if mode == "raw":
            content = original
        elif mode == "raw_plus_candidate":
            content = original if original or not vector[g] else [period_id]
        else:  # candidate_only
            content = (original or [period_id]) if vector[g] else []
        slot_start = len(ids)
        ids.extend(content)
        slots[g] = (slot_start, len(ids))
        cursor = end
    ids.extend(full_ids[cursor:])
    return ids, slots  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_splitters.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--cv-root", type=Path, default=ROOT / "data/real_system/train_cv")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    bert = config["bert"]
    protocol = bert["mask_protocol"]
    version = PROTOCOLS[protocol]
    cache_root = args.cache_root or ROOT / "data/lm_cache" / f"bert_conditioned_{version}"
    reference = read_cache(ROOT / "data/lm_cache" / f"bert_surface_{version}" / "train.jsonl")
    surfaces = sorted(load_surface_jsonl(args.data_root / "surface_train.jsonl"), key=lambda d: d.doc_id)
    if args.limit:
        surfaces = surfaces[:args.limit]
    chosen: dict[str, dict[str, set[int]]] = defaultdict(dict)
    with (args.cv_root / "frozen_scores_cv.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            chosen[row["doc_id"]][row["system"]] = set(row["candidate_boundary_offsets"])
    systems = sorted(config["systems"])

    model, tokenizer, punct_ids, _ = load_surface_bert(bert["model_name"], os.environ["HF_HUB_CACHE"])
    period_id = tokenizer.convert_tokens_to_ids(".")
    started = time.time()
    rows: dict[str, list[dict]] = {mode: [] for mode in MODES}
    max_raw_diff = 0.0
    for index, doc in enumerate(surfaces, 1):
        full_ids, plans = plan_queries(tokenizer, doc, protocol)
        cached = reference[doc.doc_id]
        if cached["text_sha256"] != doc.text_sha256 or cached["candidate_offsets"] != doc.candidate_offsets:
            raise RuntimeError(f"Reference cache does not match {doc.doc_id}")
        for mode in MODES:
            for system in (systems if mode != "raw" else [None]):
                vector = None if system is None else tuple(
                    int(o in chosen[doc.doc_id][system]) for o in doc.candidate_offsets)
                ids, slots = render_candidate(full_ids, plans, vector, mode, period_id)
                values = masked_punctuation_probabilities(
                    model, tokenizer, punct_ids, ids, slots, bert["max_length"], bert["batch_size"])
                if mode == "raw":
                    max_raw_diff = max(max_raw_diff, float(np.max(np.abs(
                        values - np.asarray(cached["raw_punctuation_probability"])))))
                rows[mode].append({
                    "doc_id": doc.doc_id, "system": system, "text_sha256": doc.text_sha256,
                    "candidate_offsets": doc.candidate_offsets, "raw_punctuation_probability": values.tolist(),
                    "context_mode": mode, "protocol": protocol, "model_name": bert["model_name"],
                })
        if index % 20 == 0 or index == len(surfaces):
            print(json.dumps({"status": "CONDITIONED_PROGRESS", "documents": index,
                              "seconds": round(time.time() - started, 1)}), flush=True)
    if max_raw_diff > 1e-3:
        raise RuntimeError(f"Refactored raw extraction disagrees with the frozen cache: {max_raw_diff}")
    for mode, mode_rows in rows.items():
        write_cache(mode_rows, cache_root / mode / "train.jsonl")
    print(json.dumps({"status": "CONDITIONED_BERT_COMPLETE", "documents": len(surfaces),
                      "max_raw_diff_vs_cache": max_raw_diff, "seconds": round(time.time() - started, 1),
                      "cache_root": str(cache_root)}), flush=True)


if __name__ == "__main__":
    main()
