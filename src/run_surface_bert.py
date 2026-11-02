from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .bert_surface import (
    extract_surface_probabilities,
    load_surface_bert,
    read_cache,
)
from .contracts import read_gold_boundaries, read_raw_documents
from .scoring import PlattCalibrator
from .surface_data import load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]


def append_record(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data/lm_cache/bert_surface_v1")
    parser.add_argument("--evidence", type=Path, default=ROOT / "data/real_system/bert_dev_evidence.jsonl")
    parser.add_argument("--calibration", type=Path, default=ROOT / "results/real_calibration.json")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Surface BERT extraction requires a GPU")
    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    bert_config = config["bert"]
    surfaces = {
        split: load_surface_jsonl(args.data_root / f"surface_{split}.jsonl")
        for split in ("train", "dev")
    }
    raw = {
        split: {doc.doc_id: doc for doc in read_raw_documents(args.data_root / f"raw_{split}.jsonl")}
        for split in ("train", "dev")
    }

    cache: dict[str, dict[str, dict]] = {}
    missing = []
    for split in ("train", "dev"):
        cache_path = args.cache_root / f"{split}.jsonl"
        cache[split] = read_cache(cache_path)
        for doc in surfaces[split]:
            record = cache[split].get(doc.doc_id)
            valid = (
                record is not None
                and record.get("text_sha256") == doc.text_sha256
                and record.get("protocol") == bert_config["mask_protocol"]
                and record.get("model_name") == bert_config["model_name"]
                and record.get("candidate_offsets") == doc.candidate_offsets
            )
            if not valid:
                missing.append((split, doc, cache_path))

    if missing:
        model, tokenizer, punct_ids, punct_mapping = load_surface_bert(
            bert_config["model_name"], os.environ["HF_HUB_CACHE"]
        )
        for index, (split, doc, cache_path) in enumerate(missing, 1):
            if raw[split][doc.doc_id].sha256 != doc.text_sha256:
                raise RuntimeError(f"Raw/surface checksum mismatch for {doc.doc_id}")
            record = extract_surface_probabilities(
                model, tokenizer, punct_ids, doc,
                bert_config["max_length"], bert_config["batch_size"],
            )
            record.update({
                "split": split,
                "model_name": bert_config["model_name"],
                "punctuation_token_mapping": punct_mapping,
            })
            append_record(record, cache_path)
            cache[split][doc.doc_id] = record
            print(json.dumps({
                "stage": "BERT_EXTRACT",
                "index": index,
                "total": len(missing),
                "split": split,
                "doc_id": doc.doc_id,
                "positions": len(doc.candidate_offsets),
            }), flush=True)

    train_gold_records = read_gold_boundaries(args.data_root / "gold_train.jsonl")
    train_gold = {record.doc_id: record for record in train_gold_records}
    train_raw: list[float] = []
    train_labels: list[int] = []
    for doc in surfaces["train"]:
        gold = train_gold[doc.doc_id]
        gold.validate_against(raw["train"][doc.doc_id])
        boundary_set = set(gold.offsets)
        train_raw.extend(cache["train"][doc.doc_id]["raw_punctuation_probability"])
        train_labels.extend(int(offset in boundary_set) for offset in doc.candidate_offsets)
    calibrator = PlattCalibrator().fit(
        np.asarray(train_raw, dtype=float), np.asarray(train_labels, dtype=float)
    )

    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("w", encoding="utf-8") as out:
        for doc in surfaces["dev"]:
            record = cache["dev"][doc.doc_id]
            probabilities = calibrator.predict(
                np.asarray(record["raw_punctuation_probability"], dtype=float)
            )
            evidence = {
                "doc_id": doc.doc_id,
                "text_sha256": doc.text_sha256,
                "candidate_offsets": doc.candidate_offsets,
                "raw_punctuation_probability": record["raw_punctuation_probability"],
                "calibrated_boundary_probability": probabilities.tolist(),
                "mask_modes": record["mask_modes"],
                "mask_anchors": record["mask_anchors"],
                "model_name": bert_config["model_name"],
                "protocol": bert_config["mask_protocol"],
            }
            out.write(json.dumps(evidence, ensure_ascii=False) + "\n")

    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    calibration = {
        **calibrator.to_dict(),
        "fit_split": "UD_English-GUM/train",
        "target_split_read": False,
        "training_positions": len(train_labels),
        "training_boundaries": int(sum(train_labels)),
        "model_name": bert_config["model_name"],
        "mask_protocol": bert_config["mask_protocol"],
        "gpu": torch.cuda.get_device_name(0),
    }
    args.calibration.write_text(json.dumps(calibration, indent=2) + "\n")
    print(json.dumps({"status": "SURFACE_BERT_COMPLETE", **calibration}), flush=True)


if __name__ == "__main__":
    main()
