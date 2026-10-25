from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .candidates import generate_candidates
from .data import load_documents, parse_conllu, save_documents
from .evaluate import aggregate, score_dev_documents, select_examples, write_jsonl, write_metrics
from .lm import append_cache, extract_document_probabilities, load_model_and_tokenizer, read_cache
from .report import write_summary
from .postprocess import diagnostics_and_cases
from .scoring import PlattCalibrator, apply_abbreviation_rules, bce, soft_fbeta, weighted_bce


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate_paths() -> None:
    root = str(ROOT.resolve())
    required = ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "XDG_CACHE_HOME", "TMPDIR", "PIP_CACHE_DIR",
                "TORCH_HOME", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "PYTHONPYCACHEPREFIX"]
    bad = []
    for name in required:
        value = os.environ.get(name, "")
        if not value.startswith("/scratch/wj2301/"):
            bad.append(f"{name}={value!r}")
    if bad:
        raise RuntimeError("Scratch cache invariant violated: " + ", ".join(bad))
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError(f"Run from {ROOT}, got {Path.cwd()}")


def prepare_data(limit: int | None = None):
    source = ROOT / "data" / "UD_English-GUM"
    processed = ROOT / "data" / "processed"
    output = {}
    for split in ("train", "dev"):
        conllu = source / f"en_gum-ud-{split}.conllu"
        if not conllu.exists():
            raise FileNotFoundError(f"Missing {conllu}; run scripts/fetch_gum.sh")
        cache_path = processed / f"gum_{split}_documents.jsonl"
        docs = parse_conllu(conllu, split)
        if limit:
            docs = docs[:limit]
        save_documents(docs, cache_path if not limit else processed / f"gum_{split}_smoke.jsonl")
        for doc in docs:
            assert len(doc.units) == len(doc.next_units) == len(doc.gold) == len(doc.char_offsets)
            assert all(doc.text[o:o + 1] == " " for o in doc.char_offsets)
        output[split] = docs
    return output


def rule_and_metric_smoke() -> None:
    units = ["Mr.", "Smith", "Dr.", "Johnson", "U.S.", "government", "U.S.", "He", "3.", "14"]
    p = np.full(len(units), 0.8)
    gold = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    adjusted, reasons, stats = apply_abbreviation_rules(units, p, gold)
    assert adjusted[0] < p[0] and reasons[0] == "title_abbreviation"
    assert adjusted[2] < p[2] and reasons[2] == "title_abbreviation"
    assert adjusted[4] < p[4] and reasons[4] == "acronym_continuation"
    assert adjusted[6] == p[6], "real sentence-final U.S. must not be blindly suppressed"
    assert adjusted[8] < p[8] and reasons[8] == "decimal_split"
    extreme = np.array([0.0, 1.0, 1e-30, 1 - 1e-30])
    assert np.isfinite(bce([0, 1, 1, 0], extreme))
    assert np.isfinite(weighted_bce([0, 1, 1, 0], extreme))
    assert 0 <= soft_fbeta([0, 1, 1, 0], np.clip(extreme, 0, 1)) <= 1
    gold_vec = [0] * 20
    gold_vec[5] = gold_vec[12] = 1
    candidates = generate_candidates(gold_vec, "smoke", 7, [0.1, 0.2])
    assert {c.condition for c in candidates} == {"gold", "over", "under", "shift", "mixed"}
    assert all(len(c.boundaries) == len(gold_vec) for c in candidates)
    print(json.dumps({"smoke_rules": stats.__dict__, "candidate_variants": len(candidates)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preliminary.json")
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    validate_paths()
    config = load_config(args.config)
    if args.mode == "smoke":
        rule_and_metric_smoke()
    docs = prepare_data(limit=1 if args.mode == "smoke" else None)

    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; do not run LM extraction on the login node")
    model_name, model, tokenizer = load_model_and_tokenizer(config["model_candidates"], os.environ["HF_HUB_CACHE"])
    print(f"MODEL={model_name} GPU={torch.cuda.get_device_name(0)}", flush=True)

    # Smoke writes the first records into the same append-only cache reused by formal.
    cache_path = ROOT / "data" / "lm_cache" / "qwen_punctuation.jsonl"
    cache = read_cache(cache_path)
    for split in ("train", "dev"):
        for index, doc in enumerate(docs[split], 1):
            existing = cache.get(doc.doc_id)
            if existing and existing.get("model_name") == model_name and len(existing["raw_punctuation_probability"]) == len(doc.gold):
                continue
            started = time.time()
            record = extract_document_probabilities(model, tokenizer, doc, config["max_context_tokens"], config["context_overlap_tokens"])
            record.update({"model_name": model_name, "split": split})
            append_cache(record, cache_path)
            cache[doc.doc_id] = record
            print(f"EXTRACT {split} {index}/{len(docs[split])} {doc.doc_id} positions={len(doc.gold)} seconds={time.time()-started:.1f}", flush=True)

    train_raw = np.concatenate([cache[d.doc_id]["raw_punctuation_probability"] for d in docs["train"]])
    train_y = np.concatenate([d.gold for d in docs["train"]])
    calibrator = PlattCalibrator().fit(train_raw, train_y)
    if args.mode == "smoke":
        dev = docs["dev"][0]
        raw = np.asarray(cache[dev.doc_id]["raw_punctuation_probability"])
        calibrated = calibrator.predict(raw)
        assert np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
        assert np.isfinite(calibrated).all() and ((calibrated >= 0) & (calibrated <= 1)).all()
        payload = {
            "status": "PASS", "model": model_name, "gpu": torch.cuda.get_device_name(0),
            "train_doc": docs["train"][0].doc_id, "dev_doc": dev.doc_id,
            "dev_positions": len(dev.gold), "raw_min": float(raw.min()), "raw_max": float(raw.max()),
            "calibrator": calibrator.to_dict(),
        }
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "smoke_test.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload), flush=True)
        return

    rows, aux = score_dev_documents(docs["dev"], cache, calibrator, config)
    agg = aggregate(rows)
    examples = select_examples(docs["dev"], rows, aux, config)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    write_metrics(rows, results / "metrics.csv")
    write_jsonl(examples, results / "examples.jsonl")
    rule_totals = {
        key: sum(getattr(v["rule_stats"], key) for v in aux.values())
        for key in ("triggered", "beneficial_nonboundary", "detrimental_boundary", "fixes", "harms")
    }
    metadata = {
        "model_name": model_name,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seed": config["seed"],
        "train_documents": len(docs["train"]),
        "dev_documents": len(docs["dev"]),
        "dev_sentences": sum(d.sentence_count for d in docs["dev"]),
        "dev_candidate_positions": sum(len(d.gold) for d in docs["dev"]),
        "dev_removed_terminal_marks": sum(d.removed_terminal_marks for d in docs["dev"]),
        "rule_totals": rule_totals,
        "calibrator": calibrator.to_dict(),
    }
    diagnostics, cases = diagnostics_and_cases(docs["dev"], cache, calibrator, rows, config)
    metadata.update({"diagnostics": diagnostics, "cases": cases})
    (results / "aggregate.json").write_text(json.dumps(agg, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (results / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (results / "calibration.json").write_text(json.dumps(calibrator.to_dict(), indent=2) + "\n", encoding="utf-8")
    write_summary(results / "summary.md", metadata, agg, rule_totals, calibrator.to_dict())
    print(json.dumps({"status": "COMPLETE", **metadata}), flush=True)


if __name__ == "__main__":
    main()
