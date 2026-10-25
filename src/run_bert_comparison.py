from __future__ import annotations

import csv
import argparse
import itertools
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bert_lm import append_cache, extract_document_probabilities, load_bert, read_cache
from .candidates import generate_candidates
from .data import load_documents
from .evaluate import spearman
from .scoring import PlattCalibrator, boundary_f1, soft_fbeta


ROOT = Path(__file__).resolve().parents[1]


def metric_summary(rows: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["doc_id"]].append(row)
    within, correct, total, gold_wins, gold_total = [], 0, 0, 0, 0
    by_condition_rate = {}
    for doc_rows in grouped.values():
        rho = spearman([r[key] for r in doc_rows], [r["gold_boundary_f1"] for r in doc_rows])
        if not math.isnan(rho):
            within.append(rho)
        for a, b in itertools.combinations(doc_rows, 2):
            if a["gold_boundary_f1"] == b["gold_boundary_f1"]:
                continue
            total += 1
            correct += ((a[key] > b[key]) == (a["gold_boundary_f1"] > b["gold_boundary_f1"]))
        gold = next(r for r in doc_rows if r["condition"] == "gold")
        for row in doc_rows:
            if row["condition"] != "gold":
                gold_total += 1
                gold_wins += gold[key] > row[key]
    for condition in ("over", "under", "shift", "mixed"):
        for rate in (0.1, 0.2):
            subset = [r for r in rows if r["condition"] == condition and r["rate"] == rate]
            wins = 0
            for row in subset:
                gold = next(r for r in grouped[row["doc_id"]] if r["condition"] == "gold")
                wins += gold[key] > row[key]
            by_condition_rate[f"{condition}_{rate:g}"] = wins / len(subset)
    return {
        "spearman_global": spearman([r[key] for r in rows], [r["gold_boundary_f1"] for r in rows]),
        "spearman_within_doc_mean": float(np.mean(within)),
        "pairwise_accuracy": correct / total,
        "gold_beats_perturbed": gold_wins / gold_total,
        "gold_wins_by_condition_rate": by_condition_rate,
    }


def main() -> None:
    import torch
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/bert_comparison.json").read_text())
    train_docs = load_documents(ROOT / "data/processed/gum_train_documents.jsonl")
    dev_docs = load_documents(ROOT / "data/processed/gum_dev_documents.jsonl")
    cache_path = ROOT / "data/lm_cache/bert_bidirectional_punctuation.jsonl"
    cache = read_cache(cache_path)
    if not args.analyze_only:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required")
        model, tokenizer, punct_ids, punct_mapping = load_bert(config["model_name"], os.environ["HF_HUB_CACHE"])
        smoke_done = False
        for split, docs in (("train", train_docs), ("dev", dev_docs)):
            for index, doc in enumerate(docs, 1):
                existing = cache.get(doc.doc_id)
                if existing and existing.get("model_name") == config["model_name"] and len(existing["raw_punctuation_probability"]) == len(doc.gold):
                    continue
                started = time.time()
                record = extract_document_probabilities(model, tokenizer, punct_ids, doc, config["max_length"], config["batch_size"])
                record.update({"model_name": config["model_name"], "split": split,
                               "context_definition": "balanced bidirectional token window with one [MASK] at candidate gap"})
                values = np.asarray(record["raw_punctuation_probability"])
                if not smoke_done:
                    assert ((values >= 0) & (values <= 1)).all() and len(values) == len(doc.gold)
                    print(json.dumps({"BERT_SMOKE": "PASS", "doc": doc.doc_id, "positions": len(values),
                                      "min": float(values.min()), "max": float(values.max()),
                                      "punctuation_ids": punct_mapping}), flush=True)
                    smoke_done = True
                append_cache(record, cache_path)
                cache[doc.doc_id] = record
                print(f"BERT_EXTRACT {split} {index}/{len(docs)} {doc.doc_id} positions={len(doc.gold)} seconds={time.time()-started:.1f}", flush=True)
    missing = [d.doc_id for d in train_docs + dev_docs if d.doc_id not in cache]
    if missing:
        raise RuntimeError(f"BERT cache incomplete: {missing[:5]}")

    train_raw = np.concatenate([cache[d.doc_id]["raw_punctuation_probability"] for d in train_docs])
    train_y = np.concatenate([d.gold for d in train_docs])
    calibrator = PlattCalibrator().fit(train_raw, train_y)
    qwen_cache = {}
    with (ROOT / "data/lm_cache/qwen_punctuation.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line); qwen_cache[record["doc_id"]] = record
    qwen_train_raw = np.concatenate([qwen_cache[d.doc_id]["raw_punctuation_probability"] for d in train_docs])
    qwen_calibrator = PlattCalibrator().fit(qwen_train_raw, train_y)
    rows = []
    for doc in dev_docs:
        raw = np.asarray(cache[doc.doc_id]["raw_punctuation_probability"])
        probability = calibrator.predict(raw)
        qwen_probability = qwen_calibrator.predict(np.asarray(qwen_cache[doc.doc_id]["raw_punctuation_probability"]))
        for candidate in generate_candidates(doc.gold, doc.doc_id, config["seed"], config["rates"]):
            key = (doc.doc_id, candidate.condition, candidate.rate)
            rows.append({"doc_id": doc.doc_id, "genre": doc.genre, "condition": candidate.condition,
                         "rate": candidate.rate, "gold_boundary_f1": boundary_f1(candidate.boundaries, doc.gold),
                         "qwen_soft_f1": soft_fbeta(candidate.boundaries, qwen_probability, 1.0),
                         "bert_soft_f1": soft_fbeta(candidate.boundaries, probability, 1.0)})
    summaries = {name: metric_summary(rows, key) for name, key in
                 (("Qwen left-context", "qwen_soft_f1"), ("BERT bidirectional", "bert_soft_f1"))}
    results = ROOT / "results"
    with (results / "bert_vs_qwen_metrics.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NVIDIA H200 (cached extraction Job 20916796)"
    metadata = {
        "bert_model": config["model_name"], "gpu": gpu_name,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "train_documents": len(train_docs), "dev_documents": len(dev_docs),
        "candidate_rows": len(rows), "max_length": config["max_length"], "batch_size": config["batch_size"],
        "bert_calibrator": calibrator.to_dict(), "qwen_calibrator": qwen_calibrator.to_dict(),
        "summaries": summaries,
    }
    (results / "bert_vs_qwen.json").write_text(json.dumps(metadata, indent=2) + "\n")
    q, b = summaries["Qwen left-context"], summaries["BERT bidirectional"]
    lines = ["# BERT 双向上下文 vs Qwen 左侧上下文", "",
             "除概率来源外，GUM 划分、train Platt calibration、候选扰动、Soft F1 公式与评价代码完全相同。", "",
             "| Probability source | 全局 Spearman | 文档内 Spearman | Pairwise accuracy | Gold 胜率 |",
             "|---|---:|---:|---:|---:|",
             f"| Qwen3.5-9B-Base left-context | {q['spearman_global']:.3f} | {q['spearman_within_doc_mean']:.3f} | {q['pairwise_accuracy']:.3f} | {q['gold_beats_perturbed']:.3f} |",
             f"| BERT-base-uncased bidirectional | {b['spearman_global']:.3f} | {b['spearman_within_doc_mean']:.3f} | {b['pairwise_accuracy']:.3f} | {b['gold_beats_perturbed']:.3f} |", "",
             "## Gold 对各扰动的胜率", "",
             "| Source | Over 10/20% | Under 10/20% | Shift 10/20% | Mixed 10/20% |", "|---|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        w = summary["gold_wins_by_condition_rate"]
        p = lambda c: f"{w[f'{c}_0.1']:.3f}/{w[f'{c}_0.2']:.3f}"
        lines.append(f"| {name} | {p('over')} | {p('under')} | {p('shift')} | {p('mixed')} |")
    delta = b["pairwise_accuracy"] - q["pairwise_accuracy"]
    verdict = "优于" if delta > 0 else "未优于"
    lines += ["", "## 结论", "",
              f"BERT 双向概率在本轮 {verdict} Qwen 左侧概率；pairwise accuracy 差值为 {delta:+.3f}。",
              "BERT 在候选位置使用右侧文本，因此可能直接利用下一词大写等强信号；这属于本实验要测的双向上下文优势，但与在线 causal 评价的可用信息不同。"]
    (results / "bert_vs_qwen_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "BERT_COMPARISON_COMPLETE", **metadata}), flush=True)


if __name__ == "__main__":
    main()
