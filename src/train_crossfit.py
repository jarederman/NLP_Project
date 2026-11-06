"""P1 development set: document-level K-fold cross-fitting on GUM train.

For fold k, the Platt calibrator is fitted only on the Gold of the other folds,
then applied to fold k's raw BERT probabilities; real-system outputs on fold k
are scored with that held-out evidence. All gold-free scores are written to disk
before any meta-evaluation runs, mirroring the dev pipeline's freeze order.

GUM dev and test are never read here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bert_surface import PROTOCOLS
from .contracts import read_candidate_segmentations, read_gold_boundaries, read_raw_documents
from .meta_eval import auxiliary_quality, evaluate
from .real_splitters import INTRA_TOKEN_REASON
from .scoring import PlattCalibrator, soft_fbeta
from .selection import ScoredSegmentation
from .surface_data import load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def assign_folds(domain_by_doc: dict[str, str], n_folds: int, seed: int) -> dict[str, int]:
    """Genre-stratified, seeded document folds (deterministic given ids and seed)."""
    by_domain: dict[str, list[str]] = defaultdict(list)
    for doc_id, domain in domain_by_doc.items():
        by_domain[domain].append(doc_id)
    rng = random.Random(seed)
    folds: dict[str, int] = {}
    offset = 0
    for domain in sorted(by_domain):
        ids = sorted(by_domain[domain])
        rng.shuffle(ids)
        for i, doc_id in enumerate(ids):
            folds[doc_id] = (offset + i) % n_folds
        offset += len(ids)  # rotate so small domains don't all start at fold 0
    return folds


def fit_fold_calibrator(
    raw_by_doc: dict[str, list[float]], labels_by_doc: dict[str, list[int]]
) -> PlattCalibrator:
    """Fit on exactly the documents passed in (the caller passes training folds only)."""
    raw = np.concatenate([np.asarray(raw_by_doc[d], dtype=float) for d in sorted(labels_by_doc)])
    labels = np.concatenate([np.asarray(labels_by_doc[d], dtype=float) for d in sorted(labels_by_doc)])
    return PlattCalibrator().fit(raw, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/real_system/train_cv/predictions")
    parser.add_argument("--failures", type=Path,
                        default=ROOT / "data/real_system/train_cv/alignment_failures.jsonl")
    parser.add_argument("--cache", type=Path,
                        help="Train BERT cache; defaults to data/lm_cache/bert_surface_<version>/train.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/real_system/train_cv")
    parser.add_argument("--results", type=Path, default=ROOT / "results/train_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    protocol = config["bert"]["mask_protocol"]
    cache_path = args.cache or (
        ROOT / "data/lm_cache" / f"bert_surface_{PROTOCOLS[protocol]}" / "train.jsonl"
    )
    seed = config["fixed_seed"]
    systems = config["systems"]
    beta = config["soft_f1_beta"]

    raw_docs = {doc.doc_id: doc for doc in read_raw_documents(args.data_root / "raw_train.jsonl")}
    surfaces = {doc.doc_id: doc for doc in load_surface_jsonl(args.data_root / "surface_train.jsonl")}
    cache = {record["doc_id"]: record for record in read_jsonl(cache_path)}
    if set(raw_docs) != set(surfaces) or not set(raw_docs) <= set(cache):
        raise RuntimeError("Train raw/surface/cache document sets do not line up")
    for doc_id, surface in surfaces.items():
        record = cache[doc_id]
        if (record["text_sha256"] != raw_docs[doc_id].sha256
                or record["candidate_offsets"] != surface.candidate_offsets
                or record["protocol"] != protocol):
            raise RuntimeError(f"Cache record for {doc_id} does not match surface/protocol {protocol}")
    domain_by_doc = {doc_id: surface.genre for doc_id, surface in surfaces.items()}
    folds = assign_folds(domain_by_doc, args.folds, seed)

    predictions: dict[str, dict[str, tuple[int, ...]]] = {}
    for system in systems:
        predictions[system] = {}
        for prediction in read_candidate_segmentations(args.predictions / f"{system}.jsonl"):
            prediction.validate_against(raw_docs[prediction.doc_id])
            predictions[system][prediction.doc_id] = prediction.offsets
        if set(predictions[system]) != set(surfaces):
            raise RuntimeError(f"{system} predictions do not cover every train document")

    # ---- Stage A: gold-free scoring, one fold at a time -----------------------
    gold_contracts = {record.doc_id: record for record in read_gold_boundaries(args.data_root / "gold_train.jsonl")}
    for doc_id, record in gold_contracts.items():
        record.validate_against(raw_docs[doc_id])
    labels_by_doc = {
        doc_id: [int(offset in set(gold_contracts[doc_id].offsets)) for offset in surface.candidate_offsets]
        for doc_id, surface in surfaces.items()
    }
    raw_by_doc = {doc_id: cache[doc_id]["raw_punctuation_probability"] for doc_id in surfaces}

    evidence_rows, score_rows, calibrations = [], [], []
    for fold in range(args.folds):
        training = {d: labels_by_doc[d] for d in surfaces if folds[d] != fold}
        held_out = sorted(d for d in surfaces if folds[d] == fold)
        calibrator = fit_fold_calibrator(raw_by_doc, training)  # held-out Gold never passed in
        calibrations.append({"fold": fold, "training_documents": len(training),
                             "held_out_documents": len(held_out), **calibrator.to_dict()})
        for doc_id in held_out:
            surface = surfaces[doc_id]
            probabilities = calibrator.predict(np.asarray(raw_by_doc[doc_id], dtype=float))
            evidence_rows.append({
                "doc_id": doc_id, "fold": fold, "text_sha256": surface.text_sha256,
                "candidate_offsets": surface.candidate_offsets,
                "calibrated_boundary_probability": probabilities.tolist(),
                "protocol": protocol,
            })
            for system in systems:
                chosen = set(predictions[system][doc_id])
                vector = [int(offset in chosen) for offset in surface.candidate_offsets]
                score_rows.append({
                    "doc_id": doc_id, "fold": fold, "system": system,
                    "candidate_boundary_offsets": sorted(chosen & set(surface.candidate_offsets)),
                    "soft_f1": soft_fbeta(vector, probabilities, beta),
                    "score_protocol": config["protocol_version"],
                })
    write_jsonl(evidence_rows, args.output_root / "evidence_cv.jsonl")
    write_jsonl(score_rows, args.output_root / "frozen_scores_cv.jsonl")
    (args.output_root / "folds.json").write_text(json.dumps(
        {"seed": seed, "n_folds": args.folds, "folds": folds, "calibrations": calibrations}, indent=2) + "\n")
    print(json.dumps({"status": "TRAIN_CV_SCORES_FROZEN", "rows": len(score_rows),
                      "folds": args.folds, "protocol": protocol}), flush=True)

    # ---- Stage B: meta-evaluation against train Gold --------------------------
    non_gap: dict[tuple[str, str], int] = defaultdict(int)
    for failure in read_jsonl(args.failures):
        if failure["reason"] != INTRA_TOKEN_REASON:
            raise RuntimeError(f"Unexpected alignment failure record: {failure}")
        non_gap[(failure["doc_id"], failure["system"])] += 1
    evidence = {row["doc_id"]: row for row in evidence_rows}
    records = []
    for row in score_rows:
        surface = surfaces[row["doc_id"]]
        chosen = set(row["candidate_boundary_offsets"])
        record = ScoredSegmentation(
            row["doc_id"], row["system"],
            tuple(int(offset in chosen) for offset in surface.candidate_offsets),
            tuple(labels_by_doc[row["doc_id"]]),
            tuple(evidence[row["doc_id"]]["calibrated_boundary_probability"]),
            non_gap_fp=non_gap[(row["doc_id"], row["system"])],
        )
        if abs(record.quality - row["soft_f1"]) > 1e-9:
            raise RuntimeError(f"Frozen score changed for {row['doc_id']} {row['system']}")
        records.append(record)

    baselines = config["baselines"]
    result = evaluate(
        records, domain_by_doc, baselines["fixed_system"], baselines["direct_internal_threshold"],
        extra_quality=auxiliary_quality, n_resamples=args.bootstrap, seed=seed,
    )
    result.update({"protocol_version": config["protocol_version"], "mask_protocol": protocol,
                   "split": "UD_English-GUM/train", "n_folds": args.folds})
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_domain_csv(result, args.results / "per_domain.csv")
    (args.results / "summary.md").write_text(render_summary(result))
    print(json.dumps({"status": "TRAIN_CV_META_EVAL_COMPLETE", "results": str(args.results)}), flush=True)


def write_domain_csv(result: dict, path: Path) -> None:
    rows = []
    for scorer, entry in result["point"]["scorers"].items():
        for domain, values in entry["domains"].items():
            rows.append({"scorer": scorer, "domain": domain, **values})
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(result: dict) -> str:
    point, ci = result["point"], result["bootstrap"]
    fixed = result["fixed_system"]
    soft = point["scorers"]["soft_f1"]

    def interval(name: str) -> str:
        v = ci[name]
        return f"{v['point']:.4f} [{v['low']:.4f}, {v['high']:.4f}]"

    lines = [
        f"# GUM Train {result['n_folds']}-fold cross-fitting（P1 开发集）", "",
        f"协议：{result['protocol_version']}（mask：{result['mask_protocol']}）；"
        f"{result['n_documents']} 篇文档，{result['n_domains']} 个领域（genre）。"
        "每折的 Platt 只在其余各折的 Gold 上拟合；GUM dev/test 未读取。", "",
        f"置信区间：按领域分层、按文档重采样的 bootstrap，95% 百分位区间。", "",
        "## 关键指标（Soft F1）", "",
        "| 层次 | 指标 | 估计 [95% CI] |", "|---|---|---|",
        f"| 前提 | Gold-win 率 | {interval('gold_win_rate')} |",
        f"| L1 | Pairwise accuracy | {interval('soft_f1.pairwise_accuracy')} |",
        f"| L2 | 领域平均 Kendall τ | {interval('soft_f1.domain_mean_kendall_tau')} |",
        f"| L3 | 领域平均 Selection Regret | {interval('soft_f1.domain_mean_regret')} |",
        f"| L3 | 领域平均（所选 − Always-{fixed}） | {interval('soft_f1.domain_mean_selected_minus_fixed')} |",
        f"| 参考 | 语料级 Macro Regret | {interval('soft_f1.corpus_macro_regret')} |",
        "", f"领域内选中 oracle 的比例：{soft['domain_mean']['selected_is_oracle']:.2f}；"
        f"Always-{fixed} 的领域平均 regret：{soft['domain_mean']['fixed_regret']:.4f}。", "",
        "## 系统（语料级）", "",
        "| System | Gold Macro F1 | Gold Micro F1 | Mean Soft F1 |", "|---|---:|---:|---:|",
    ]
    for system, values in soft["systems"].items():
        lines.append(f"| {system} | {values['macro_gold_f1']:.4f} | {values['micro_gold_f1']:.4f} | "
                     f"{values['macro_quality']:.4f} |")
    direct = point["direct_internal"]
    lines.append(f"| Direct Internal（p ≥ {result['direct_internal_threshold']}） | "
                 f"{direct['macro_gold_f1']:.4f} | {direct['micro_gold_f1']:.4f} | — |")
    lines += ["", "## 按领域（Soft F1，Macro）", "",
              "| Domain | Docs | Selected | Oracle | Regret | 所选 − fixed | Kendall τ | Direct Internal |",
              "|---|---:|---|---|---:|---:|---:|---:|"]
    for domain, v in soft["domains"].items():
        lines.append(f"| {domain} | {v['documents']} | {v['selected']} | {v['oracle']} | {v['regret']:.4f} | "
                     f"{v['selected_minus_fixed']:+.4f} | {v['kendall_tau']:.2f} | "
                     f"{v['direct_internal_macro_gold_f1']:.4f} |")
    lines += ["", "## 评分函数对照（辅助）", "",
              "| Scorer | Pairwise acc | 领域平均 Regret | 领域平均（所选 − fixed） | 领域平均 τ |", "|---|---|---|---|---|"]
    for scorer in result["scorers"]:
        lines.append(f"| {scorer} | {interval(scorer + '.pairwise_accuracy')} | "
                     f"{interval(scorer + '.domain_mean_regret')} | "
                     f"{interval(scorer + '.domain_mean_selected_minus_fixed')} | "
                     f"{interval(scorer + '.domain_mean_kendall_tau')} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
