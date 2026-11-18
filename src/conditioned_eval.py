"""Compare raw-text vs candidate-conditioned BERT context (CPU stage).

For each context mode written by ``src.conditioned_bert``: fit the two-parameter
Platt calibration per P1 fold on the other folds only (pooling the rows of every
system for the conditioned modes), write the cross-fitted evidence, then run the
cross-document quality-agreement evaluation (``src.quality_agreement``).

Also reports a self-confirmation check: mean calibrated p at gaps where the
candidate itself splits vs. does not, by Gold label. If conditioning on the
candidate's own rendering mainly makes the evidence agree with the candidate,
p at the candidate's non-Gold boundaries rises sharply.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bert_surface import PROTOCOLS, read_cache
from .conditioned_bert import MODES
from .contracts import read_gold_boundaries, read_raw_documents
from .quality_agreement import run as run_agreement
from .scoring import PlattCalibrator


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_splitters.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--cv-root", type=Path, default=ROOT / "data/real_system/train_cv")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/real_system/train_cv/conditioned")
    parser.add_argument("--results", type=Path, default=ROOT / "results/quality_agreement")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    version = PROTOCOLS[config["bert"]["mask_protocol"]]
    cache_root = args.cache_root or ROOT / "data/lm_cache" / f"bert_conditioned_{version}"
    documents = {d.doc_id: d for d in read_raw_documents(args.data_root / "raw_train.jsonl")}
    gold = {}
    for record in read_gold_boundaries(args.data_root / "gold_train.jsonl"):
        record.validate_against(documents[record.doc_id])
        gold[record.doc_id] = set(record.offsets)
    folds = json.loads((args.cv_root / "folds.json").read_text())["folds"]
    chosen: dict[str, dict[str, set[int]]] = defaultdict(dict)
    with (args.cv_root / "frozen_scores_cv.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            chosen[row["doc_id"]][row["system"]] = set(row["candidate_boundary_offsets"])

    summary = {}
    for mode in MODES:
        with (cache_root / mode / "train.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if {r["doc_id"] for r in rows} != set(documents):
            raise RuntimeError(f"{mode}: cache does not cover every train document")
        labels = {r["doc_id"]: np.array([int(o in gold[r["doc_id"]]) for o in r["candidate_offsets"]], dtype=float)
                  for r in rows}
        evidence_rows = []
        for fold in sorted(set(folds.values())):
            train = [r for r in rows if folds[r["doc_id"]] != fold]
            calibrator = PlattCalibrator().fit(
                np.concatenate([r["raw_punctuation_probability"] for r in train]),
                np.concatenate([labels[r["doc_id"]] for r in train]),
            )
            for r in rows:
                if folds[r["doc_id"]] != fold:
                    continue
                p = calibrator.predict(np.asarray(r["raw_punctuation_probability"], dtype=float))
                evidence_rows.append({
                    "doc_id": r["doc_id"], "system": r["system"], "fold": fold,
                    "text_sha256": r["text_sha256"], "candidate_offsets": r["candidate_offsets"],
                    "calibrated_boundary_probability": p.tolist(), "context_mode": mode,
                })
        path = args.output_root / f"evidence_{mode}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in evidence_rows))

        # Self-confirmation: does the evidence follow the candidate it was conditioned on?
        cells = defaultdict(list)
        for r in evidence_rows:
            p = np.asarray(r["calibrated_boundary_probability"])
            y = labels[r["doc_id"]]
            for system in ([r["system"]] if r["system"] else sorted(chosen[r["doc_id"]])):
                c = np.array([o in chosen[r["doc_id"]][system] for o in r["candidate_offsets"]])
                for split, split_name in ((True, "candidate_splits"), (False, "candidate_does_not_split")):
                    for gold_value, gold_name in ((1.0, "gold"), (0.0, "non_gold")):
                        mask = (c == split) & (y == gold_value)
                        cells[f"{split_name}/{gold_name}"].append(p[mask])
        confirmation = {k: float(np.concatenate(v).mean()) for k, v in sorted(cells.items())}
        result = run_agreement(path, f"context_{mode}", args.results, args.bootstrap,
                               args.config, args.data_root, args.cv_root)
        summary[mode] = {"self_confirmation_mean_p": confirmation, "point": result["point"], "ci": result["ci"]}

    lines = ["# BERT 上下文对照：原文 vs 原文+候选 vs 只用候选（GUM train，P1 5 折）", "",
             "三种条件下，每个空隙的查询完全相同（该空隙的终止标点换成 [MASK]，或在空隙插入 [MASK]）；只改变其他空隙的上下文。"
             "每种条件各自做 5 折交叉拟合的两参数 Platt。", "",
             "| 上下文 | 评分 | 全局 Spearman（主） | 大差距配对（|Δ|≥0.05） | 领域×系统 Spearman | 文档内配对 |",
             "|---|---|---|---|---|---|"]
    for mode, entry in summary.items():
        for scorer in ("soft_f1", "normalized_soft_f1"):
            v, c = entry["point"][scorer], entry["ci"][scorer]
            lines.append(f"| {mode} | {scorer} | {v['global_spearman']:.3f} [{c['global_spearman'][0]:.3f}, "
                         f"{c['global_spearman'][1]:.3f}] | {v['large_gap_pairwise']:.3f} | "
                         f"{v['cell_spearman']:.3f} | {v['within_doc_pairwise']:.3f} |")
    lines += ["", "自我印证检查（校准后平均 p）：", "",
              "| 上下文 | 候选切了·Gold | 候选切了·非 Gold | 候选没切·Gold | 候选没切·非 Gold |", "|---|---|---|---|---|"]
    for mode, entry in summary.items():
        s = entry["self_confirmation_mean_p"]
        lines.append(f"| {mode} | {s['candidate_splits/gold']:.3f} | {s['candidate_splits/non_gold']:.3f} | "
                     f"{s['candidate_does_not_split/gold']:.3f} | {s['candidate_does_not_split/non_gold']:.3f} |")
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "context_comparison.md").write_text("\n".join(lines) + "\n")
    (args.results / "context_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "CONDITIONED_EVAL_COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
