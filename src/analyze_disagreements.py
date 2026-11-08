"""Where do systems disagree, who is right, and does the evidence reward it?

For each configuration of ``src.calibration_search`` and each comparison system,
looks at the gaps where exactly one of (fixed system, other) places a boundary,
split by gold-free left context (terminal unit vs. none). Reports how often that
boundary is Gold, the mean evidence, and how often the evidence exceeds the
Soft F1 break-even F/2 of the system that placed it (i.e. Soft F1 rewards it).

Train only (P2 development set); GUM dev and test are never read.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_evidence_gaps import left_context_category
from .contracts import read_gold_boundaries, read_raw_documents


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_splitters.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--search-root", type=Path, default=ROOT / "data/real_system/train_cv/search")
    parser.add_argument("--scheme", default="logo")
    parser.add_argument("--configs", nargs="+", default=["platt", "B+C+K+A+L+O/hgb"])
    parser.add_argument("--results", type=Path, default=ROOT / "results/calibration_search")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    fixed = config["baselines"]["fixed_system"]
    others = [s for s in sorted(config["systems"]) if s != fixed]
    documents = {d.doc_id: d for d in read_raw_documents(args.data_root / "raw_train.jsonl")}
    gold = {}
    for record in read_gold_boundaries(args.data_root / "gold_train.jsonl"):
        record.validate_against(documents[record.doc_id])
        gold[record.doc_id] = set(record.offsets)
    cache = {}
    with (ROOT / "data/lm_cache/bert_surface_v1.1/train.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            cache[row["doc_id"]] = row["candidate_offsets"]
    chosen = defaultdict(dict)
    with (args.data_root / "train_cv/frozen_scores_cv.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            chosen[row["doc_id"]][row["system"]] = set(row["candidate_boundary_offsets"])
    layout = json.loads((args.search_root / "gap_rows.json").read_text())

    report, lines = {}, [
        "# 系统分歧位置分析（GUM train，" + args.scheme + "）", "",
        f"只看 {fixed} 与另一系统恰有一方切分的空隙。“被奖励”= 证据 p 超过放置该边界的系统自身 Soft F1 的一半"
        "（此时 Soft F1 因这个边界而上升）。", "",
        "| 配置 | 对比 | 只有谁切 | 左侧上下文 | 空隙数 | 其中 Gold | 平均 p | 被奖励 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for name in args.configs:
        probabilities = np.load(args.search_root / args.scheme / f"{name.replace('/', '_')}.npy")
        stats = defaultdict(lambda: [0, 0, 0.0, 0])
        start = 0
        for doc_id, n in zip(layout["doc_ids"], layout["gaps_per_doc"], strict=True):
            p = probabilities[start:start + n]
            start += n
            offsets, text = cache[doc_id], documents[doc_id].text
            if len(offsets) != n:
                raise RuntimeError(f"Gap count mismatch for {doc_id}")
            y = np.array([o in gold[doc_id] for o in offsets])
            vectors = {s: np.array([o in chosen[doc_id][s] for o in offsets]) for s in chosen[doc_id]}
            soft = {s: 2 * (v * p).sum() / (v.sum() + p.sum()) for s, v in vectors.items()}
            for other in others:
                for owner, mask in ((fixed, vectors[fixed] & ~vectors[other]), (other, vectors[other] & ~vectors[fixed])):
                    for i in np.flatnonzero(mask):
                        context = "terminal" if left_context_category(text, offsets[i]) == "terminal" else "non_terminal"
                        entry = stats[(other, owner, context)]
                        entry[0] += 1
                        entry[1] += int(y[i])
                        entry[2] += float(p[i])
                        entry[3] += int(p[i] > soft[owner] / 2)
        report[name] = {}
        for (other, owner, context), (n, g, sp, rewarded) in sorted(stats.items()):
            report[name][f"{fixed}-vs-{other}/{owner}-only/{context}"] = {
                "gaps": n, "gold_share": g / n, "mean_p": sp / n, "rewarded_share": rewarded / n}
            lines.append(f"| {name} | {fixed}–{other} | {owner} | {context} | {n} | {g / n:.1%} | "
                         f"{sp / n:.3f} | {rewarded / n:.1%} |")
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "disagreements.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.results / "disagreements.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "DISAGREEMENT_ANALYSIS_COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
