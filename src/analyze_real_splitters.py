from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .contracts import read_gold_boundaries, read_raw_documents
from .evaluate import spearman
from .selection import ScoredSegmentation, selection_report
from .surface_data import load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pairwise_details(rows: list[dict]) -> dict:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[row["doc_id"]].append(row)
    totals = defaultdict(lambda: {"correct": 0, "ties": 0, "total": 0})
    identical = 0
    for doc_rows in by_doc.values():
        for left, right in itertools.combinations(sorted(doc_rows, key=lambda r: r["system"]), 2):
            pair = f"{left['system']}--{right['system']}"
            if left["candidate_boundary_offsets"] == right["candidate_boundary_offsets"]:
                identical += 1
            if left["gold_f1"] == right["gold_f1"]:
                continue
            stats = totals[pair]
            stats["total"] += 1
            if left["soft_f1"] == right["soft_f1"]:
                stats["ties"] += 1
            elif ((left["soft_f1"] > right["soft_f1"]) ==
                  (left["gold_f1"] > right["gold_f1"])):
                stats["correct"] += 1
    output = {}
    combined = {"correct": 0, "ties": 0, "total": 0}
    for pair, stats in sorted(totals.items()):
        combined = {key: combined[key] + stats[key] for key in combined}
        output[pair] = {
            **stats,
            "accuracy_half_ties": (
                (stats["correct"] + 0.5 * stats["ties"]) / stats["total"]
                if stats["total"] else float("nan")
            ),
            "strict_accuracy": stats["correct"] / stats["total"] if stats["total"] else float("nan"),
        }
    output["ALL"] = {
        **combined,
        "accuracy_half_ties": (
            (combined["correct"] + 0.5 * combined["ties"]) / combined["total"]
            if combined["total"] else float("nan")
        ),
        "strict_accuracy": combined["correct"] / combined["total"] if combined["total"] else float("nan"),
        "identical_output_pairs": identical,
    }
    return output


def build_examples(
    per_doc_rows: list[dict], documents: dict, gold: dict, evidence: dict, limit: int = 24
) -> list[dict]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in per_doc_rows:
        by_doc[row["doc_id"]].append(row)
    candidates = []
    for doc_id, rows in by_doc.items():
        for left, right in itertools.combinations(rows, 2):
            if left["gold_f1"] == right["gold_f1"]:
                continue
            predicted = np.sign(left["soft_f1"] - right["soft_f1"])
            truth = np.sign(left["gold_f1"] - right["gold_f1"])
            incorrect = predicted == 0 or predicted != truth
            candidates.append((
                int(incorrect),
                abs(left["gold_f1"] - right["gold_f1"]),
                doc_id,
                left,
                right,
            ))
    ordering = lambda item: (item[0], item[1], item[2], item[3]["system"], item[4]["system"])
    failures = sorted((item for item in candidates if item[0]), key=ordering, reverse=True)
    successes = sorted((item for item in candidates if not item[0]), key=ordering, reverse=True)
    selected = failures[:max(0, limit - 4)] + successes[:4]
    examples = []
    for incorrect, gap, doc_id, left, right in selected:
        document = documents[doc_id]
        differing = sorted(
            set(left["candidate_boundary_offsets"]) ^ set(right["candidate_boundary_offsets"])
        )
        focus = differing[0] if differing else len(document.text) // 2
        start, end = max(0, focus - 180), min(len(document.text), focus + 180)
        ev = evidence[doc_id]
        probability_by_offset = dict(zip(
            ev["candidate_offsets"], ev["calibrated_boundary_probability"]
        ))
        diagnostic = sorted(set(differing) | set(gold[doc_id].offsets))
        diagnostic = [offset for offset in diagnostic if start <= offset <= end]
        examples.append({
            "selection_reason": "ranking_error" if incorrect else "ranking_success",
            "doc_id": doc_id,
            "genre": document.genre,
            "document_text": document.text,
            "focus_excerpt": document.text[start:end],
            "focus_excerpt_start": start,
            "gold_boundary_offsets": list(gold[doc_id].offsets),
            "systems": {
                left["system"]: {
                    "boundary_offsets": left["candidate_boundary_offsets"],
                    "gold_f1": left["gold_f1"],
                    "soft_f1": left["soft_f1"],
                },
                right["system"]: {
                    "boundary_offsets": right["candidate_boundary_offsets"],
                    "gold_f1": right["gold_f1"],
                    "soft_f1": right["soft_f1"],
                },
            },
            "gold_f1_gap": gap,
            "diagnostic_probabilities": [
                {"offset": offset, "p": probability_by_offset.get(offset),
                 "context": document.text[max(0, offset - 45):min(len(document.text), offset + 45)]}
                for offset in diagnostic
            ],
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--scores", type=Path, default=ROOT / "data/real_system/frozen_scores.jsonl")
    parser.add_argument("--evidence", type=Path, default=ROOT / "data/real_system/bert_dev_evidence.jsonl")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    documents = {doc.doc_id: doc for doc in read_raw_documents(args.data_root / "raw_dev.jsonl")}
    surfaces = {doc.doc_id: doc for doc in load_surface_jsonl(args.data_root / "surface_dev.jsonl")}
    gold_records = read_gold_boundaries(args.data_root / "gold_dev.jsonl")
    gold = {record.doc_id: record for record in gold_records}
    evidence = {record["doc_id"]: record for record in read_jsonl(args.evidence)}
    frozen = read_jsonl(args.scores)

    scored: list[ScoredSegmentation] = []
    per_doc_rows: list[dict] = []
    for row in frozen:
        document = documents[row["doc_id"]]
        gold_record = gold[row["doc_id"]]
        gold_record.validate_against(document)
        surface = surfaces[row["doc_id"]]
        ev = evidence[row["doc_id"]]
        candidate_set = set(row["candidate_boundary_offsets"])
        gold_set = set(gold_record.offsets)
        candidate = tuple(int(offset in candidate_set) for offset in surface.candidate_offsets)
        target = tuple(int(offset in gold_set) for offset in surface.candidate_offsets)
        probabilities = tuple(ev["calibrated_boundary_probability"])
        record = ScoredSegmentation(
            row["doc_id"], row["system"], candidate, target, probabilities
        )
        if not math.isclose(record.quality, row["soft_f1"], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"Frozen score changed for {row['doc_id']} {row['system']}")
        counts = record.gold_counts
        scored.append(record)
        per_doc_rows.append({
            "doc_id": row["doc_id"],
            "genre": document.genre,
            "system": row["system"],
            "candidate_positions": len(candidate),
            "candidate_boundary_count": int(sum(candidate)),
            "gold_boundary_count": int(sum(target)),
            "gold_precision": counts.precision,
            "gold_recall": counts.recall,
            "gold_f1": counts.f1,
            "soft_f1": record.quality,
            "candidate_boundary_offsets": row["candidate_boundary_offsets"],
        })

    report = selection_report(scored, fixed_system="sat")
    pairwise = pairwise_details(per_doc_rows)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in per_doc_rows:
        by_doc[row["doc_id"]].append(row)
    rhos = []
    gold_distinct_docs = soft_distinct_docs = all_identical_docs = 0
    per_doc_rho = {}
    for doc_id, rows in by_doc.items():
        gold_values = [row["gold_f1"] for row in rows]
        soft_values = [row["soft_f1"] for row in rows]
        gold_distinct_docs += len(set(gold_values)) > 1
        soft_distinct_docs += len(set(soft_values)) > 1
        all_identical_docs += len({tuple(row["candidate_boundary_offsets"]) for row in rows}) == 1
        rho = spearman(soft_values, gold_values)
        per_doc_rho[doc_id] = rho
        if not math.isnan(rho):
            rhos.append(rho)
    ranking = {
        "documents": len(by_doc),
        "documents_with_distinct_gold_f1": gold_distinct_docs,
        "documents_with_distinct_soft_f1": soft_distinct_docs,
        "documents_with_all_system_outputs_identical": all_identical_docs,
        "within_document_spearman_eligible": len(rhos),
        "within_document_spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "within_document_spearman_median": float(np.median(rhos)) if rhos else float("nan"),
        "pairwise": pairwise,
        "selection": report,
    }

    summary_rows = []
    for system, values in sorted(report["systems"].items()):
        summary_rows.append({"system": system, **values})
    csv_rows = [{key: value for key, value in row.items() if key != "candidate_boundary_offsets"}
                for row in per_doc_rows]
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(csv_rows, args.results / "real_splitter_per_doc.csv")
    write_csv(summary_rows, args.results / "real_splitter_summary.csv")
    (args.results / "real_splitter_ranking.json").write_text(
        json.dumps(ranking, ensure_ascii=False, indent=2) + "\n"
    )

    examples = build_examples(per_doc_rows, documents, gold, evidence)
    for target in (args.results / "examples.jsonl", args.results / "real_splitter_examples.jsonl"):
        with target.open("w", encoding="utf-8") as out:
            for example in examples:
                out.write(json.dumps(example, ensure_ascii=False) + "\n")

    lines = [
        "# GUM Dev 真实 Sentence Splitter 排名实验", "",
        "本报告仅在四个系统的预测与 BERT Soft F1 分数冻结后读取 GUM Dev Gold。Soft F1 公式未修改。", "",
        "## 系统结果", "",
        "| System | Gold Macro F1 | Gold Micro F1 | Mean Soft F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['system']} | {row['macro_gold_f1']:.4f} | "
            f"{row['micro_gold_f1']:.4f} | {row['macro_quality']:.4f} |"
        )
    overall = pairwise["ALL"]
    lines += [
        "", "## 文档内排序", "",
        f"- 可计算 Spearman 的文档：{len(rhos)}/{len(by_doc)}",
        f"- Mean/Median within-document Spearman：{ranking['within_document_spearman_mean']:.4f} / {ranking['within_document_spearman_median']:.4f}",
        f"- Pairwise accuracy（Soft F1 tie 计 0.5）：{overall['accuracy_half_ties']:.4f}，可判定 pairs={overall['total']}",
        f"- Strict pairwise accuracy：{overall['strict_accuracy']:.4f}",
        f"- 四系统输出完全相同的文档：{all_identical_docs}/{len(by_doc)}",
        "", "### 各系统对", "",
        "| Pair | Accuracy | Strict | Pairs | Ties |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair, values in pairwise.items():
        if pair == "ALL":
            continue
        lines.append(
            f"| {pair} | {values['accuracy_half_ties']:.4f} | "
            f"{values['strict_accuracy']:.4f} | {values['total']} | {values['ties']} |"
        )
    lines += [
        "", "## Corpus-level 选择诊断", "",
        f"Soft F1 选择：**{report['macro']['selected']}**；Gold oracle：**{report['macro']['oracle']}**；"
        f"selection regret={report['macro']['selection_regret']:.4f}。",
        "", "## 解释限制", "",
        "这是 GUM 内域真实系统实验。BERT 的 Platt calibration 使用 GUM Train Gold；Dev Gold 仅用于最终 meta-evaluation。"
        "本轮没有运行 Always-SaT、Direct Internal Splitter 或 EWT，也没有据此调整 Soft F1。",
    ]
    (args.results / "real_splitter_ranking.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "REAL_SPLITTER_ANALYSIS_COMPLETE", **ranking}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
