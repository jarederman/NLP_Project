"""Error analysis of the cross-fitted BERT evidence on GUM train (P1 dev set).

Groups every candidate gap by its gold-free left context (terminal unit, no
punctuation, dash, colon, ...) and reports, separately for Gold and non-Gold
gaps, how much calibrated boundary probability the evidence assigns. This shows
which kinds of Gold boundaries Soft F1 cannot "see" (p below the F/2 break-even).

Reads only GUM train artifacts and the held-out-fold evidence written by
``src.train_crossfit``; GUM dev and test are never read.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .bert_surface import TRAILING_PUNCTUATION_V11
from .contracts import read_gold_boundaries, read_raw_documents


ROOT = Path(__file__).resolve().parents[1]
LEFT_MARKS = {":": "colon", ";": "semicolon", ",": "comma",
              "-": "dash", "–": "dash", "—": "dash"}
CATEGORY_ORDER = ("terminal", "no_punct_alnum", "no_punct_other", "dash",
                  "colon", "semicolon", "comma")


def left_context_category(text: str, offset: int) -> str:
    """Gold-free category of the text immediately left of a gap."""
    left = text[:offset].rstrip()
    if TRAILING_PUNCTUATION_V11.search(left):
        return "terminal"
    last = left[-1:]
    if last in LEFT_MARKS:
        return LEFT_MARKS[last]
    return "no_punct_alnum" if last.isalnum() else "no_punct_other"


def right_is_capitalized(text: str, offset: int) -> bool:
    return text[offset:].lstrip()[:1].isupper()


def summarize(rows: list[dict]) -> dict:
    """rows: dicts with keys category, gold (bool), p (float), right_cap (bool)."""
    buckets: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["category"], row["gold"])].append(row)
    total_gold = sum(1 for row in rows if row["gold"])
    categories = sorted({row["category"] for row in rows},
                        key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else len(CATEGORY_ORDER))
    out = {}
    for category in categories:
        entry = {}
        for is_gold, name in ((True, "gold"), (False, "non_gold")):
            items = buckets.get((category, is_gold), [])
            n = len(items)
            entry[name] = {
                "count": n,
                "mean_p": sum(r["p"] for r in items) / n if n else None,
                "share_p_below_0.5": sum(r["p"] < 0.5 for r in items) / n if n else None,
                "share_right_capitalized": sum(r["right_cap"] for r in items) / n if n else None,
            }
        gold_n, non_gold_n = entry["gold"]["count"], entry["non_gold"]["count"]
        entry["share_of_gold"] = gold_n / total_gold if total_gold else None
        entry["gold_base_rate"] = gold_n / (gold_n + non_gold_n) if gold_n + non_gold_n else None
        out[category] = entry
    return {"total_gold_boundaries": total_gold, "total_gaps": len(rows), "categories": out}


def render(summary: dict, protocol: str) -> str:
    def fmt(value, pct=False):
        if value is None:
            return "—"
        return f"{value:.1%}" if pct else f"{value:.3f}"

    lines = [
        "# 交叉拟合证据的误差分析（GUM train）",
        "",
        f"协议：{protocol}。证据来自 `src.train_crossfit` 的 held-out fold（每篇文档的 Platt 未见过该文档 Gold）。"
        "分类只看候选空隙左侧文本，不使用 Gold。Soft F1 中一个边界只有在 p > F/2（约 0.37）时才划算。",
        "",
        f"Gold 边界总数 {summary['total_gold_boundaries']}，候选空隙总数 {summary['total_gaps']}。",
        "",
        "| 左侧上下文 | 占 Gold | 该类 Gold 基率 | Gold 平均 p | Gold p<0.5 | Gold 右侧大写 | 非 Gold 数 | 非 Gold 平均 p | 非 Gold 右侧大写 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, entry in summary["categories"].items():
        gold, non_gold = entry["gold"], entry["non_gold"]
        lines.append(
            f"| {category} | {fmt(entry['share_of_gold'], True)} | {fmt(entry['gold_base_rate'], True)} | "
            f"{fmt(gold['mean_p'])} | {fmt(gold['share_p_below_0.5'], True)} | "
            f"{fmt(gold['share_right_capitalized'], True)} | {non_gold['count']} | "
            f"{fmt(non_gold['mean_p'])} | {fmt(non_gold['share_right_capitalized'], True)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--evidence", type=Path, default=ROOT / "data/real_system/train_cv/evidence_cv.jsonl")
    parser.add_argument("--results", type=Path, default=ROOT / "results/train_cv")
    args = parser.parse_args()

    documents = {doc.doc_id: doc for doc in read_raw_documents(args.data_root / "raw_train.jsonl")}
    gold = {}
    for record in read_gold_boundaries(args.data_root / "gold_train.jsonl"):
        record.validate_against(documents[record.doc_id])
        gold[record.doc_id] = set(record.offsets)

    rows, protocols, covered = [], set(), set()
    with args.evidence.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            evidence = json.loads(line)
            document = documents[evidence["doc_id"]]
            if evidence["text_sha256"] != document.sha256:
                raise RuntimeError(f"Evidence text hash mismatch for {evidence['doc_id']}")
            protocols.add(evidence["protocol"])
            covered.add(evidence["doc_id"])
            for offset, p in zip(evidence["candidate_offsets"], evidence["calibrated_boundary_probability"],
                                 strict=True):
                rows.append({
                    "category": left_context_category(document.text, offset),
                    "gold": offset in gold[evidence["doc_id"]],
                    "p": float(p),
                    "right_cap": right_is_capitalized(document.text, offset),
                })
    if covered != set(documents) or set(gold) != set(documents) or len(protocols) != 1:
        raise RuntimeError("Evidence must cover all train documents under exactly one protocol")

    summary = summarize(rows)
    protocol = protocols.pop()
    summary["protocol"] = protocol
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "evidence_by_context.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results / "evidence_by_context.md").write_text(render(summary, protocol))
    print(json.dumps({"status": "EVIDENCE_GAP_ANALYSIS_COMPLETE", "results": str(args.results)}), flush=True)


if __name__ == "__main__":
    main()
