"""Evidence-strength ceiling diagnostic on the P1 train cross-fitting set.

Question: how good does the boundary evidence have to be before the frozen
Soft F1 selector (a) ranks Gold first, (b) ranks real systems correctly, and
(c) picks SaT where SaT is the oracle? The candidates, Soft F1 and meta-eval
stay fixed; only the evidence is replaced by hypothetical stronger evidence.

This is an analysis, not a method: the hypothetical evidence is built FROM TRAIN
GOLD, so none of these numbers may be used to choose a configuration. GUM dev
and test are never read.

Scenarios (all start from the held-out-fold calibrated evidence p):
- ``shift_b``: logit(p') = logit(p) + b * (2y - 1) on every gap — a uniform
  amount of extra correct signal, b = inf meaning p' = y (Gold itself).
- ``oracle_<category set>``: p' = y only on gaps whose gold-free left context is
  in the set (optionally only Gold or only non-Gold gaps), original p elsewhere —
  which error class decides selection.
- ``non_terminal shift b``: the logit shift applied to non-terminal gaps only.
Each scenario reports the evidence's own strength (Direct Internal F1 at the
frozen threshold, and accuracy on gaps where the systems disagree) next to the
meta-evaluation numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .analyze_evidence_gaps import left_context_category
from .contracts import read_gold_boundaries, read_raw_documents
from .meta_eval import doc_stats, point_report, record_stats
from .real_splitters import INTRA_TOKEN_REASON
from .selection import ScoredSegmentation


ROOT = Path(__file__).resolve().parents[1]
SHIFTS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, math.inf)
NON_TERMINAL = frozenset({"no_punct_alnum", "no_punct_other", "dash", "colon", "semicolon", "comma"})
ORACLE_SETS = {
    "terminal": frozenset({"terminal"}),
    "non_terminal": NON_TERMINAL,
}
EPS = 1e-6
NON_TERMINAL_SHIFTS = (1.0, 2.0, 3.0, 4.0)


def shift_evidence(evidence: list[float], gold: list[int], shift: float) -> list[float]:
    """Move each probability toward its Gold label by ``shift`` logits (inf = Gold)."""
    if math.isinf(shift):
        return [float(y) for y in gold]
    out = []
    for p, y in zip(evidence, gold, strict=True):
        p = min(max(p, EPS), 1 - EPS)
        z = math.log(p / (1 - p)) + shift * (1 if y else -1)
        out.append(1 / (1 + math.exp(-z)))
    return out


def oracle_on(evidence: list[float], gold: list[int], categories: list[str],
              targets: frozenset[str], shift: float = math.inf,
              labels: frozenset[int] = frozenset({0, 1})) -> list[float]:
    """Apply ``shift_evidence`` only to gaps whose category is in ``targets`` and whose
    Gold label is in ``labels``; every other gap keeps its original evidence."""
    shifted = shift_evidence(evidence, gold, shift)
    return [q if c in targets and y in labels else p
            for p, q, y, c in zip(evidence, shifted, gold, categories, strict=True)]


def disagreement_accuracy(docs: dict, evidence_by_doc: dict[str, list[float]], threshold: float) -> float:
    """Share of system-disagreement gaps where thresholded evidence matches Gold."""
    hits = total = 0
    for doc_id, doc in docs.items():
        evidence = evidence_by_doc[doc_id]
        vectors = list(doc["candidates"].values())
        for i, y in enumerate(doc["gold"]):
            if len({v[i] for v in vectors}) > 1:
                total += 1
                hits += int((evidence[i] >= threshold) == bool(y))
    return hits / total if total else float("nan")


def evaluate_scenario(docs: dict, evidence_by_doc: dict[str, list[float]], fixed_system: str,
                      threshold: float) -> dict:
    stats, doc_rows = [], []
    for doc_id, doc in docs.items():
        evidence = tuple(evidence_by_doc[doc_id])
        records = [
            ScoredSegmentation(doc_id, system, vector, doc["gold"], evidence, non_gap_fp=doc["non_gap"][system])
            for system, vector in sorted(doc["candidates"].items())
        ]
        stats.extend(record_stats(record, doc["domain"]) for record in records)
        doc_rows.append(doc_stats(records[0], doc["domain"], threshold))
    point = point_report(stats, doc_rows, fixed_system)
    soft = point["scorers"]["soft_f1"]
    domains = soft["domains"]
    return {
        "direct_internal_macro_f1": point["direct_internal"]["macro_gold_f1"],
        "disagreement_gap_accuracy": disagreement_accuracy(docs, evidence_by_doc, threshold),
        "gold_win_rate": point["gold_win_rate"],
        "pairwise_accuracy": soft["pairwise"]["accuracy"],
        "domain_mean_kendall_tau": soft["domain_mean"]["kendall_tau"],
        "domain_mean_regret": soft["domain_mean"]["regret"],
        "domain_mean_selected_minus_fixed": soft["domain_mean"]["selected_minus_fixed"],
        "domain_selected_is_oracle": soft["domain_mean"]["selected_is_oracle"],
        "domains_selecting_fixed": sum(d["selected"] == fixed_system for d in domains.values()),
        "corpus_macro_selected": soft["corpus_macro"]["selected"],
        "selected_by_domain": {name: d["selected"] for name, d in domains.items()},
        # Evidence-independent; repeated per scenario as a consistency check.
        "fixed_domain_mean_regret": soft["domain_mean"]["fixed_regret"],
        "system_macro_gold_f1": {name: v["macro_gold_f1"] for name, v in sorted(soft["systems"].items())},
    }


def render(result: dict) -> str:
    fixed = result["fixed_system"]
    header = ("| 情景 | 证据 Direct Internal F1 | 分歧空隙准确率 | Gold-win | Pairwise | 领域平均 τ | "
              f"领域平均 Regret | 所选 − Always-{fixed} | 选中 oracle 的领域比例 | 选 {fixed} 的领域数 | 语料级选择 |")
    lines = [
        "# 证据强度上限诊断（GUM train 交叉拟合）", "",
        "**这是分析，不是方法。** 假想的更强证据由 train Gold 构造，这些数字不能用于选择任何配置。"
        "候选、Soft F1 和元评测都不变，只替换证据。", "",
        f"协议：{result['protocol']}；{result['n_documents']} 篇文档，{result['n_domains']} 个领域；"
        f"Always-{fixed} 的领域平均 Regret 为 {result['fixed_domain_mean_regret']:.4f}。"
        f"作为参照，各系统的 Gold Macro F1：" +
        "，".join(f"{s} {f:.3f}" for s, f in result["system_macro_gold_f1"].items()) + "。", "",
        "- `shift b`：在 logit 上把每个空隙的概率向它的 Gold 标签推 b，b=∞ 时证据等于 Gold。",
        "- `oracle 某类`：只把该类空隙（按左侧上下文分类，见 `evidence_by_context.md`）的证据换成 Gold，"
        "其余保持原样；“只改 Gold / 非 Gold 空隙”进一步把修改限制在一侧。",
        "- `non_terminal 上 shift b`：只在无终止标点的空隙上做 logit shift。", "",
        header, "|" + "---|" * 11,
    ]
    for name, row in result["scenarios"].items():
        lines.append(
            f"| {name} | {row['direct_internal_macro_f1']:.3f} | {row['disagreement_gap_accuracy']:.3f} | "
            f"{row['gold_win_rate']:.3f} | {row['pairwise_accuracy']:.3f} | {row['domain_mean_kendall_tau']:.3f} | "
            f"{row['domain_mean_regret']:.4f} | {row['domain_mean_selected_minus_fixed']:+.4f} | "
            f"{row['domain_selected_is_oracle']:.2f} | {row['domains_selecting_fixed']}/{result['n_domains']} | "
            f"{row['corpus_macro_selected']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_splitters.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--cv-root", type=Path, default=ROOT / "data/real_system/train_cv")
    parser.add_argument("--results", type=Path, default=ROOT / "results/train_cv")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    fixed_system = config["baselines"]["fixed_system"]
    threshold = config["baselines"]["direct_internal_threshold"]
    documents = {doc.doc_id: doc for doc in read_raw_documents(args.data_root / "raw_train.jsonl")}
    gold_offsets = {}
    for record in read_gold_boundaries(args.data_root / "gold_train.jsonl"):
        record.validate_against(documents[record.doc_id])
        gold_offsets[record.doc_id] = set(record.offsets)

    def jsonl(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    docs: dict[str, dict] = {}
    protocols = set()
    for row in jsonl(args.cv_root / "evidence_cv.jsonl"):
        document = documents[row["doc_id"]]
        if row["text_sha256"] != document.sha256:
            raise RuntimeError(f"Evidence text hash mismatch for {row['doc_id']}")
        protocols.add(row["protocol"])
        offsets = row["candidate_offsets"]
        docs[row["doc_id"]] = {
            "domain": document.genre,
            "offsets": offsets,
            "gold": tuple(int(o in gold_offsets[row["doc_id"]]) for o in offsets),
            "evidence": row["calibrated_boundary_probability"],
            "categories": [left_context_category(document.text, o) for o in offsets],
            "candidates": {},
            "non_gap": defaultdict(int),
        }
    if set(docs) != set(documents) or len(protocols) != 1:
        raise RuntimeError("Evidence must cover all train documents under exactly one protocol")
    for row in jsonl(args.cv_root / "frozen_scores_cv.jsonl"):
        doc = docs[row["doc_id"]]
        chosen = set(row["candidate_boundary_offsets"])
        doc["candidates"][row["system"]] = tuple(int(o in chosen) for o in doc["offsets"])
    for failure in jsonl(args.cv_root / "alignment_failures.jsonl"):
        if failure["reason"] != INTRA_TOKEN_REASON:
            raise RuntimeError(f"Unexpected alignment failure record: {failure}")
        docs[failure["doc_id"]]["non_gap"][failure["system"]] += 1
    if any(set(doc["candidates"]) != set(config["systems"]) for doc in docs.values()):
        raise RuntimeError("Every document needs frozen candidates for every configured system")

    scenarios = {}
    for shift in SHIFTS:
        name = "shift ∞（= Gold）" if math.isinf(shift) else f"shift {shift:g}" + ("（实际证据）" if shift == 0 else "")
        evidence = {d: shift_evidence(doc["evidence"], doc["gold"], shift) for d, doc in docs.items()}
        scenarios[name] = evaluate_scenario(docs, evidence, fixed_system, threshold)
    targeted = [(f"oracle {label}", targets, math.inf, frozenset({0, 1})) for label, targets in ORACLE_SETS.items()]
    targeted += [
        ("oracle non_terminal，只改 Gold 空隙", NON_TERMINAL, math.inf, frozenset({1})),
        ("oracle non_terminal，只改非 Gold 空隙", NON_TERMINAL, math.inf, frozenset({0})),
    ]
    targeted += [(f"non_terminal 上 shift {b:g}", NON_TERMINAL, b, frozenset({0, 1})) for b in NON_TERMINAL_SHIFTS]
    for name, targets, shift, labels in targeted:
        evidence = {d: oracle_on(doc["evidence"], doc["gold"], doc["categories"], targets, shift, labels)
                    for d, doc in docs.items()}
        scenarios[name] = evaluate_scenario(docs, evidence, fixed_system, threshold)

    baseline = scenarios["shift 0（实际证据）"]
    for row in scenarios.values():
        if (row["system_macro_gold_f1"] != baseline["system_macro_gold_f1"]
                or abs(row["fixed_domain_mean_regret"] - baseline["fixed_domain_mean_regret"]) > 1e-12):
            raise RuntimeError("Gold-side quantities must not depend on the evidence")
    result = {
        "protocol": protocols.pop(),
        "n_documents": len(docs),
        "n_domains": len({doc["domain"] for doc in docs.values()}),
        "fixed_system": fixed_system,
        "fixed_domain_mean_regret": baseline["fixed_domain_mean_regret"],
        "system_macro_gold_f1": baseline["system_macro_gold_f1"],
        "scenarios": scenarios,
        "note": "Hypothetical evidence is constructed from train Gold; diagnostic only.",
    }

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "evidence_ceiling.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.results / "evidence_ceiling.md").write_text(render(result))
    print(json.dumps({"status": "EVIDENCE_CEILING_COMPLETE", "results": str(args.results)}), flush=True)


if __name__ == "__main__":
    main()
