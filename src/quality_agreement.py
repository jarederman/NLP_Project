"""Cross-document agreement between a reference-free score and Gold quality.

Primary goal metric (PREREGISTRATION.md, amendment 2): over every (document,
system) item, how well does the gold-free score track Gold F1 *across
documents and domains*, especially when the quality difference is large?

- global Spearman / Pearson between score and Gold F1 over all items;
- pairwise accuracy over all item pairs (including cross-document pairs),
  binned by |delta Gold F1|; quality ties count 0.5, Gold ties are skipped;
- Spearman over (domain, system) cells (mean score vs mean Gold F1);
- within-document pairwise accuracy, for reference.
CIs: documents resampled within domain; scorer differences are paired.

Evidence may be shared by all systems of a document (current protocol) or be
specific to each (document, system) item (candidate-conditioned evidence).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .contracts import read_gold_boundaries, read_raw_documents
from .meta_eval import max_soft_f1
from .real_splitters import INTRA_TOKEN_REASON
from .selection import ScoredSegmentation, non_gap_counts


ROOT = Path(__file__).resolve().parents[1]
GAP_BINS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, math.inf))
LARGE_GAP = 0.05
TIE_DECIMALS = 10


def tie_exact(values) -> np.ndarray:
    """Round to TIE_DECIMALS so that mathematically equal values (e.g. two systems with identical
    boundaries in one document) tie exactly instead of being ordered by floating-point noise
    (erratum 2026-11-19; the noise had lowered global Spearman, most on EWT)."""
    return np.round(np.asarray(values, float), TIE_DECIMALS)


def _neg_bce(record: ScoredSegmentation) -> float:
    eps = 1e-7
    p = np.clip(np.asarray(record.evidence), eps, 1 - eps)
    c = np.asarray(record.candidate)
    return float(np.mean(c * np.log(p) + (1 - c) * np.log1p(-p)))


SCORERS: dict[str, Callable[[ScoredSegmentation], float]] = {
    "soft_f1": lambda r: r.quality,
    "normalized_soft_f1": lambda r: r.quality / max_soft_f1(r.evidence),
    "neg_bce": _neg_bce,
}


@dataclass
class Items:
    doc: np.ndarray
    domain: np.ndarray
    system: np.ndarray
    gold: np.ndarray
    scores: dict[str, np.ndarray] = field(default_factory=dict)


def build_items(records: list[ScoredSegmentation], domain_by_doc: dict[str, str],
                scorers: dict[str, Callable] = SCORERS) -> Items:
    items = Items(
        doc=np.array([r.doc_id for r in records]),
        domain=np.array([domain_by_doc[r.doc_id] for r in records]),
        system=np.array([r.system for r in records]),
        gold=np.array([r.gold_counts.f1 for r in records]),
    )
    for name, fn in scorers.items():
        items.scores[name] = np.array([fn(r) for r in records])
    return items


def pairwise_bins(score: np.ndarray, gold: np.ndarray, same_doc: np.ndarray | None = None) -> dict:
    """All-pairs accuracy by |delta Gold F1| bin. ``same_doc`` restricts to within-doc pairs."""
    dg = gold[:, None] - gold[None, :]
    dq = score[:, None] - score[None, :]
    upper = np.triu(np.ones_like(dg, dtype=bool), k=1)
    valid = upper & (dg != 0)
    if same_doc is not None:
        valid &= same_doc
    credit = np.where(dq == 0, 0.5, (np.sign(dq) == np.sign(dg)).astype(float))
    gap = np.abs(dg)
    out = {}
    for low, high in GAP_BINS:
        mask = valid & (gap >= low) & (gap < high)
        n = int(mask.sum())
        out[f"[{low:g},{high:g})"] = {"pairs": n, "accuracy": float(credit[mask].mean()) if n else float("nan")}
    mask = valid & (gap >= LARGE_GAP)
    out["large"] = {"pairs": int(mask.sum()), "accuracy": float(credit[mask].mean()) if mask.any() else float("nan")}
    out["all"] = {"pairs": int(valid.sum()), "accuracy": float(credit[valid].mean()) if valid.any() else float("nan")}
    return out


def cell_spearman(score: np.ndarray, gold: np.ndarray, domain: np.ndarray, system: np.ndarray) -> float:
    keys = sorted(set(zip(domain.tolist(), system.tolist())))
    ms, mg = [], []
    for d, s in keys:
        mask = (domain == d) & (system == s)
        ms.append(score[mask].mean())
        mg.append(gold[mask].mean())
    return float(spearmanr(ms, mg).statistic)


def statistics(items: Items, idx: np.ndarray | None = None, doc_key: np.ndarray | None = None) -> dict:
    """Key statistics for every scorer on the item subset ``idx`` (resampled copies allowed)."""
    idx = np.arange(len(items.gold)) if idx is None else idx
    gold = tie_exact(items.gold[idx])
    keys = items.doc[idx] if doc_key is None else doc_key
    same_doc = keys[:, None] == keys[None, :]
    out = {}
    for name, values in items.scores.items():
        score = tie_exact(values[idx])
        all_pairs = pairwise_bins(score, gold)
        out[name] = {
            "global_spearman": float(spearmanr(score, gold).statistic),
            "global_pearson": float(pearsonr(score, gold).statistic),
            "large_gap_pairwise": all_pairs["large"]["accuracy"],
            "all_pairwise": all_pairs["all"]["accuracy"],
            "cell_spearman": cell_spearman(score, gold, items.domain[idx], items.system[idx]),
            "within_doc_pairwise": pairwise_bins(score, gold, same_doc)["all"]["accuracy"],
        }
    return out


def _resample_plan(items: Items, n_resamples: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Documents resampled with replacement within each domain; duplicated documents get
    distinct keys so within-document pairs never cross copies."""
    docs = sorted(set(items.doc.tolist()))
    domain_of = {d: items.domain[items.doc == d][0] for d in docs}
    rows_of = {d: np.flatnonzero(items.doc == d) for d in docs}
    by_domain = defaultdict(list)
    for d in docs:
        by_domain[domain_of[d]].append(d)
    rng = np.random.default_rng(seed)
    plan = []
    for _ in range(n_resamples):
        idx, keys = [], []
        for domain in sorted(by_domain):
            pool = by_domain[domain]
            for k, j in enumerate(rng.integers(0, len(pool), size=len(pool))):
                rows = rows_of[pool[j]]
                idx.extend(rows.tolist())
                keys.extend([f"{pool[j]}#{k}"] * len(rows))
        plan.append((np.array(idx), np.array(keys)))
    return plan


_WORKER_ITEMS: Items | None = None


def _worker_init(items: Items) -> None:
    global _WORKER_ITEMS
    _WORKER_ITEMS = items


def _worker_statistics(draw: tuple[np.ndarray, np.ndarray]) -> dict:
    return statistics(_WORKER_ITEMS, draw[0], draw[1])


def evaluate(items: Items, n_resamples: int, seed: int, baseline: str = "soft_f1", workers: int = 1) -> dict:
    """Point estimates, stratified-bootstrap CIs, and paired differences vs ``baseline``.
    The resampling plan depends only on ``seed``, so results do not depend on ``workers``."""
    point = statistics(items)
    detail = {name: {"cross_document_bins": pairwise_bins(tie_exact(values), tie_exact(items.gold))}
              for name, values in items.scores.items()}
    plan = _resample_plan(items, n_resamples, seed)
    if workers > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(workers, _worker_init, (items,)) as pool:
            all_stats = pool.map(_worker_statistics, plan, chunksize=max(1, len(plan) // (4 * workers)))
    else:
        all_stats = [statistics(items, idx, keys) for idx, keys in plan]
    draws: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for stats in all_stats:
        for name, values in stats.items():
            for metric, value in values.items():
                draws[name][metric].append(value)

    def interval(values):
        v = np.array([x for x in values if x == x])
        if not len(v):                                 # statistic undefined in every draw
            return [float("nan"), float("nan")]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    result = {"point": point, "detail": detail, "ci": {}, "baseline": baseline, "paired_vs_baseline": {}}
    for name in items.scores:
        result["ci"][name] = {m: interval(v) for m, v in draws[name].items()}
        if name != baseline:
            result["paired_vs_baseline"][name] = {
                m: interval(np.array(draws[name][m]) - np.array(draws[baseline][m])) for m in draws[name]
            }
    if baseline == "soft_f1":
        result["paired_vs_soft_f1"] = result["paired_vs_baseline"]
    return result


def render(result: dict, title: str, note: str) -> str:
    metrics = [("global_spearman", "全局 Spearman（主）"), ("global_pearson", "全局 Pearson"),
               ("large_gap_pairwise", f"大差距配对（|Δ|≥{LARGE_GAP:g}）"), ("all_pairwise", "全部配对"),
               ("cell_spearman", "领域×系统 Spearman"), ("within_doc_pairwise", "文档内配对")]
    lines = [f"# {title}", "", note, "",
             "| 评分函数 | " + " | ".join(label for _, label in metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for name, values in result["point"].items():
        cells = [f"{values[m]:.3f} [{result['ci'][name][m][0]:.3f}, {result['ci'][name][m][1]:.3f}]" for m, _ in metrics]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    base = result.get("baseline", "soft_f1")
    lines += ["", f"与 {base} 的配对差 [95% CI]：", "",
              "| 评分函数 | " + " | ".join(label for _, label in metrics) + " |", "|---|" + "---|" * len(metrics)]
    for name, diffs in result["paired_vs_baseline"].items():
        cells = [f"{result['point'][name][m] - result['point'][base][m]:+.3f} "
                 f"[{diffs[m][0]:+.3f}, {diffs[m][1]:+.3f}]" for m, _ in metrics]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "跨文档配对按 |ΔGold F1| 分档的准确率：", "",
              "| 评分函数 | " + " | ".join(f"{k}" for k in result["detail"][base]["cross_document_bins"]) + " |",
              "|---|" + "---|" * len(result["detail"][base]["cross_document_bins"])]
    for name, detail in result["detail"].items():
        lines.append(f"| {name} | " + " | ".join(
            f"{v['accuracy']:.3f}（{v['pairs']}）" for v in detail["cross_document_bins"].values()) + " |")
    return "\n".join(lines) + "\n"


def load_items(evidence_path: Path, config_path: Path = ROOT / "configs/real_splitters.json",
               data_root: Path = ROOT / "data/real_system",
               cv_root: Path = ROOT / "data/real_system/train_cv", split: str = "train",
               candidates: Path | None = None, failures: Path | None = None) -> tuple[Items, int]:
    """(document, system) items scored on one evidence file (shared or per-system rows).
    Defaults: the GUM-train development set; ``split``/``candidates``/``failures`` select another."""
    config = json.loads(config_path.read_text())
    systems = sorted(config["systems"])
    documents = {d.doc_id: d for d in read_raw_documents(data_root / f"raw_{split}.jsonl")}
    gold = {}
    for record in read_gold_boundaries(data_root / f"gold_{split}.jsonl"):
        record.validate_against(documents[record.doc_id])
        gold[record.doc_id] = set(record.offsets)

    def jsonl(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    evidence, offsets_of = {}, {}
    for row in jsonl(evidence_path):
        if row["text_sha256"] != documents[row["doc_id"]].sha256:
            raise RuntimeError(f"Evidence hash mismatch for {row['doc_id']}")
        key = (row["doc_id"], row.get("system"))          # system=None: shared by all systems
        evidence[key] = tuple(row["calibrated_boundary_probability"])
        offsets_of[row["doc_id"]] = row["candidate_offsets"]
    chosen = defaultdict(dict)
    for row in jsonl(candidates or cv_root / "frozen_scores_cv.jsonl"):
        chosen[row["doc_id"]][row["system"]] = set(row["candidate_boundary_offsets"])
    non_gap = defaultdict(list)
    for failure in jsonl(failures or cv_root / "alignment_failures.jsonl"):
        if failure["reason"] != INTRA_TOKEN_REASON:
            raise RuntimeError(f"Unexpected alignment failure record: {failure}")
        non_gap[(failure["doc_id"], failure["system"])].append(failure["offset"])

    records = []
    for doc_id in sorted(documents):
        offsets = offsets_of[doc_id]
        labels = tuple(int(o in gold[doc_id]) for o in offsets)
        for system in systems:
            p = evidence.get((doc_id, system), evidence.get((doc_id, None)))
            if p is None:
                raise RuntimeError(f"No evidence for {doc_id}/{system}")
            vector = tuple(int(o in chosen[doc_id][system]) for o in offsets)
            tp, fp, fn = non_gap_counts(non_gap[(doc_id, system)], gold[doc_id] - set(offsets))
            records.append(ScoredSegmentation(doc_id, system, vector, labels, p,
                                              non_gap_fp=fp, non_gap_tp=tp, non_gap_fn=fn))
    return build_items(records, {d: documents[d].genre for d in documents}), len(documents)


def run(evidence_path: Path, name: str, results: Path, bootstrap: int,
        config_path: Path = ROOT / "configs/real_splitters.json",
        data_root: Path = ROOT / "data/real_system", cv_root: Path = ROOT / "data/real_system/train_cv",
        workers: int = 1) -> dict:
    """Evaluate one evidence file (shared or per-system rows) on the GUM-train items."""
    config = json.loads(config_path.read_text())
    items, n_documents = load_items(evidence_path, config_path, data_root, cv_root)
    result = evaluate(items, bootstrap, int(config["fixed_seed"]), workers=workers)
    shown = evidence_path.relative_to(ROOT) if evidence_path.is_relative_to(ROOT) else evidence_path
    n_items = len(items.gold)
    result.update({"evidence": str(shown), "items": n_items, "documents": n_documents})
    out = results / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    note = (f"证据：`{shown}`；{n_documents} 篇 GUM train 文档 × {len(config['systems'])} 个系统 = {n_items} 个条目。"
            "评分函数之间只换打分方式，证据、候选和 Gold 完全相同（控制变量）。")
    (out / "summary.md").write_text(render(result, f"跨文档质量一致性：{name}", note))
    print(json.dumps({"status": "QUALITY_AGREEMENT_COMPLETE", "results": str(out)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_splitters.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--cv-root", type=Path, default=ROOT / "data/real_system/train_cv")
    parser.add_argument("--evidence", type=Path, default=None,
                        help="JSONL with doc_id, [system], calibrated_boundary_probability; "
                             "defaults to the P1 cross-fitted evidence")
    parser.add_argument("--results", type=Path, default=ROOT / "results/quality_agreement")
    parser.add_argument("--name", default="p1_platt")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.evidence or args.cv_root / "evidence_cv.jsonl", args.name, args.results, args.bootstrap,
        args.config, args.data_root, args.cv_root, args.workers)


if __name__ == "__main__":
    main()
