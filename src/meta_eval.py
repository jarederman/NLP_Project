"""Per-domain meta-evaluation with document-level bootstrap.

Works on cached per-(document, system) counts so that the same code serves GUM
dev, GUM-train cross-fitting and the final EWT run. All Gold-dependent numbers
here are meta-evaluation only; the evaluator's choices depend on ``soft`` counts
(or another gold-free ``quality``) alone.

Conventions match ``selection.py``: Macro = mean of per-document F1, Micro =
F1 of pooled counts, and a selector is always judged with the Gold F1 of the
same aggregation it selected with.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

from .selection import Counts, ScoredSegmentation, hard_counts, threshold_evidence


@dataclass(frozen=True)
class RecordStats:
    doc_id: str
    domain: str
    system: str
    gold: Counts
    soft: Counts
    # Additional gold-free per-document scores (e.g. "bce"); higher is better.
    extra_quality: dict[str, float] = field(default_factory=dict)

    @property
    def gold_f1(self) -> float:
        return self.gold.f1

    def quality(self, scorer: str = "soft_f1") -> float:
        return self.soft.f1 if scorer == "soft_f1" else self.extra_quality[scorer]


@dataclass(frozen=True)
class DocStats:
    """Per-document quantities that do not depend on a candidate system."""

    doc_id: str
    domain: str
    gold_soft_f1: float          # Soft F1 of the Gold segmentation itself
    direct_internal: Counts      # Gold counts of thresholded evidence


def record_stats(
    record: ScoredSegmentation, domain: str, extra_quality: dict[str, float] | None = None
) -> RecordStats:
    return RecordStats(
        record.doc_id, domain, record.system, record.gold_counts, record.soft_counts,
        dict(extra_quality or {}),
    )


def doc_stats(record: ScoredSegmentation, domain: str, threshold: float) -> DocStats:
    gold_soft = ScoredSegmentation(
        record.doc_id, "__gold__", record.gold, record.gold, record.evidence
    ).quality
    direct = hard_counts(threshold_evidence(record.evidence, threshold), record.gold)
    return DocStats(record.doc_id, domain, gold_soft, direct)


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    """Kendall tau-b; NaN when either ranking is constant."""
    concordant = discordant = ties_x = ties_y = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denominator = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)) ** 0.5
    return (concordant - discordant) / denominator if denominator else float("nan")


def _systems_summary(stats: list[RecordStats], scorer: str) -> dict[str, dict[str, float]]:
    by_system: dict[str, list[RecordStats]] = defaultdict(list)
    for item in stats:
        by_system[item.system].append(item)
    summary = {}
    for system, items in sorted(by_system.items()):
        gold_total, soft_total = Counts(), Counts()
        for item in items:
            gold_total += item.gold
            soft_total += item.soft
        summary[system] = {
            "macro_quality": sum(item.quality(scorer) for item in items) / len(items),
            "macro_gold_f1": sum(item.gold_f1 for item in items) / len(items),
            "micro_quality": soft_total.f1 if scorer == "soft_f1" else float("nan"),
            "micro_gold_f1": gold_total.f1,
        }
    return summary


def _rank(summary: dict[str, dict[str, float]], field_name: str) -> list[str]:
    return sorted(summary, key=lambda system: (-summary[system][field_name], system))


def _selection(summary: dict, fixed_system: str, aggregation: str) -> dict:
    quality, gold = f"{aggregation}_quality", f"{aggregation}_gold_f1"
    selected, oracle = _rank(summary, quality)[0], _rank(summary, gold)[0]
    systems = sorted(summary)
    return {
        "selected": selected,
        "oracle": oracle,
        "regret": summary[oracle][gold] - summary[selected][gold],
        "fixed_regret": summary[oracle][gold] - summary[fixed_system][gold],
        "selected_minus_fixed": summary[selected][gold] - summary[fixed_system][gold],
        "kendall_tau": kendall_tau_b(
            [summary[s][quality] for s in systems], [summary[s][gold] for s in systems]
        ),
    }


def pairwise(stats: list[RecordStats], scorer: str = "soft_f1") -> dict[str, float]:
    by_doc: dict[str, list[RecordStats]] = defaultdict(list)
    for item in stats:
        by_doc[item.doc_id].append(item)
    correct = ties = total = 0
    for items in by_doc.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                left, right = items[i], items[j]
                if left.gold_f1 == right.gold_f1:
                    continue
                total += 1
                lq, rq = left.quality(scorer), right.quality(scorer)
                if lq == rq:
                    ties += 1
                elif (lq > rq) == (left.gold_f1 > right.gold_f1):
                    correct += 1
    return {
        "accuracy": (correct + 0.5 * ties) / total if total else float("nan"),
        "pairs": float(total),
    }


def gold_win_rate(stats: list[RecordStats], docs: list[DocStats]) -> float:
    """Share of documents where the Gold segmentation scores >= every system."""
    best: dict[str, float] = defaultdict(lambda: float("-inf"))
    for item in stats:
        best[item.doc_id] = max(best[item.doc_id], item.soft.f1)
    return sum(doc.gold_soft_f1 >= best[doc.doc_id] for doc in docs) / len(docs)


def _direct_internal(docs: list[DocStats]) -> dict[str, float]:
    total = Counts()
    for doc in docs:
        total += doc.direct_internal
    return {
        "macro_gold_f1": sum(doc.direct_internal.f1 for doc in docs) / len(docs),
        "micro_gold_f1": total.f1,
    }


def point_report(
    stats: list[RecordStats], docs: list[DocStats], fixed_system: str,
    scorers: Iterable[str] = ("soft_f1",),
) -> dict:
    """Corpus-level and per-domain numbers for one (possibly resampled) document set."""
    scorers = list(scorers)
    by_domain_stats: dict[str, list[RecordStats]] = defaultdict(list)
    by_domain_docs: dict[str, list[DocStats]] = defaultdict(list)
    for item in stats:
        by_domain_stats[item.domain].append(item)
    for doc in docs:
        by_domain_docs[doc.domain].append(doc)

    report: dict = {"gold_win_rate": gold_win_rate(stats, docs), "scorers": {}}
    report["direct_internal"] = _direct_internal(docs)
    for scorer in scorers:
        summary = _systems_summary(stats, scorer)
        entry = {
            "systems": summary,
            "pairwise": pairwise(stats, scorer),
            "corpus_macro": _selection(summary, fixed_system, "macro"),
            "domains": {},
        }
        if scorer == "soft_f1":
            entry["corpus_micro"] = _selection(summary, fixed_system, "micro")
        for domain in sorted(by_domain_stats):
            domain_summary = _systems_summary(by_domain_stats[domain], scorer)
            entry["domains"][domain] = {
                "documents": len(by_domain_docs[domain]),
                **_selection(domain_summary, fixed_system, "macro"),
                "direct_internal_macro_gold_f1":
                    _direct_internal(by_domain_docs[domain])["macro_gold_f1"],
            }
        domains = list(entry["domains"].values())
        taus = [d["kendall_tau"] for d in domains if d["kendall_tau"] == d["kendall_tau"]]
        entry["domain_mean"] = {
            "regret": sum(d["regret"] for d in domains) / len(domains),
            "fixed_regret": sum(d["fixed_regret"] for d in domains) / len(domains),
            "selected_minus_fixed": sum(d["selected_minus_fixed"] for d in domains) / len(domains),
            "kendall_tau": sum(taus) / len(taus) if taus else float("nan"),
            "selected_is_oracle": sum(d["selected"] == d["oracle"] for d in domains) / len(domains),
        }
        report["scorers"][scorer] = entry
    return report


# Scalar statistics that get bootstrap confidence intervals.
def _key_statistics(report: dict) -> dict[str, float]:
    values = {"gold_win_rate": report["gold_win_rate"]}
    for scorer, entry in report["scorers"].items():
        values[f"{scorer}.pairwise_accuracy"] = entry["pairwise"]["accuracy"]
        values[f"{scorer}.corpus_macro_regret"] = entry["corpus_macro"]["regret"]
        values[f"{scorer}.corpus_macro_selected_minus_fixed"] = entry["corpus_macro"]["selected_minus_fixed"]
        values[f"{scorer}.domain_mean_regret"] = entry["domain_mean"]["regret"]
        values[f"{scorer}.domain_mean_selected_minus_fixed"] = entry["domain_mean"]["selected_minus_fixed"]
        values[f"{scorer}.domain_mean_kendall_tau"] = entry["domain_mean"]["kendall_tau"]
    return values


def bootstrap(
    stats: list[RecordStats], docs: list[DocStats], fixed_system: str,
    scorers: Iterable[str] = ("soft_f1",), n_resamples: int = 2000, seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Percentile CIs, resampling documents with replacement within each domain.

    Resampled copies get fresh document ids so a document drawn twice forms two
    independent within-document groups instead of being merged.
    """
    scorers = list(scorers)
    stats_by_doc: dict[str, list[RecordStats]] = defaultdict(list)
    for item in stats:
        stats_by_doc[item.doc_id].append(item)
    docs_by_domain: dict[str, list[DocStats]] = defaultdict(list)
    for doc in docs:
        docs_by_domain[doc.domain].append(doc)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for b in range(n_resamples):
        sample_stats: list[RecordStats] = []
        sample_docs: list[DocStats] = []
        for domain in sorted(docs_by_domain):
            pool = docs_by_domain[domain]
            for k in range(len(pool)):
                doc = pool[rng.randrange(len(pool))]
                new_id = f"{doc.doc_id}#{b}.{k}"
                sample_docs.append(replace(doc, doc_id=new_id))
                sample_stats.extend(replace(item, doc_id=new_id) for item in stats_by_doc[doc.doc_id])
        for name, value in _key_statistics(point_report(sample_stats, sample_docs, fixed_system, scorers)).items():
            if value == value:  # drop NaN draws (e.g. all-tied Kendall)
                draws[name].append(value)
    point = _key_statistics(point_report(stats, docs, fixed_system, scorers))
    intervals = {}
    for name, value in point.items():
        values = sorted(draws[name])
        if not values:
            intervals[name] = {"point": value, "low": float("nan"), "high": float("nan")}
            continue
        low = values[int((alpha / 2) * (len(values) - 1))]
        high = values[int((1 - alpha / 2) * (len(values) - 1))]
        intervals[name] = {"point": value, "low": low, "high": high, "draws": float(len(values))}
    return intervals


def evaluate(
    records: list[ScoredSegmentation], domain_by_doc: dict[str, str], fixed_system: str,
    direct_threshold: float, extra_quality: Callable[[ScoredSegmentation], dict[str, float]] | None = None,
    n_resamples: int = 2000, seed: int = 0,
) -> dict:
    """Full meta-evaluation: point estimates per scorer and domain, plus bootstrap CIs."""
    stats = [
        record_stats(record, domain_by_doc[record.doc_id], extra_quality(record) if extra_quality else None)
        for record in records
    ]
    first: dict[str, ScoredSegmentation] = {}
    for record in records:
        first.setdefault(record.doc_id, record)
    docs = [doc_stats(record, domain_by_doc[doc_id], direct_threshold) for doc_id, record in first.items()]
    scorers = ["soft_f1"] + sorted(stats[0].extra_quality) if stats else ["soft_f1"]
    return {
        "point": point_report(stats, docs, fixed_system, scorers),
        "bootstrap": bootstrap(stats, docs, fixed_system, scorers, n_resamples, seed),
        "n_documents": len(docs),
        "n_domains": len({doc.domain for doc in docs}),
        "fixed_system": fixed_system,
        "direct_internal_threshold": direct_threshold,
        "scorers": scorers,
    }


def max_soft_f1(evidence: tuple[float, ...]) -> float:
    """Exact maximum Soft F1 over all candidates: the optimum is a top-k prefix by p."""
    ordered = sorted(evidence, reverse=True)
    total = sum(evidence)
    best, running = 0.0, 0.0
    for k, probability in enumerate(ordered, 1):
        running += probability
        best = max(best, 2 * running / (k + total))
    return best if total else 1.0


def auxiliary_quality(record: ScoredSegmentation) -> dict[str, float]:
    """Gold-free comparison scorers (higher is better). Soft F1 stays primary."""
    import math

    eps = 1e-7
    log_likelihood = 0.0
    for chosen, probability in zip(record.candidate, record.evidence):
        p = min(max(probability, eps), 1 - eps)
        log_likelihood += math.log(p) if chosen else math.log1p(-p)
    maximum = max_soft_f1(record.evidence)
    return {
        "neg_bce": log_likelihood / len(record.candidate),
        "normalized_soft_f1": record.quality / maximum if maximum else float("nan"),
    }
