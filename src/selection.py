"""Aligned macro/micro evaluation for real sentence-splitter outputs."""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Counts:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 1.0


@dataclass(frozen=True)
class ScoredSegmentation:
    """One system output projected onto a shared candidate-position vector."""

    doc_id: str
    system: str
    candidate: tuple[int, ...]
    gold: tuple[int, ...]
    evidence: tuple[float, ...]
    # Splitter boundaries that are not a canonical inter-token whitespace gap:
    # inside a surface token (e.g. an e-mail address) or between tokens with no
    # whitespace (e.g. "al.|:"). The gold-free scorer has no evidence there, and
    # Gold boundaries only occur at whitespace gaps, so meta-evaluation charges
    # each one as a hard false positive.
    non_gap_fp: int = 0

    def __post_init__(self) -> None:
        lengths = {len(self.candidate), len(self.gold), len(self.evidence)}
        if len(lengths) != 1 or not self.candidate:
            raise ValueError("Candidate, gold, and evidence must have equal non-zero length")
        if any(value not in (0, 1) for value in (*self.candidate, *self.gold)):
            raise ValueError("Candidate and gold vectors must be binary")
        if any(probability < 0.0 or probability > 1.0 for probability in self.evidence):
            raise ValueError("Evidence probabilities must be in [0, 1]")
        if self.non_gap_fp < 0:
            raise ValueError("non_gap_fp must be non-negative")

    @property
    def gold_counts(self) -> Counts:
        counts = hard_counts(self.candidate, self.gold)
        return Counts(counts.tp, counts.fp + self.non_gap_fp, counts.fn)

    @property
    def soft_counts(self) -> Counts:
        return expected_counts(self.candidate, self.evidence)

    @property
    def gold_f1(self) -> float:
        return self.gold_counts.f1

    @property
    def quality(self) -> float:
        return self.soft_counts.f1


def hard_counts(candidate: tuple[int, ...], gold: tuple[int, ...]) -> Counts:
    if len(candidate) != len(gold):
        raise ValueError("Candidate and gold vectors must have equal length")
    tp = sum(predicted == 1 and target == 1 for predicted, target in zip(candidate, gold))
    fp = sum(predicted == 1 and target == 0 for predicted, target in zip(candidate, gold))
    fn = sum(predicted == 0 and target == 1 for predicted, target in zip(candidate, gold))
    return Counts(float(tp), float(fp), float(fn))


def expected_counts(candidate: tuple[int, ...], evidence: tuple[float, ...]) -> Counts:
    if len(candidate) != len(evidence):
        raise ValueError("Candidate and evidence vectors must have equal length")
    tp = sum(predicted * probability for predicted, probability in zip(candidate, evidence))
    fp = sum(predicted * (1.0 - probability) for predicted, probability in zip(candidate, evidence))
    fn = sum((1 - predicted) * probability for predicted, probability in zip(candidate, evidence))
    return Counts(tp, fp, fn)


def threshold_evidence(evidence: tuple[float, ...], threshold: float) -> tuple[int, ...]:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("Threshold must be in [0, 1]")
    return tuple(int(probability >= threshold) for probability in evidence)


def direct_internal_summary(
    records: list[ScoredSegmentation], threshold: float
) -> dict[str, float]:
    """Gold quality of splitting directly with the evaluator's own evidence.

    One evidence/gold vector per document is taken from the scored records (they
    are validated to be identical across systems), thresholded, and evaluated
    with the same Macro and pooled-Micro conventions as the real systems.
    """
    _validate_records(records)
    first_by_doc: dict[str, ScoredSegmentation] = {}
    for record in records:
        first_by_doc.setdefault(record.doc_id, record)
    total = Counts()
    f1s = []
    for record in first_by_doc.values():
        counts = hard_counts(threshold_evidence(record.evidence, threshold), record.gold)
        total += counts
        f1s.append(counts.f1)
    return {
        "threshold": threshold,
        "documents": float(len(f1s)),
        "macro_gold_f1": sum(f1s) / len(f1s),
        "micro_gold_f1": total.f1,
        "micro_gold_precision": total.precision,
        "micro_gold_recall": total.recall,
    }


def _validate_records(records: list[ScoredSegmentation]) -> None:
    if not records:
        raise ValueError("At least one scored segmentation is required")
    by_doc: dict[str, list[ScoredSegmentation]] = defaultdict(list)
    for record in records:
        by_doc[record.doc_id].append(record)
    systems = {record.system for record in records}
    for doc_id, doc_records in by_doc.items():
        if {record.system for record in doc_records} != systems:
            raise ValueError(f"Document {doc_id} does not contain every system")
        first = doc_records[0]
        for record in doc_records[1:]:
            if record.gold != first.gold or record.evidence != first.evidence:
                raise ValueError(f"Document {doc_id} has inconsistent gold/evidence vectors")


def summarize_systems(records: list[ScoredSegmentation]) -> dict[str, dict[str, float]]:
    """Return aligned macro and pooled-micro quality/Gold F1 per system."""
    _validate_records(records)
    by_system: dict[str, list[ScoredSegmentation]] = defaultdict(list)
    for record in records:
        by_system[record.system].append(record)

    summary: dict[str, dict[str, float]] = {}
    for system, system_records in sorted(by_system.items()):
        gold_total = Counts()
        soft_total = Counts()
        for record in system_records:
            gold_total += record.gold_counts
            soft_total += record.soft_counts
        summary[system] = {
            "documents": float(len(system_records)),
            "macro_quality": sum(record.quality for record in system_records) / len(system_records),
            "macro_gold_f1": sum(record.gold_f1 for record in system_records) / len(system_records),
            "micro_quality": soft_total.f1,
            "micro_gold_f1": gold_total.f1,
            "micro_gold_precision": gold_total.precision,
            "micro_gold_recall": gold_total.recall,
            "non_gap_false_positives": float(
                sum(record.non_gap_fp for record in system_records)
            ),
        }
    return summary


def _descending(summary: dict[str, dict[str, float]], field: str) -> list[str]:
    return sorted(summary, key=lambda system: (-summary[system][field], system))


def pairwise_ranking_accuracy(records: list[ScoredSegmentation]) -> dict[str, float]:
    """Compare candidates only within the same document; exclude Gold-F1 ties."""
    _validate_records(records)
    by_doc: dict[str, list[ScoredSegmentation]] = defaultdict(list)
    for record in records:
        by_doc[record.doc_id].append(record)
    correct = ties = total = 0
    for doc_records in by_doc.values():
        for left, right in itertools.combinations(doc_records, 2):
            if left.gold_f1 == right.gold_f1:
                continue
            total += 1
            if left.quality == right.quality:
                ties += 1
            elif (left.quality > right.quality) == (left.gold_f1 > right.gold_f1):
                correct += 1
    return {
        "accuracy": (correct + 0.5 * ties) / total if total else float("nan"),
        "strict_accuracy": correct / total if total else float("nan"),
        "tie_rate": ties / total if total else float("nan"),
        "pairs": float(total),
        "correct": float(correct),
        "ties": float(ties),
    }


def selection_report(records: list[ScoredSegmentation], fixed_system: str = "sat") -> dict:
    """Make macro the primary decision target and keep micro strictly paired."""
    summary = summarize_systems(records)
    if fixed_system not in summary:
        raise ValueError(f"Fixed system {fixed_system!r} is not present")

    predicted_macro = _descending(summary, "macro_quality")
    oracle_macro = _descending(summary, "macro_gold_f1")
    predicted_micro = _descending(summary, "micro_quality")
    oracle_micro = _descending(summary, "micro_gold_f1")
    selected_macro, best_macro = predicted_macro[0], oracle_macro[0]
    selected_micro, best_micro = predicted_micro[0], oracle_micro[0]

    return {
        "systems": summary,
        "macro": {
            "predicted_ranking": predicted_macro,
            "gold_ranking": oracle_macro,
            "selected": selected_macro,
            "oracle": best_macro,
            "selection_regret": summary[best_macro]["macro_gold_f1"] - summary[selected_macro]["macro_gold_f1"],
            "oracle_headroom_over_fixed": summary[best_macro]["macro_gold_f1"] - summary[fixed_system]["macro_gold_f1"],
            "fixed_system": fixed_system,
            "fixed_system_regret": summary[best_macro]["macro_gold_f1"] - summary[fixed_system]["macro_gold_f1"],
            # Negative: the reference-free selector is worse than always using the fixed system.
            "selected_minus_fixed": summary[selected_macro]["macro_gold_f1"] - summary[fixed_system]["macro_gold_f1"],
        },
        "micro_sensitivity": {
            "predicted_ranking": predicted_micro,
            "gold_ranking": oracle_micro,
            "selected": selected_micro,
            "oracle": best_micro,
            "selection_regret": summary[best_micro]["micro_gold_f1"] - summary[selected_micro]["micro_gold_f1"],
            "fixed_system_regret": summary[best_micro]["micro_gold_f1"] - summary[fixed_system]["micro_gold_f1"],
            "selected_minus_fixed": summary[selected_micro]["micro_gold_f1"] - summary[fixed_system]["micro_gold_f1"],
        },
        "within_document_pairwise": pairwise_ranking_accuracy(records),
    }
