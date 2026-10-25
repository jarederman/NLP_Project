from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .candidates import generate_candidates
from .scoring import (
    PlattCalibrator,
    apply_abbreviation_rules,
    bce,
    boundary_f1,
    length_baseline,
    soft_fbeta,
    weighted_bce,
)


def _rankdata(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


METRIC_DIRECTIONS = {
    "raw_bce_v1": -1,
    "bce_v1": -1,
    "bce_v2": -1,
    "weighted_bce_v1": -1,
    "weighted_bce_v2": -1,
    "soft_f1_v1": 1,
    "soft_f1_v2": 1,
    "soft_fbeta_v1": 1,
    "soft_fbeta_v2": 1,
    "length_baseline": 1,
}


def _segments(text: str, offsets: list[int], boundaries: list[int]) -> list[str]:
    cuts = [offsets[i] for i, y in enumerate(boundaries) if y]
    segments, start = [], 0
    for cut in cuts:
        segments.append(text[start:cut].strip())
        start = cut
    segments.append(text[start:].strip())
    return [s for s in segments if s]


def score_dev_documents(docs, cache, calibrator: PlattCalibrator, config: dict):
    rows: list[dict] = []
    doc_aux: dict[str, dict] = {}
    for doc in docs:
        raw = np.asarray(cache[doc.doc_id]["raw_punctuation_probability"], dtype=float)
        calibrated = calibrator.predict(raw)
        adjusted, reasons, rule_stats = apply_abbreviation_rules(doc.units, calibrated, doc.gold, doc.next_units)
        doc_aux[doc.doc_id] = {
            "raw": raw,
            "calibrated": calibrated,
            "adjusted": adjusted,
            "rule_reasons": reasons,
            "rule_stats": rule_stats,
        }
        candidates = generate_candidates(doc.gold, doc.doc_id, config["seed"], config["rates"])
        for candidate in candidates:
            y = candidate.boundaries
            row = {
                "doc_id": doc.doc_id,
                "genre": doc.genre,
                "condition": candidate.condition,
                "rate": candidate.rate,
                "candidate_boundary_count": int(sum(y)),
                "gold_boundary_count": int(sum(doc.gold)),
                "candidate_positions": len(y),
                "gold_boundary_f1": boundary_f1(y, doc.gold),
                "raw_bce_v1": bce(y, raw, config["probability_clip"]),
                "bce_v1": bce(y, calibrated, config["probability_clip"]),
                "bce_v2": bce(y, adjusted, config["probability_clip"]),
                "weighted_bce_v1": weighted_bce(y, calibrated, config["weighted_bce_positive_weight"], config["probability_clip"]),
                "weighted_bce_v2": weighted_bce(y, adjusted, config["weighted_bce_positive_weight"], config["probability_clip"]),
                "soft_f1_v1": soft_fbeta(y, calibrated, 1.0),
                "soft_f1_v2": soft_fbeta(y, adjusted, 1.0),
                "soft_fbeta_v1": soft_fbeta(y, calibrated, config["soft_f_beta"]),
                "soft_fbeta_v2": soft_fbeta(y, adjusted, config["soft_f_beta"]),
                "length_baseline": length_baseline(y),
            }
            rows.append(row)
    return rows, doc_aux


def aggregate(rows: list[dict]) -> dict:
    summary: dict = {"metrics": {}, "by_condition": {}, "by_genre": {},
                     "gold_wins_by_condition_rate": {}, "by_genre_ranking": {}}
    gold_quality = [r["gold_boundary_f1"] for r in rows]
    for metric, direction in METRIC_DIRECTIONS.items():
        qualities = [direction * r[metric] for r in rows]
        global_rho = spearman(qualities, gold_quality)
        per_doc = []
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["doc_id"]].append(row)
        correct = total = ties = 0
        gold_wins = gold_total = 0
        for doc_rows in grouped.values():
            rho = spearman([direction * r[metric] for r in doc_rows], [r["gold_boundary_f1"] for r in doc_rows])
            if not math.isnan(rho):
                per_doc.append(rho)
            for a, b in itertools.combinations(doc_rows, 2):
                if a["gold_boundary_f1"] == b["gold_boundary_f1"]:
                    continue
                total += 1
                truth = a["gold_boundary_f1"] > b["gold_boundary_f1"]
                pred_a, pred_b = direction * a[metric], direction * b[metric]
                if pred_a == pred_b:
                    ties += 1
                elif (pred_a > pred_b) == truth:
                    correct += 1
            gold_row = next((r for r in doc_rows if r["condition"] == "gold"), None)
            if gold_row:
                g = direction * gold_row[metric]
                for r in doc_rows:
                    if r["condition"] == "gold":
                        continue
                    gold_total += 1
                    gold_wins += int(g > direction * r[metric])
        summary["metrics"][metric] = {
            "spearman_global": global_rho,
            "spearman_within_doc_mean": float(np.mean(per_doc)) if per_doc else float("nan"),
            "pairwise_accuracy": correct / total if total else float("nan"),
            "pairwise_tie_rate": ties / total if total else float("nan"),
            "gold_beats_perturbed": gold_wins / gold_total if gold_total else float("nan"),
            "pairwise_n": total,
        }
        summary["gold_wins_by_condition_rate"][metric] = {}
        gold_lookup = {r["doc_id"]: r for r in rows if r["condition"] == "gold"}
        for condition in ("over", "under", "shift", "mixed"):
            for rate in sorted({r["rate"] for r in rows if r["condition"] == condition}):
                candidates = [r for r in rows if r["condition"] == condition and r["rate"] == rate]
                wins = sum(direction * gold_lookup[r["doc_id"]][metric] > direction * r[metric] for r in candidates)
                summary["gold_wins_by_condition_rate"][metric][f"{condition}_{rate:g}"] = wins / len(candidates)
    for grouping, target in [("condition", "by_condition"), ("genre", "by_genre")]:
        values = defaultdict(list)
        for row in rows:
            values[row[grouping]].append(row)
        for key, group in sorted(values.items()):
            summary[target][key] = {
                "n": len(group),
                "gold_f1": float(np.mean([r["gold_boundary_f1"] for r in group])),
                **{metric: float(np.mean([r[metric] for r in group])) for metric in METRIC_DIRECTIONS},
            }
    for genre in sorted({r["genre"] for r in rows}):
        genre_rows = [r for r in rows if r["genre"] == genre]
        result = {"documents": len({r["doc_id"] for r in genre_rows})}
        for metric in ("bce_v1", "weighted_bce_v1", "soft_f1_v1", "length_baseline"):
            direction = METRIC_DIRECTIONS[metric]
            correct = total = 0
            rhos = []
            for doc_id in {r["doc_id"] for r in genre_rows}:
                doc_rows = [r for r in genre_rows if r["doc_id"] == doc_id]
                rho = spearman([direction * r[metric] for r in doc_rows], [r["gold_boundary_f1"] for r in doc_rows])
                if not math.isnan(rho):
                    rhos.append(rho)
                for a, b in itertools.combinations(doc_rows, 2):
                    if a["gold_boundary_f1"] == b["gold_boundary_f1"]:
                        continue
                    total += 1
                    correct += ((direction * a[metric] > direction * b[metric]) ==
                                (a["gold_boundary_f1"] > b["gold_boundary_f1"]))
            result[metric] = {"pairwise_accuracy": correct / total,
                              "within_doc_spearman": float(np.mean(rhos))}
        summary["by_genre_ranking"][genre] = result
    return summary


def write_metrics(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_examples(docs, rows: list[dict], aux: dict, config: dict) -> list[dict]:
    by_doc = {doc.doc_id: doc for doc in docs}
    candidates_by_doc = {
        doc.doc_id: {(c.condition, c.rate): c for c in generate_candidates(doc.gold, doc.doc_id, config["seed"], config["rates"])}
        for doc in docs
    }
    # Save representative rows plus strongest V1/V2 changes and worst ranking errors.
    chosen: list[tuple[float, dict, str]] = []
    for row in rows:
        if row["condition"] == "gold":
            continue
        direction_error = row["gold_boundary_f1"] + 0.15 * row["bce_v2"]
        chosen.append((direction_error, row, "representative_or_failure"))
    chosen.sort(key=lambda x: x[0])
    selected = chosen[:12] + chosen[-12:]
    # Explicitly include documents in which abbreviation rules fired.
    for doc_id, data in aux.items():
        if data["rule_stats"].triggered:
            row = next(r for r in rows if r["doc_id"] == doc_id and r["condition"] == "gold")
            selected.append((0.0, row, "abbreviation_rule"))
    seen, examples = set(), []
    for _, row, reason in selected:
        key = (row["doc_id"], row["condition"], row["rate"])
        if key in seen:
            continue
        seen.add(key)
        doc = by_doc[row["doc_id"]]
        cand = candidates_by_doc[doc.doc_id][(row["condition"], row["rate"])]
        daux = aux[doc.doc_id]
        triggered = [
            {"position": i, "left": doc.units[i], "right": doc.next_units[i],
             "gold": doc.gold[i], "reason": rule, "p_before": float(daux["calibrated"][i]), "p_after": float(daux["adjusted"][i])}
            for i, rule in enumerate(daux["rule_reasons"]) if rule
        ]
        examples.append({
            "selection_reason": reason,
            "doc_id": doc.doc_id,
            "genre": doc.genre,
            "controlled_text": doc.text,
            "gold_boundary_indices": [i for i, y in enumerate(doc.gold) if y],
            "candidate_condition": row["condition"],
            "rate": row["rate"],
            "candidate_boundary_indices": [i for i, y in enumerate(cand.boundaries) if y],
            "candidate_segments": _segments(doc.text, doc.char_offsets, cand.boundaries),
            "raw_punctuation_probabilities": daux["raw"].tolist(),
            "calibrated_boundary_probabilities_v1": daux["calibrated"].tolist(),
            "boundary_probabilities_v2": daux["adjusted"].tolist(),
            "abbreviation_events": triggered,
            "scores": {k: row[k] for k in ["gold_boundary_f1", *METRIC_DIRECTIONS]},
        })
        if len(examples) >= 36:
            break
    return examples


def write_jsonl(records: list[dict], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
