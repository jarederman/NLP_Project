from __future__ import annotations

import numpy as np

from .candidates import generate_candidates
from .evaluate import spearman


def _context(doc, position: int, probability: float) -> str:
    offset = doc.char_offsets[position]
    left = doc.text[max(0, offset - 55):offset].replace("\n", " ").strip()
    right = doc.text[offset + 1:offset + 56].replace("\n", " ").strip()
    return f"“{left} | {right}”（p={probability:.4f}）"


def diagnostics_and_cases(docs, cache: dict, calibrator, rows: list[dict], config: dict) -> tuple[dict, dict]:
    by_doc = {d.doc_id: d for d in docs}
    calibrated = {d.doc_id: calibrator.predict(np.asarray(cache[d.doc_id]["raw_punctuation_probability"])) for d in docs}
    raw = np.concatenate([cache[d.doc_id]["raw_punctuation_probability"] for d in docs])
    probs = np.concatenate([calibrated[d.doc_id] for d in docs])
    labels = np.concatenate([d.gold for d in docs]).astype(bool)
    gold_rows = [r for r in rows if r["condition"] == "gold"]
    diagnostics = {
        "boundary_prevalence": float(labels.mean()),
        "raw_positive_mean": float(raw[labels].mean()), "raw_negative_mean": float(raw[~labels].mean()),
        "raw_below_clip": int((raw < config["probability_clip"]).sum()),
        "calibrated_positive_mean": float(probs[labels].mean()),
        "calibrated_negative_mean": float(probs[~labels].mean()),
        "calibrated_min": float(probs.min()), "calibrated_max": float(probs.max()),
        "gold_length_rho_bce": spearman([-r["bce_v1"] for r in gold_rows], [r["candidate_positions"] for r in gold_rows]),
        "gold_length_rho_weighted": spearman([-r["weighted_bce_v1"] for r in gold_rows], [r["candidate_positions"] for r in gold_rows]),
        "gold_length_rho_soft_f1": spearman([r["soft_f1_v1"] for r in gold_rows], [r["candidate_positions"] for r in gold_rows]),
    }
    lookup = {(r["doc_id"], r["condition"], r["rate"]): r for r in rows}
    gold_lookup = {r["doc_id"]: r for r in gold_rows}

    under_rows = [r for r in rows if r["condition"] == "under" and r["rate"] == 0.2]
    bce_row = max(under_rows, key=lambda r: gold_lookup[r["doc_id"]]["bce_v1"] - r["bce_v1"])
    doc = by_doc[bce_row["doc_id"]]
    cand = next(c for c in generate_candidates(doc.gold, doc.doc_id, config["seed"], config["rates"])
                if c.condition == "under" and c.rate == 0.2)
    errors = [i for i, (a, b) in enumerate(zip(cand.boundaries, doc.gold)) if a != b]
    pos = min(errors, key=lambda i: calibrated[doc.doc_id][i])
    bce_case = {"doc_id": doc.doc_id, "gold_f1": bce_row["gold_boundary_f1"],
                "gold_bce": gold_lookup[doc.doc_id]["bce_v1"], "candidate_bce": bce_row["bce_v1"],
                "context": _context(doc, pos, calibrated[doc.doc_id][pos])}

    shift_rows = [r for r in rows if r["condition"] == "shift" and r["rate"] == 0.2]
    shift_row = max(shift_rows, key=lambda r: gold_lookup[r["doc_id"]]["soft_f1_v1"] - r["soft_f1_v1"])
    doc = by_doc[shift_row["doc_id"]]
    cand = next(c for c in generate_candidates(doc.gold, doc.doc_id, config["seed"], config["rates"])
                if c.condition == "shift" and c.rate == 0.2)
    removed = [i for i, (a, b) in enumerate(zip(cand.boundaries, doc.gold)) if not a and b]
    added = [i for i, (a, b) in enumerate(zip(cand.boundaries, doc.gold)) if a and not b]
    old = max(removed, key=lambda i: calibrated[doc.doc_id][i])
    new = min(added, key=lambda i: calibrated[doc.doc_id][i])
    shift_case = {"doc_id": doc.doc_id, "gold_f1": shift_row["gold_boundary_f1"],
                  "gold_soft_f1": gold_lookup[doc.doc_id]["soft_f1_v1"], "candidate_soft_f1": shift_row["soft_f1_v1"],
                  "context": f"真边界 {_context(doc, old, calibrated[doc.doc_id][old])} 被移到非边界 {_context(doc, new, calibrated[doc.doc_id][new])}"}

    soft_row = max(under_rows, key=lambda r: r["soft_f1_v1"] - gold_lookup[r["doc_id"]]["soft_f1_v1"])
    doc = by_doc[soft_row["doc_id"]]
    cand = next(c for c in generate_candidates(doc.gold, doc.doc_id, config["seed"], config["rates"])
                if c.condition == "under" and c.rate == 0.2)
    errors = [i for i, (a, b) in enumerate(zip(cand.boundaries, doc.gold)) if a != b]
    pos = min(errors, key=lambda i: calibrated[doc.doc_id][i])
    soft_case = {"doc_id": doc.doc_id, "gold_f1": soft_row["gold_boundary_f1"],
                 "delta": soft_row["soft_f1_v1"] - gold_lookup[doc.doc_id]["soft_f1_v1"],
                 "context": _context(doc, pos, calibrated[doc.doc_id][pos])}
    return diagnostics, {"bce_under": bce_case, "shift_success": shift_case, "soft_failure": soft_case}
