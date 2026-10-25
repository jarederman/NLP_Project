from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np


TITLE_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "sen.", "rep.", "rev."
}
COMMON_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "fig.", "no.", "inc.", "corp.", "ltd.", "dept.", "approx."
}
ACRONYM = re.compile(r"^(?:[A-Za-z]\.){2,}$")
INITIAL = re.compile(r"^[A-Z]\.$")


@dataclass
class RuleStats:
    triggered: int = 0
    beneficial_nonboundary: int = 0
    detrimental_boundary: int = 0
    fixes: int = 0
    harms: int = 0


def clip_prob(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def apply_abbreviation_rules(units: list[str], probs: np.ndarray, gold: list[int] | None = None,
                             next_units: list[str] | None = None) -> tuple[np.ndarray, list[str], RuleStats]:
    """Lower high-confidence non-boundary positions; never hard-delete a position."""
    adjusted = np.asarray(probs, dtype=float).copy()
    reasons = ["" for _ in units]
    stats = RuleStats()
    for i, left in enumerate(units):
        right = next_units[i] if next_units is not None else (units[i + 1] if i + 1 < len(units) else "")
        low = left.lower()
        factor = 1.0
        reason = ""
        if low in TITLE_ABBREVIATIONS:
            factor, reason = 0.05, "title_abbreviation"
        elif re.fullmatch(r"\d+\.", left) and re.match(r"^\d", right):
            factor, reason = 0.01, "decimal_split"
        elif ACRONYM.fullmatch(left):
            # Preserve sentence-final acronym hypotheses before clear sentence starters.
            if right and (right[0].islower() or right[0] in ",;:"):
                factor, reason = 0.10, "acronym_continuation"
        elif low in COMMON_ABBREVIATIONS and right and right[0].islower():
            factor, reason = 0.20, "common_abbreviation_continuation"
        elif INITIAL.fullmatch(left) and right and right[0].isupper():
            factor, reason = 0.10, "name_initial"
        if factor < 1.0:
            before = adjusted[i]
            adjusted[i] *= factor
            reasons[i] = reason
            stats.triggered += 1
            if gold is not None:
                if gold[i] == 0 and adjusted[i] < before:
                    stats.beneficial_nonboundary += 1
                    stats.fixes += int(before >= 0.5 and adjusted[i] < 0.5)
                elif gold[i] == 1 and adjusted[i] < before:
                    stats.detrimental_boundary += 1
                    stats.harms += int(before >= 0.5 and adjusted[i] < 0.5)
    return adjusted, reasons, stats


def boundary_f1(y: list[int], gold: list[int]) -> float:
    yv, gv = np.asarray(y, dtype=bool), np.asarray(gold, dtype=bool)
    tp = int(np.sum(yv & gv))
    fp = int(np.sum(yv & ~gv))
    fn = int(np.sum(~yv & gv))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0


def bce(y: list[int], p: np.ndarray, eps: float = 1e-4) -> float:
    yv, pv = np.asarray(y, dtype=float), clip_prob(p, eps)
    return float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log1p(-pv)))


def weighted_bce(y: list[int], p: np.ndarray, positive_weight: float = 10.0, eps: float = 1e-4) -> float:
    yv, pv = np.asarray(y, dtype=float), clip_prob(p, eps)
    weights = np.where(yv > 0.5, positive_weight, 1.0)
    losses = -(yv * np.log(pv) + (1 - yv) * np.log1p(-pv))
    return float(np.sum(weights * losses) / np.sum(weights))


def soft_fbeta(y: list[int], p: np.ndarray, beta: float = 1.0) -> float:
    yv, pv = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    tp = float(np.sum(yv * pv))
    fp = float(np.sum(yv * (1 - pv)))
    fn = float(np.sum((1 - yv) * pv))
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / denom if denom else 1.0


def segment_lengths(y: list[int]) -> list[int]:
    lengths, last = [], -1
    for i, boundary in enumerate(y):
        if boundary:
            lengths.append(i - last)
            last = i
    lengths.append(len(y) - last)
    return lengths


def length_baseline(y: list[int], target: float = 20.0) -> float:
    lengths = np.asarray(segment_lengths(y), dtype=float)
    # Higher is better. Log-distance limits domination by a single long sentence.
    return float(-np.mean(np.abs(np.log((lengths + 1e-6) / target))))


class PlattCalibrator:
    """Two-parameter held-out calibration from raw punctuation proxy to boundary p."""
    def __init__(self, slope: float = 1.0, intercept: float = 0.0):
        self.slope = float(slope)
        self.intercept = float(intercept)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p) - np.log1p(-p)

    def fit(self, raw: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> "PlattCalibrator":
        x = self._logit(np.asarray(raw, dtype=float))
        y = np.asarray(labels, dtype=float)
        a, b = 1.0, math.log((y.mean() + 1e-4) / (1 - y.mean() + 1e-4)) - float(np.mean(x))
        reg = 1e-4
        for _ in range(max_iter):
            z = np.clip(a * x + b, -30, 30)
            p = 1 / (1 + np.exp(-z))
            w = p * (1 - p)
            grad = np.array([np.sum((p - y) * x) + reg * a, np.sum(p - y)])
            hess = np.array([[np.sum(w * x * x) + reg, np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
            step = np.linalg.solve(hess + np.eye(2) * 1e-8, grad)
            a -= step[0]
            b -= step[1]
            if np.max(np.abs(step)) < 1e-7:
                break
        self.slope, self.intercept = float(np.clip(a, 0.01, 20.0)), float(b)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        z = np.clip(self.slope * self._logit(np.asarray(raw)) + self.intercept, -30, 30)
        return 1 / (1 + np.exp(-z))

    def to_dict(self) -> dict:
        return {"slope": self.slope, "intercept": self.intercept, "method": "Platt scaling on official GUM train"}
