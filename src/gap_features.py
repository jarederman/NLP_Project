"""Gold-free per-gap features for evidence calibration (P2).

Every feature is a function of the raw document text, the candidate gap offsets,
and the BERT evidence for that document only; no Gold boundary and no candidate
segmentation is ever an input (the separate ``vote_features`` is the one,
explicitly flagged, exception used only as a comparison).

Groups (see results/calibration_search/PREREGISTRATION.md):
  B  BERT logit and mask mode
  C  left-context category, and its interaction with the BERT logit
  K  left/right word shape, crossed with the context category
  A  abbreviation cues with Kiss & Strunk-style within-document type statistics
  L  layout: distance to neighbouring terminal units, Title-Case ratio of runs
  O  Punkt-style orthographic statistics of the right word within the document
     (is it normally lower-case here, and was it capitalised mid-sentence elsewhere?)
  W  left/right word identities (categorical; target-encoded on training folds only)
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter

import numpy as np

from .analyze_evidence_gaps import left_context_category


CATEGORIES = ("terminal", "no_punct_alnum", "no_punct_other", "dash", "colon", "semicolon", "comma")
SHAPES = ("cap", "allcap", "lower", "digit", "other")
OPENERS = "\"'“‘«([{"
TRAILING = ".!?\"'”’»)]}:;,"
EPS = 1e-6
GROUPS = ("B", "C", "K", "A", "L", "O")
LEXICAL_GROUP = "W"  # categorical word ids, target-encoded inside the fold (see lexical_ids)


def word_shape(word: str) -> str:
    core = word.strip(OPENERS + TRAILING)
    if not core:
        return "other"
    if core[0].isdigit():
        return "digit"
    letters = [ch for ch in core if ch.isalpha()]
    if not letters:
        return "other"
    if len(letters) > 1 and all(ch.isupper() for ch in letters):
        return "allcap"
    if core[0].isupper():
        return "cap"
    if core[0].islower():
        return "lower"
    return "other"


def _chunks(text: str) -> tuple[list[tuple[int, int]], dict[int, int]]:
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    return spans, {end: i for i, (_, end) in enumerate(spans)}


def _abbreviation_core(word: str) -> str | None:
    """'Dr.' -> 'dr', 'U.S.' -> 'u.s'; None unless the chunk ends in a single period."""
    stripped = word.rstrip("\"'”’»)]}")
    if not stripped.endswith(".") or stripped.endswith(".."):
        return None
    core = stripped[:-1].lstrip(OPENERS)
    return core.lower() if core else None


def document_features(text: str, offsets: list[int], raw_probability: list[float],
                      mask_modes: list[str]) -> dict[str, np.ndarray]:
    """Return {group: (n_gaps, n_features) array} for one document."""
    n = len(offsets)
    if not (n == len(raw_probability) == len(mask_modes)):
        raise ValueError("offsets, probabilities and mask modes must align")
    spans, by_end = _chunks(text)
    words = [text[s:e] for s, e in spans]
    lower_counts = Counter(w.strip(OPENERS + TRAILING).lower() for w in words)
    period_counts = Counter(c for c in (_abbreviation_core(w) for w in words) if c)

    categories = [left_context_category(text, o) for o in offsets]
    left_idx = []
    for o in offsets:
        if o not in by_end:
            raise ValueError(f"Gap offset {o} does not end a whitespace chunk")
        left_idx.append(by_end[o])

    z = np.array([math.log(min(max(p, EPS), 1 - EPS) / (1 - min(max(p, EPS), 1 - EPS)))
                  for p in raw_probability])
    replace = np.array([m == "replace_terminal_punctuation" for m in mask_modes], dtype=float)
    cat = np.zeros((n, len(CATEGORIES)))
    for i, c in enumerate(categories):
        cat[i, CATEGORIES.index(c)] = 1.0
    terminal = cat[:, 0]
    non_terminal = 1.0 - terminal

    out: dict[str, np.ndarray] = {}
    out["B"] = np.column_stack([z, replace])
    out["C"] = np.column_stack([cat[:, 1:], cat * z[:, None]])

    left_shape = np.zeros((n, len(SHAPES)))
    right_shape = np.zeros((n, len(SHAPES)))
    right_opener = np.zeros(n)
    for i, li in enumerate(left_idx):
        left_shape[i, SHAPES.index(word_shape(words[li]))] = 1.0
        if li + 1 < len(words):
            right = words[li + 1]
            right_shape[i, SHAPES.index(word_shape(right))] = 1.0
            right_opener[i] = float(right[:1] in OPENERS)
    right_upper = right_shape[:, 0] + right_shape[:, 1]
    out["K"] = np.column_stack([
        left_shape, right_shape, right_opener,
        right_upper * non_terminal, right_upper * terminal, right_shape[:, 2] * terminal,
    ])

    abbreviation = np.zeros((n, 5))
    for i, li in enumerate(left_idx):
        core = _abbreviation_core(words[li])
        if core is None or categories[i] != "terminal":
            continue
        bare = lower_counts[core] - period_counts[core]  # occurrences without the period
        abbreviation[i] = [
            1.0,
            float(len(core.replace(".", "")) == 1),
            float(len(core) <= 3),
            float("." in core),
            math.log1p(max(bare, 0)) - math.log1p(period_counts[core]),
        ]
    out["A"] = abbreviation

    # Layout: distance (in chunks) to the neighbouring terminal-unit gaps, and how
    # Title-Case the runs on each side are (headings are short, capitalised runs).
    cap = np.array([word_shape(w) in ("cap", "allcap") for w in words], dtype=float)
    term_chunks = [left_idx[i] for i in range(n) if categories[i] == "terminal"]
    layout = np.zeros((n, 4))
    for i, li in enumerate(left_idx):
        k = bisect.bisect_left(term_chunks, li)
        prev_end = term_chunks[k - 1] if k > 0 else -1          # last terminal chunk strictly before
        j = bisect.bisect_right(term_chunks, li)
        next_end = term_chunks[j] if j < len(term_chunks) else len(words) - 1   # first strictly after
        left_run = cap[prev_end + 1: li + 1]
        right_run = cap[li + 1: next_end + 1]
        layout[i] = [
            math.log1p(li - prev_end),
            math.log1p(max(next_end - li, 0)),
            left_run.mean() if left_run.size else 0.0,
            right_run.mean() if right_run.size else 0.0,
        ]
    out["L"] = layout

    # Orthographic context (after Kiss & Strunk 2006): a right word that is capitalised
    # here but normally lower-case in this document is likely sentence-initial; one that
    # is also capitalised in mid-sentence positions is likely a proper noun.
    after_terminal = np.zeros(len(words), dtype=bool)   # chunk follows a terminal-unit gap
    for i, li in enumerate(left_idx):
        if categories[i] == "terminal" and li + 1 < len(words):
            after_terminal[li + 1] = True
    lower_n, upper_mid_n = Counter(), Counter()
    for k, w in enumerate(words):
        core = w.strip(OPENERS + TRAILING)
        if not core or not core[0].isalpha():
            continue
        if core[0].islower():
            lower_n[core.lower()] += 1
        elif k > 0 and not after_terminal[k]:
            upper_mid_n[core.lower()] += 1
    ortho = np.zeros((n, 4))
    for i, li in enumerate(left_idx):
        if li + 1 >= len(words):
            continue
        core = words[li + 1].strip(OPENERS + TRAILING)
        if not core or not core[0].isalpha():
            continue
        key, is_upper = core.lower(), core[0].isupper()
        lower, upper_mid = lower_n[key], upper_mid_n[key] - (1 if is_upper and not after_terminal[li + 1] else 0)
        ortho[i] = [
            math.log1p(lower),
            math.log1p(max(upper_mid, 0)),
            float(is_upper and lower > 0),
            float(is_upper and lower > 0 and upper_mid <= 0) * (1.0 - terminal[i]),
        ]
    out["O"] = ortho
    return out


def lexical_ids(text: str, offsets: list[int]) -> tuple[list[str], list[str]]:
    """Lower-cased core of the word left and right of each gap (categorical features)."""
    spans, by_end = _chunks(text)
    words = [text[s:e] for s, e in spans]
    left, right = [], []
    for o in offsets:
        li = by_end[o]
        left.append(words[li].strip(OPENERS + TRAILING).lower() or words[li])
        nxt = words[li + 1] if li + 1 < len(words) else ""
        right.append(nxt.strip(OPENERS + TRAILING).lower() or nxt)
    return left, right


def vote_features(offsets: list[int], candidates: dict[str, tuple[int, ...]]) -> np.ndarray:
    """Share of candidate systems that place a boundary at each gap (NOT gold-free of
    the candidates; reported only as a comparison, never eligible for selection)."""
    vectors = np.array(list(candidates.values()), dtype=float)
    if vectors.shape[1] != len(offsets):
        raise ValueError("candidate vectors must align with offsets")
    return vectors.mean(axis=0)[:, None]
