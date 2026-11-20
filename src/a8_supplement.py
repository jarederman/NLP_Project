"""Amendment 8: supplementary validity analyses (analysis only, CPU, cached per-gap evidence).

  8.1 Gold as a fifth candidate            8.2 controlled perturbations by error type (large scale)
  8.3 per-error-type pricing of real errors 8.4 selection regret vs Always-SaT (original question)
  8.5 trivial baselines                     8.6 Stack without E1 / E2 (same-family bias)
  8.7 document length

GUM = GUM train with leave-one-genre-out evidence (development set). GENTLE / LinES parts are
post-test analyses. Nothing frozen is overwritten; every output goes to ``results/a8``.
Candidates live on whitespace gaps (the Soft F1 candidate space); "true quality" of a synthetic
candidate is its whitespace-gap boundary F1 against Gold.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from .a6_combine import E0_DEV, FINAL, STACK_LAMBDA, all_scores, decomposition, dev_evidence, dev_qe, evaluate, \
    fairness, gold_items, ngram_matrix, raw_signal, rule_matrix, soft_scores, stack_features, terminal_mask, \
    with_scores
from .a6_data import CORPORA, ROOT, SEED, SYSTEMS, doc_slices, gold_sets, jsonl, logo_folds, predictions, \
    safe_lower, surfaces, vote_matrix
from .a6_signals import dawid_skene, fit_logistic_np, length_prior, predict_logistic_np
from .analyze_evidence_gaps import left_context_category
from .meta_eval import max_soft_f1
from .quality_agreement import Items, statistics, tie_exact


RESULTS = ROOT / "results/a8"
# Review fix (2026-11-20): leave-one-genre-out Platt calibration (was the genre-stratified 5-fold file
# data/real_system/train_cv/evidence_cv.jsonl, whose calibration had seen each document's own genre).
PUNCEV_GUM = ROOT / "data/real_system/train_cv/boundary_probe/raw/platt_lam1e-06/logo.jsonl"
TERMINAL, PUNCT, NOPUNCT = 0, 1, 2
PUNCT_CATEGORIES = {"comma", "colon", "semicolon", "dash"}
BASE_TYPES = ("merge_terminal", "merge_other", "split_terminal", "split_punct", "split_nopunct", "shift")
TYPES = BASE_TYPES + ("mixed",)
RATES = (0.05, 0.10, 0.20, 0.40)
REPLICATES = 20
SHIFT_WINDOW = 3
SOFT = {"PuncEv": "PuncEv", "Probe": "E0", "E1": "E1", "E2": "E2", "Stack": "stack"}
HARD = {"H(Probe)": "E0", "H(Stack)": "stack"}
SCORERS = tuple(SOFT) + tuple(HARD) + ("Rule", "Cons", "DS", "LenPrior")
MULTI = ("Cons", "DS")
COMPARE = (("Stack", "Probe"), ("Stack", "PuncEv"), ("Stack", "Cons"), ("Stack", "H(Stack)"))
ERROR_TYPES = ("miss_terminal", "miss_other", "fp_terminal", "fp_punct", "fp_nopunct")
BLIND_SPOT_RATIO, INSENSITIVE, RECOGNIZE_GOLD, FAMILY_MARGIN = 0.5, 0.6, 0.8, 0.05
TOL = 1e-12


# ------------------------------------------------------------------ primitive scorers (vectorised over rows)
def hard_f1(c: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Boundary F1 of every row of ``c`` (m, n) against the 0/1 reference ``ref`` (n,); both empty = 1."""
    c = np.atleast_2d(c).astype(float)
    ref = np.asarray(ref, float)
    denom = c.sum(1) + ref.sum()
    return np.where(denom == 0, 1.0, 2 * (c @ ref) / np.where(denom == 0, 1.0, denom))


def soft_f1(c: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Soft F1 = 2 sum(c p) / (sum c + sum p) per row (identical to ``scoring.soft_fbeta``)."""
    return hard_f1(c, p)


def sentence_lengths(row: np.ndarray) -> np.ndarray:
    """Word counts of the sentences of one candidate; gap j follows word j, the document end closes."""
    return np.diff(np.concatenate([[-1], np.flatnonzero(row), [len(row)]]))


def length_score(c: np.ndarray, prior: np.ndarray) -> np.ndarray:
    top = len(prior) - 1
    log_prior = np.log(np.maximum(prior, 1e-300))
    return np.array([float(log_prior[np.minimum(sentence_lengths(r), top)].mean()) for r in np.atleast_2d(c)])


def credit(diff: np.ndarray) -> np.ndarray:
    """1 if diff > 0, 0.5 on a tie, 0 otherwise."""
    diff = np.asarray(diff, float)
    return np.where(np.abs(diff) <= TOL, 0.5, (diff > 0).astype(float))


def top1_credit(wins: np.ndarray) -> float:
    """Credit for being first, from pairwise outcomes in {1, 0.5, 0}: any loss = 0, ties share."""
    wins = np.asarray(wins, float)
    if (wins == 0).any():
        return 0.0
    return 1.0 / (1 + int((wins == 0.5).sum()))


# ------------------------------------------------------------------ per-document context
@dataclass
class DocCtx:
    doc_id: str
    genre: str
    gold: np.ndarray                      # (n,) bool
    cats: np.ndarray                      # (n,) TERMINAL / PUNCT / NOPUNCT
    ev: dict                              # name -> (n,) evidence
    norm: dict                            # name -> max Soft F1 (normaliser)
    votes: np.ndarray                     # (4, n) bool
    cons: np.ndarray                      # (4, n) bool, majority of the other three systems
    ds: np.ndarray                        # (4, n) leave-one-system-out Dawid-Skene posteriors
    prior: np.ndarray                     # sentence-length prior

    @property
    def n(self) -> int:
        return len(self.gold)


def category_codes(text: str, offsets) -> np.ndarray:
    lower = safe_lower(text)
    out = []
    for o in offsets:
        cat = left_context_category(lower, o)
        out.append(TERMINAL if cat == "terminal" else PUNCT if cat in PUNCT_CATEGORIES else NOPUNCT)
    return np.array(out, dtype=int)


def consensus_refs(votes: np.ndarray) -> np.ndarray:
    total = votes.sum(0)
    return np.array([(total - votes[s]) >= 2 for s in range(len(votes))])


def score_matrix(c: np.ndarray, ctx: DocCtx, ref: int | None = None) -> dict[str, np.ndarray]:
    """Every scorer on the rows of ``c``. Multi-system scorers use the reference without system
    ``ref``, or (``ref`` None: Gold / synthetic candidates) the mean over the four such references."""
    c = np.atleast_2d(c).astype(bool)
    out = {}
    for name, key in SOFT.items():
        if key in ctx.ev:
            out[name] = soft_f1(c, ctx.ev[key]) / ctx.norm[key] if ctx.n else np.ones(len(c))
    for name, key in HARD.items():
        out[name] = hard_f1(c, ctx.ev[key] >= 0.5)
    out["Rule"] = hard_f1(c, ctx.cats == TERMINAL)
    refs = range(len(SYSTEMS)) if ref is None else [ref]
    out["Cons"] = np.mean([hard_f1(c, ctx.cons[s]) for s in refs], axis=0)
    out["DS"] = np.mean([soft_f1(c, ctx.ds[s]) for s in refs], axis=0)
    out["LenPrior"] = length_score(c, ctx.prior)
    return out


# ------------------------------------------------------------------ 8.2 perturbations
def apply_op(c: np.ndarray, gold: np.ndarray, cats: np.ndarray, op: str, rng) -> bool:
    """One perturbation operation in place on ``c``; False if the document has no position for it."""
    if op.startswith("merge"):
        where = cats == TERMINAL if op == "merge_terminal" else cats != TERMINAL
        pool = np.flatnonzero(c & gold & where)
        if not len(pool):
            return False
        c[pool[rng.integers(len(pool))]] = False
        return True
    if op.startswith("split"):
        cat = {"split_terminal": TERMINAL, "split_punct": PUNCT, "split_nopunct": NOPUNCT}[op]
        pool = np.flatnonzero(~c & ~gold & (cats == cat))
        if not len(pool):
            return False
        c[pool[rng.integers(len(pool))]] = True
        return True
    if op != "shift":
        raise ValueError(op)
    n, options = len(c), []
    for b in np.flatnonzero(c & gold):
        targets = [j for j in range(max(0, b - SHIFT_WINDOW), min(n, b + SHIFT_WINDOW + 1))
                   if j != b and not c[j] and not gold[j]]
        if targets:
            options.append((b, targets))
    if not options:
        return False
    b, targets = options[rng.integers(len(options))]
    c[b] = False
    c[targets[rng.integers(len(targets))]] = True
    return True


def perturb(gold: np.ndarray, cats: np.ndarray, kind: str, k: int, rng) -> tuple[np.ndarray, int]:
    c = gold.copy()
    done = 0
    for _ in range(k):
        op = BASE_TYPES[rng.integers(len(BASE_TYPES))] if kind == "mixed" else kind
        done += apply_op(c, gold, cats, op, rng)
    return c, done


def n_ops(rate: float, n_gold: int) -> int:
    return max(1, int(np.floor(rate * n_gold + 0.5)))


def candidates(ctx: DocCtx, doc_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gold (row 0) plus every synthetic candidate; returns (C, type index, rate index), -1 for Gold."""
    rows, t_idx, r_idx = [ctx.gold.copy()], [-1], [-1]
    for t, kind in enumerate(TYPES):
        for r, rate in enumerate(RATES):
            rng = np.random.default_rng([SEED, doc_index, t, r])
            k = n_ops(rate, int(ctx.gold.sum()))
            for _ in range(REPLICATES):
                c, done = perturb(ctx.gold, ctx.cats, kind, k, rng)
                if done:
                    rows.append(c)
                    t_idx.append(t)
                    r_idx.append(r)
    return np.array(rows), np.array(t_idx), np.array(r_idx)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx = x.var()
    return float(((x - x.mean()) * (y - y.mean())).mean() / vx) if vx > TOL else float("nan")


def pair_agreement(x: np.ndarray, y: np.ndarray) -> float:
    """Share of pairs with different x whose y order agrees (y ties 0.5)."""
    dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
    valid = np.triu(np.abs(dx) > TOL, 1)
    if not valid.any():
        return float("nan")
    agree = np.where(np.abs(dy) <= TOL, 0.5, (np.sign(dx) == np.sign(dy)).astype(float))
    return float(agree[valid].mean())


def doc_perturbation_stats(ctx: DocCtx, doc_index: int) -> dict:
    c, t_idx, r_idx = candidates(ctx, doc_index)
    true = hard_f1(c, ctx.gold)
    scores = score_matrix(c, ctx)
    names = [s for s in SCORERS if s in scores]
    detect = np.full((len(SCORERS), len(TYPES), len(RATES)), np.nan)
    slope = np.full((len(SCORERS), len(TYPES)), np.nan)
    mono = np.full((len(SCORERS), len(TYPES)), np.nan)
    available = np.zeros(len(TYPES), dtype=bool)
    for t in range(len(TYPES)):
        rows = np.flatnonzero(t_idx == t)
        if not len(rows):
            continue
        available[t] = True
        idx = np.concatenate([[0], rows])
        for name in names:
            i, s = SCORERS.index(name), scores[name]
            for r in range(len(RATES)):
                sel = rows[r_idx[rows] == r]
                if len(sel):
                    detect[i, t, r] = credit(s[0] - s[sel]).mean()
            slope[i, t] = ols_slope(true[idx], s[idx])
            mono[i, t] = pair_agreement(true[idx], s[idx])
    mixed = np.concatenate([[0], np.flatnonzero(t_idx == TYPES.index("mixed"))])
    return {"detect": detect, "slope": slope, "mono": mono, "available": available,
            "mixed_true": true[mixed], "mixed_scores": {n: scores[n][mixed] for n in names}}


# ------------------------------------------------------------------ 8.1 / 8.3 real systems
def gold_vs_systems(ctx: DocCtx) -> dict[str, np.ndarray]:
    """Per scorer: pairwise credit of Gold vs each system (4,), same reference for both sides."""
    out = defaultdict(list)
    for s in range(len(SYSTEMS)):
        sc = score_matrix(np.vstack([ctx.gold, ctx.votes[s]]), ctx, ref=s)
        for name, v in sc.items():
            out[name].append(float(credit(v[0] - v[1])))
    return {k: np.array(v) for k, v in out.items()}


def error_masks(c: np.ndarray, gold: np.ndarray, cats: np.ndarray) -> dict[str, np.ndarray]:
    miss, fp = gold & ~c, c & ~gold
    return {"miss_terminal": miss & (cats == TERMINAL), "miss_other": miss & (cats != TERMINAL),
            "fp_terminal": fp & (cats == TERMINAL), "fp_punct": fp & (cats == PUNCT),
            "fp_nopunct": fp & (cats == NOPUNCT)}


def fixed(c: np.ndarray, mask: np.ndarray, kind: str) -> np.ndarray:
    out = c.copy()
    out[mask] = kind.startswith("miss")       # add back missed boundaries / remove false splits
    return out


def doc_error_pricing(ctx: DocCtx) -> dict:
    """(system, error type) -> count, delta true F1, delta score per scorer when only that type is fixed."""
    out = {}
    for s in range(len(SYSTEMS)):
        c = ctx.votes[s]
        masks = error_masks(c, ctx.gold, ctx.cats)
        rows = [c] + [fixed(c, masks[e], e) for e in ERROR_TYPES]
        mat = np.array(rows)
        true = hard_f1(mat, ctx.gold)
        sc = score_matrix(mat, ctx, ref=s)
        for j, e in enumerate(ERROR_TYPES, 1):
            out[(SYSTEMS[s], e)] = {"count": int(masks[e].sum()), "d_true": float(true[j] - true[0]),
                                    "d_score": {n: float(v[j] - v[0]) for n, v in sc.items()}}
    return out


# ------------------------------------------------------------------ corpus contexts
@dataclass
class CorpusCtx:
    name: str
    docs: list
    ctx: list[DocCtx]
    items: Items
    scores: dict                          # official real-item scores (+ added ones)
    ev: dict                              # full per-gap evidence
    y: np.ndarray
    stack_x: np.ndarray | None = None     # stack feature matrix (for 8.6)
    extra: dict = field(default_factory=dict)


def _puncev_gum(docs) -> np.ndarray:
    rows = {r["doc_id"]: r for r in jsonl(PUNCEV_GUM)}
    out = []
    for d in docs:
        r = rows[d.doc_id]
        if r["text_sha256"] != d.text_sha256 or r["candidate_offsets"] != d.candidate_offsets:
            raise RuntimeError(f"PuncEv evidence does not match {d.doc_id}")
        out.append(r["calibrated_boundary_probability"])
    return np.concatenate(out)


def _build_docs(docs, gold, ev, preds, terminal, priors) -> list[DocCtx]:
    votes = vote_matrix(docs, preds).astype(bool)
    fit = dawid_skene(votes.astype(float), terminal.astype(int), ev["stack"])
    out = []
    for d, sl in zip(docs, doc_slices(docs)):
        e = {k: np.asarray(v[sl], float) for k, v in ev.items()}
        g = np.array([o in gold[d.doc_id] for o in d.candidate_offsets], dtype=bool)
        v = votes[:, sl]
        out.append(DocCtx(d.doc_id, d.genre, g, category_codes(d.text, d.candidate_offsets), e,
                          {k: (max_soft_f1(tuple(p)) if len(p) else 1.0) for k, p in e.items()},
                          v, consensus_refs(v), fit["leave_one_out"][:, sl], priors[d.doc_id]))
    return out


def gum_corpus() -> CorpusCtx:
    corpus = CORPORA["gum"]
    docs = surfaces(corpus)
    gold = gold_sets(corpus)
    ev, extras = dev_evidence(corpus, docs, gold)
    ev["PuncEv"] = _puncev_gum(docs)
    items = gold_items(corpus, docs, E0_DEV, ROOT / "data/real_system/train_cv/frozen_scores_cv.jsonl")
    scores, features, _ = all_scores(corpus, docs, ev, "stack")
    scores.update(dev_qe(docs, features, items.gold))
    scores["PuncEv"] = soft_scores(docs, predictions(corpus), ev["PuncEv"])[1]
    priors = {}
    for genre in sorted({d.genre for d in docs}):
        prior = length_prior([d for d in docs if d.genre != genre], gold)
        priors.update({d.doc_id: prior for d in docs if d.genre == genre})
    keep = {k: ev[k] for k in ("PuncEv", "E0", "E1", "E2", "stack")}
    ctx = _build_docs(docs, gold, keep, predictions(corpus), terminal_mask(docs), priors)
    x = stack_features(ev["E0"], raw_signal(corpus, docs, "E1"), raw_signal(corpus, docs, "E2"),
                       extras["rules"], extras["ngrams"])
    return CorpusCtx("gum", docs, ctx, items, scores, ev, extras["y"], x)


def test_corpus(name: str, final: dict) -> CorpusCtx:
    corpus = CORPORA[name]
    docs = surfaces(corpus)
    gold = gold_sets(corpus)
    items = gold_items(corpus, docs, corpus.path("evidence", "E0.jsonl"), corpus.path("frozen_scores.jsonl"))
    frozen = jsonl(corpus.path("frozen_scores.jsonl"))
    scores = {n: np.array([r["scores"][n] for r in frozen]) for n in frozen[0]["scores"]}
    ev = {n: np.concatenate([r["calibrated_boundary_probability"] for r in jsonl(corpus.path("evidence", f"{n}.jsonl"))])
          for n in ("E0", "E1", "E2", "stack")}
    y = np.concatenate([[int(o in gold[d.doc_id]) for o in d.candidate_offsets] for d in docs]).astype(float)
    ctx = _build_docs(docs, gold, ev, predictions(corpus), terminal_mask(docs),
                      {d.doc_id: final["length_prior"] for d in docs})
    x = stack_features(ev["E0"], raw_signal(corpus, docs, "E1"), raw_signal(corpus, docs, "E2"),
                       rule_matrix(docs), ngram_matrix(final["ngram"], docs))
    return CorpusCtx(name, docs, ctx, items, scores, ev, y, x)


# ------------------------------------------------------------------ bootstrap helpers
def resample_plan(genres: np.ndarray, n: int, seed: int = SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(genres == g) for g in sorted(set(genres.tolist()))]
    return [np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata]) for _ in range(n)]


def ci(values) -> list[float]:
    v = np.asarray([x for x in values if x == x], float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def boot_mean(per_doc: np.ndarray, plan) -> dict:
    """Macro mean over documents (axis 0, NaN-aware) with its bootstrap CI, elementwise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN cells (scorer / type absent)
        point = np.nanmean(per_doc, 0)
        draws = np.array([np.nanmean(per_doc[idx], 0) for idx in plan])
        lo, hi = np.nanpercentile(draws, 2.5, 0), np.nanpercentile(draws, 97.5, 0)
    return {"point": point, "lo": lo, "hi": hi, "draws": draws}


# ------------------------------------------------------------------ 8.2 corpus level
def perturbation_report(cc: CorpusCtx, n_boot: int) -> dict:
    stats = [doc_perturbation_stats(ctx, i) for i, ctx in enumerate(cc.ctx)]
    genres = np.array([c.genre for c in cc.ctx])
    plan = resample_plan(genres, n_boot)
    present = [s for s in SCORERS if s in stats[0]["mixed_scores"]]
    det = boot_mean(np.array([s["detect"] for s in stats]), plan)
    slo = boot_mean(np.array([s["slope"] for s in stats]), plan)
    mon = boot_mean(np.array([s["mono"] for s in stats]), plan)
    avail = np.array([s["available"] for s in stats]).sum(0)
    r10 = RATES.index(0.10)
    out = {"documents": len(stats), "available_docs": dict(zip(TYPES, avail.tolist())), "scorers": present,
           "candidates": int(sum(len(s["mixed_true"]) for s in stats)), "per_scorer": {}}
    mo, mt = TYPES.index("merge_other"), TYPES.index("merge_terminal")
    for name in present:
        i = SCORERS.index(name)
        ratio_draws = slo["draws"][:, i, mo] / slo["draws"][:, i, mt]
        ratio = float(slo["point"][i, mo] / slo["point"][i, mt])
        out["per_scorer"][name] = {
            "detect": {t: {f"{r:g}": [float(det["point"][i, k, j]), float(det["lo"][i, k, j]), float(det["hi"][i, k, j])]
                           for j, r in enumerate(RATES)} for k, t in enumerate(TYPES)},
            "slope": {t: [float(slo["point"][i, k]), float(slo["lo"][i, k]), float(slo["hi"][i, k])] for k, t in enumerate(TYPES)},
            "monotone": {t: [float(mon["point"][i, k]), float(mon["lo"][i, k]), float(mon["hi"][i, k])] for k, t in enumerate(TYPES)},
            "asymmetry": [ratio, *ci(ratio_draws)],
            "blind_spot_merge_other": bool(ratio < BLIND_SPOT_RATIO),
            "insensitive_types_r10": [t for k, t in enumerate(TYPES) if det["point"][i, k, r10] < INSENSITIVE],
        }
    comps = {}
    for a, b in COMPARE:
        if a not in present or b not in present:
            continue
        ia, ib = SCORERS.index(a), SCORERS.index(b)
        comps[f"{a} - {b}"] = {
            "detect_r10": {t: [float(det["point"][ia, k, r10] - det["point"][ib, k, r10]),
                               *ci(det["draws"][:, ia, k, r10] - det["draws"][:, ib, k, r10])] for k, t in enumerate(TYPES)},
            "slope": {t: [float(slo["point"][ia, k] - slo["point"][ib, k]),
                          *ci(slo["draws"][:, ia, k] - slo["draws"][:, ib, k])] for k, t in enumerate(TYPES)}}
    out["paired"] = comps
    out["cross_document"] = cross_document(stats, genres, present, plan)
    out["_per_doc_detect"] = np.array([s["detect"] for s in stats])     # for 8.7, removed before writing
    return out


def cross_document(stats, genres, present, plan) -> dict:
    sizes = [len(s["mixed_true"]) for s in stats]
    doc_of = np.repeat(np.arange(len(stats)), sizes)
    true = tie_exact(np.concatenate([s["mixed_true"] for s in stats]))
    out = {}
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    rows_of = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(stats))]
    for name in present:
        score = tie_exact(np.concatenate([s["mixed_scores"][name] for s in stats]))
        draws = []
        for idx in plan[:200]:                         # 200 resamples: ~16k candidates per draw on GUM
            rows = np.concatenate([rows_of[i] for i in idx])
            draws.append(spearmanr(score[rows], true[rows]).statistic)
        ms = np.array([score[doc_of == i].mean() for i in range(len(stats))])
        mg = np.array([true[doc_of == i].mean() for i in range(len(stats))])
        rs = score - ms[doc_of]
        rg = true - mg[doc_of]
        out[name] = {"global_spearman": [float(spearmanr(score, true).statistic), *ci(draws)],
                     "between_doc": float(spearmanr(tie_exact(ms), tie_exact(mg)).statistic),
                     "within_doc": float(spearmanr(tie_exact(rs), tie_exact(rg)).statistic)}
    return out


# ------------------------------------------------------------------ 8.1 / 8.3 corpus level
def gold_candidate_report(cc: CorpusCtx) -> dict:
    per = [gold_vs_systems(ctx) for ctx in cc.ctx]
    names = [s for s in SCORERS if s in per[0]]
    out = {}
    for name in names:
        wins = np.array([p[name] for p in per])                    # (docs, 4)
        top = np.array([top1_credit(w) for w in wins])
        out[name] = {"gold_top1": float(top.mean()), "win_vs": dict(zip(SYSTEMS, wins.mean(0).tolist())),
                     "recognizes_gold": bool(top.mean() >= RECOGNIZE_GOLD)}
    return out


def error_pricing_report(cc: CorpusCtx, n_boot: int) -> dict:
    per = [doc_error_pricing(ctx) for ctx in cc.ctx]
    names = [s for s in SCORERS if s in per[0][(SYSTEMS[0], ERROR_TYPES[0])]["d_score"]]
    plan = resample_plan(np.array([c.genre for c in cc.ctx]), n_boot)
    # arrays (docs, systems, types)
    d_true = np.array([[[p[(s, e)]["d_true"] for e in ERROR_TYPES] for s in SYSTEMS] for p in per])
    count = np.array([[[p[(s, e)]["count"] for e in ERROR_TYPES] for s in SYSTEMS] for p in per])
    out = {"composition": {s: {e: {"mean_count": float(count[:, i, j].mean()), "mean_d_true": float(d_true[:, i, j].mean())}
                               for j, e in enumerate(ERROR_TYPES)} for i, s in enumerate(SYSTEMS)},
           "reward_ratio": {}}
    for name in names:
        d_s = np.array([[[p[(s, e)]["d_score"][name] for e in ERROR_TYPES] for s in SYSTEMS] for p in per])

        def ratio(idx, sys_axis=slice(None)):
            num, den = d_s[idx][:, sys_axis].sum((0, 1)), d_true[idx][:, sys_axis].sum((0, 1))
            return num / np.where(den == 0, np.nan, den)
        all_idx = np.arange(len(per))
        point = ratio(all_idx)
        draws = np.array([ratio(idx) for idx in plan])
        mo, mt = ERROR_TYPES.index("miss_other"), ERROR_TYPES.index("miss_terminal")
        out["reward_ratio"][name] = {
            "pooled": {e: [float(point[j]), *ci(draws[:, j])] for j, e in enumerate(ERROR_TYPES)},
            "miss_other_over_miss_terminal": [float(point[mo] / point[mt]), *ci(draws[:, mo] / draws[:, mt])],
            "per_system": {s: dict(zip(ERROR_TYPES, ratio(all_idx, slice(i, i + 1)).tolist())) for i, s in enumerate(SYSTEMS)},
        }
    total_loss = d_true.sum(2).mean(0)                                # mean F1 gain from fixing everything (approx.)
    share_other = d_true[:, :, ERROR_TYPES.index("miss_other")].mean(0) / np.where(total_loss == 0, np.nan, total_loss)
    out["miss_other_share"] = dict(zip(SYSTEMS, share_other.tolist()))
    st = out["reward_ratio"].get("Stack")
    if st:
        out["punkt_blind_spot_explained"] = bool(
            st["miss_other_over_miss_terminal"][0] < BLIND_SPOT_RATIO
            and max(out["miss_other_share"], key=out["miss_other_share"].get) == "punkt")
    return out


# ------------------------------------------------------------------ 8.4 selection
def selection_report(cc: CorpusCtx, names, n_boot: int) -> dict:
    items = cc.items
    docs = sorted(set(items.doc.tolist()))
    rows = {d: np.flatnonzero(items.doc == d) for d in docs}
    systems = items.system[rows[docs[0]]].tolist()
    if systems != list(SYSTEMS):
        raise RuntimeError("Unexpected system order")
    gold = np.array([items.gold[rows[d]] for d in docs])                # (docs, 4)
    genres = np.array([items.domain[rows[d][0]] for d in docs])
    plan = resample_plan(genres, n_boot)
    sat = SYSTEMS.index("sat")
    always_doc = gold.max(1) - gold[:, sat]
    out = {"always_sat": {"macro_regret": float(gold.mean(0).max() - gold.mean(0)[sat]),
                          "doc_regret": float(always_doc.mean())},
           "gold_macro_f1": dict(zip(SYSTEMS, gold.mean(0).tolist())), "scorers": {}}
    for name in names:
        sc = np.array([cc.scores[name][rows[d]] for d in docs])
        selected = int(np.argmax(sc.mean(0)))
        best = sc.max(1, keepdims=True)
        tied = np.abs(sc - best) <= TOL
        routed = (gold * tied).sum(1) / tied.sum(1)                      # ties: mean Gold F1 of the tied systems
        doc_regret = gold.max(1) - routed
        diff = doc_regret - always_doc
        out["scorers"][name] = {
            "corpus_selected": SYSTEMS[selected],
            "macro_regret": float(gold.mean(0).max() - gold.mean(0)[selected]),
            "doc_regret": float(doc_regret.mean()),
            "doc_regret_minus_always_sat": [float(diff.mean()), *ci([diff[idx].mean() for idx in plan])],
            "beats_always_sat": False}
        out["scorers"][name]["beats_always_sat"] = out["scorers"][name]["doc_regret_minus_always_sat"][2] < 0
    return out


# ------------------------------------------------------------------ 8.5 trivial baselines
def baseline_scores(cc: CorpusCtx) -> dict[str, np.ndarray]:
    out = defaultdict(list)
    for ctx in cc.ctx:
        agree = np.mean([hard_f1(ctx.votes[a], ctx.votes[b])[0]
                         for a in range(len(SYSTEMS)) for b in range(a + 1, len(SYSTEMS))])
        for s in range(len(SYSTEMS)):
            c = ctx.votes[s][None]
            out["LenPrior"].append(float(length_score(c, ctx.prior)[0]))
            out["Rule"].append(float(hard_f1(c, ctx.cats == TERMINAL)[0]))
            out["DocAgree"].append(float(agree))
    return {k: np.array(v) for k, v in out.items()}


def baselines_report(cc: CorpusCtx, n_boot: int, workers: int) -> dict:
    scores = {"S1": cc.scores["S1"], **baseline_scores(cc)}
    res = evaluate(with_scores(cc.items, scores, list(scores)), n_boot, SEED, "S1", workers)
    out = {}
    for name in scores:
        entry = {"global_spearman": [res["point"][name]["global_spearman"], *res["ci"][name]["global_spearman"]],
                 "within_doc_pairwise": res["point"][name]["within_doc_pairwise"],
                 **decomposition(cc.items, scores[name])}
        if name != "S1":
            lo, hi = res["paired_vs_baseline"][name]["global_spearman"]
            entry["S1_minus"] = [res["point"]["S1"]["global_spearman"] - res["point"][name]["global_spearman"], -hi, -lo]
        out[name] = entry
    return out


# ------------------------------------------------------------------ 8.6 stack ablation
ABLATIONS = {"stack": None, "stack-E1": 1, "stack-E2": 2}


def ablated_stack_gum(cc: CorpusCtx) -> dict[str, np.ndarray]:
    fold = logo_folds(cc.docs)
    out = {}
    for name, drop in ABLATIONS.items():
        keep = [j for j in range(cc.stack_x.shape[1]) if j != drop]
        x = cc.stack_x[:, keep]
        p = np.zeros(len(cc.y))
        for g in sorted(set(fold.tolist())):
            held = fold == g
            p[held] = predict_logistic_np(fit_logistic_np(x[~held], cc.y[~held], STACK_LAMBDA), x[held])
        out[name] = p
    return out


def ablated_stack_test(cc: CorpusCtx, gum: CorpusCtx) -> dict[str, np.ndarray]:
    out = {}
    for name, drop in ABLATIONS.items():
        keep = [j for j in range(gum.stack_x.shape[1]) if j != drop]
        model = fit_logistic_np(gum.stack_x[:, keep], gum.y, STACK_LAMBDA)
        out[name] = predict_logistic_np(model, cc.stack_x[:, keep])
    return out


def ablation_report(cc: CorpusCtx, ev: dict, n_boot: int, workers: int) -> dict:
    preds = predictions(CORPORA[cc.name])
    scores = {n: soft_scores(cc.docs, preds, p)[1] for n, p in ev.items()}
    res = evaluate(with_scores(cc.items, scores, list(scores)), n_boot, SEED, "stack", workers)
    fair = {n: fairness(cc.items, scores[n]) for n in scores}
    out = {"reproduces_stack": float(np.max(np.abs(ev["stack"] - cc.ev["stack"]))), "variants": {}}
    for n in scores:
        entry = {"global_spearman": [res["point"][n]["global_spearman"], *res["ci"][n]["global_spearman"]],
                 "rank_bias": fair[n]["rank_bias"], "corpus_ranking": fair[n]["corpus_ranking"]}
        if n != "stack":
            entry["minus_stack"] = [res["point"][n]["global_spearman"] - res["point"]["stack"]["global_spearman"],
                                    *res["paired_vs_baseline"][n]["global_spearman"]]
            entry["bias_change"] = {s: fair[n]["rank_bias"][s] - fair["stack"]["rank_bias"][s] for s in SYSTEMS}
        out["variants"][n] = entry
    ch1, ch2 = out["variants"]["stack-E1"]["bias_change"], out["variants"]["stack-E2"]["bias_change"]
    out["family_bias_flags"] = {"E1_sat": abs(ch1["sat"]) > FAMILY_MARGIN, "E1_wtp": abs(ch1["wtp"]) > FAMILY_MARGIN,
                                "E2_stanza": abs(ch2["stanza"]) > FAMILY_MARGIN}
    return out


# ------------------------------------------------------------------ 8.7 document length
def length_report(cc: CorpusCtx, per_doc_detect: np.ndarray) -> dict:
    n = np.array([c.n for c in cc.ctx])
    cuts = np.quantile(n, [1 / 3, 2 / 3])
    tercile = np.digitize(n, cuts)
    order = [c.doc_id for c in cc.ctx]
    st, mixed, r10 = SCORERS.index("Stack"), TYPES.index("mixed"), RATES.index(0.10)
    out = {"cuts": cuts.tolist()}
    for t, label in enumerate(("short", "medium", "long")):
        docs = {order[i] for i in np.flatnonzero(tercile == t)}
        idx = np.flatnonzero(np.isin(cc.items.doc, list(docs)))
        stat = statistics(with_scores(cc.items, cc.scores, ("S1",)), idx)["S1"]
        out[label] = {"documents": len(docs), "gap_range": [int(n[tercile == t].min()), int(n[tercile == t].max())],
                      "S1_global_spearman": stat["global_spearman"], "S1_within_doc_pairwise": stat["within_doc_pairwise"],
                      "stack_detect_mixed_r10": float(np.nanmean(per_doc_detect[tercile == t, st, mixed, r10]))}
    return out


# ------------------------------------------------------------------ rendering
LABEL = {"gum": "GUM 开发（LOGO）", "gentle": "GENTLE（测试后）", "lines": "LinES（测试后）"}
TYPE_LABEL = {"merge_terminal": "合并-终止", "merge_other": "合并-其他", "split_terminal": "多切-终止",
              "split_punct": "多切-标点", "split_nopunct": "多切-无标点", "shift": "错位", "mixed": "混合"}
ERR_LABEL = {"miss_terminal": "漏切-终止", "miss_other": "漏切-其他", "fp_terminal": "多切-终止",
             "fp_punct": "多切-标点", "fp_nopunct": "多切-无标点"}


def _f(v, sign=False) -> str:
    return "—" if v is None or v != v else (f"{v:+.3f}" if sign else f"{v:.3f}")


def _fci(v, sign=False) -> str:
    return f"{_f(v[0], sign)} [{_f(v[1], sign)}, {_f(v[2], sign)}]"


def render(report: dict) -> str:
    L = ["# 修订 8：补充有效性实验（只作分析）", "",
         "GUM = GUM train 的 LOGO 证据（开发集）；GENTLE / LinES 为测试后分析。PuncEv 只有 GUM（5 折交叉拟合证据）。"
         "CI：领域内按文档重采样（8.2 跨文档 Spearman 用 200 次，其余 1000 次）。", ""]
    for name, r in report["corpora"].items():
        L += [f"# {LABEL[name]}", ""]
        g = r["gold_candidate"]
        L += ["## 8.1 Gold 作为第 5 个候选", "", "| 评分 | Gold 排第一 | " + " | ".join(f"胜 {s}" for s in SYSTEMS) + " | 能认出 Gold |",
              "|---|---:|" + "---:|" * len(SYSTEMS) + "---|"]
        for n, v in g.items():
            L.append(f"| {n} | {v['gold_top1']:.3f} | " + " | ".join(f"{v['win_vs'][s]:.3f}" for s in SYSTEMS)
                     + f" | {v['recognizes_gold']} |")
        p = r["perturbation"]
        L += ["", f"## 8.2 人工扰动（{p['documents']} 篇；各类型可用文档："
              + "，".join(f"{TYPE_LABEL[t]} {k}" for t, k in p["available_docs"].items()) + "）", "",
              "### 识别率（Gold 分数高于扰动候选的比例），r = 0.10", "",
              "| 评分 | " + " | ".join(TYPE_LABEL[t] for t in TYPES) + " |", "|---|" + "---:|" * len(TYPES)]
        for n, v in p["per_scorer"].items():
            L.append(f"| {n} | " + " | ".join(_f(v["detect"][t]["0.1"][0]) for t in TYPES) + " |")
        L += ["", "### 识别率随扰动比例（混合类型）", "", "| 评分 | " + " | ".join(f"r = {r:g}" for r in RATES) + " |",
              "|---|" + "---:|" * len(RATES)]
        for n, v in p["per_scorer"].items():
            L.append(f"| {n} | " + " | ".join(_f(v["detect"]["mixed"][f"{r:g}"][0]) for r in RATES) + " |")
        L += ["", "### 敏感度（文档内分数对真实 F1 的斜率，理想 = 1）与不对称指数", "",
              "| 评分 | " + " | ".join(TYPE_LABEL[t] for t in TYPES) + " | 不对称指数 合并-其他/合并-终止 [CI] | 盲区 |",
              "|---|" + "---:|" * len(TYPES) + "---:|---|"]
        for n, v in p["per_scorer"].items():
            L.append(f"| {n} | " + " | ".join(_f(v["slope"][t][0]) for t in TYPES)
                     + f" | {_fci(v['asymmetry'])} | {v['blind_spot_merge_other']} |")
        L += ["", "### 单调性（同类型候选对中分数顺序与真实 F1 一致的比例）", "",
              "| 评分 | " + " | ".join(TYPE_LABEL[t] for t in TYPES) + " |", "|---|" + "---:|" * len(TYPES)]
        for n, v in p["per_scorer"].items():
            L.append(f"| {n} | " + " | ".join(_f(v["monotone"][t][0]) for t in TYPES) + " |")
        L += ["", "### 预先写定的配对比较（r = 0.10 识别率差 / 斜率差，[95% CI]）", ""]
        for comp, v in p["paired"].items():
            L += [f"**{comp}**", "", "| 类型 | 识别率差 | 斜率差 |", "|---|---:|---:|"]
            for t in TYPES:
                L.append(f"| {TYPE_LABEL[t]} | {_fci(v['detect_r10'][t], True)} | {_fci(v['slope'][t], True)} |")
            L.append("")
        L += [f"### 跨文档（混合类型全部候选 + Gold，共 {p['candidates']} 个）", "",
              "| 评分 | 全局 Spearman [CI] | 文档之间 | 文档之内 |", "|---|---:|---:|---:|"]
        for n, v in p["cross_document"].items():
            L.append(f"| {n} | {_fci(v['global_spearman'])} | {_f(v['between_doc'])} | {_f(v['within_doc'])} |")
        e = r["error_pricing"]
        L += ["", "## 8.3 真实系统错误的分类计价", "", "错误构成（每篇平均个数 / 修正该类后真实 F1 的平均增量）：", "",
              "| 系统 | " + " | ".join(ERR_LABEL[x] for x in ERROR_TYPES) + " | 漏切-其他占比 |", "|---|" + "---:|" * (len(ERROR_TYPES) + 1)]
        for s in SYSTEMS:
            c = e["composition"][s]
            L.append(f"| {s} | " + " | ".join(f"{c[x]['mean_count']:.1f} / {c[x]['mean_d_true']:+.3f}" for x in ERROR_TYPES)
                     + f" | {_f(e['miss_other_share'][s])} |")
        L += ["", "奖励比 = Σ评分增量 / Σ真实 F1 增量（理想 = 1；越小 = 该类错误越不被扣分）：", "",
              "| 评分 | " + " | ".join(ERR_LABEL[x] for x in ERROR_TYPES) + " | 漏切-其他 / 漏切-终止 [CI] |",
              "|---|" + "---:|" * (len(ERROR_TYPES) + 1)]
        for n, v in e["reward_ratio"].items():
            L.append(f"| {n} | " + " | ".join(_f(v["pooled"][x][0]) for x in ERROR_TYPES)
                     + f" | {_fci(v['miss_other_over_miss_terminal'])} |")
        if "Stack" in e["reward_ratio"]:
            L += ["", "Stack 奖励比按系统：", "", "| 系统 | " + " | ".join(ERR_LABEL[x] for x in ERROR_TYPES) + " |",
                  "|---|" + "---:|" * len(ERROR_TYPES)]
            for s, v in e["reward_ratio"]["Stack"]["per_system"].items():
                L.append(f"| {s} | " + " | ".join(_f(v[x]) for x in ERROR_TYPES) + " |")
        L += ["", f"Punkt 被高估由共同盲区解释：**{e.get('punkt_blind_spot_explained')}**", ""]
        s = r["selection"]
        L += ["## 8.4 选择遗憾（原始问题）", "", "Gold Macro F1：" + "，".join(f"{k} {v:.3f}" for k, v in s["gold_macro_f1"].items()),
              "", f"Always-SaT：Macro 遗憾 {s['always_sat']['macro_regret']:.4f}，逐文档遗憾 {s['always_sat']['doc_regret']:.4f}", "",
              "| 评分 | 语料级选中 | Macro 遗憾 | 逐文档路由遗憾 | − Always-SaT [CI] | 优于 Always-SaT |", "|---|---|---:|---:|---:|---|"]
        for n, v in s["scorers"].items():
            L.append(f"| {n} | {v['corpus_selected']} | {v['macro_regret']:.4f} | {v['doc_regret']:.4f} | "
                     f"{_fci(v['doc_regret_minus_always_sat'], True)} | {v['beats_always_sat']} |")
        b = r["baselines"]
        L += ["", "## 8.5 简单基线（真实系统条目）", "", "| 评分 | 全局 Spearman [CI] | 文档之间 | 文档之内 | 文档内配对 | S1 − 基线 [CI] |",
              "|---|---:|---:|---:|---:|---:|"]
        for n, v in b.items():
            L.append(f"| {n} | {_fci(v['global_spearman'])} | {_f(v['between_doc'])} | {_f(v['within_doc'])} | "
                     f"{_f(v['within_doc_pairwise'])} | {_fci(v['S1_minus'], True) if 'S1_minus' in v else '—'} |")
        a = r["ablation"]
        L += ["", f"## 8.6 Stack 去掉单一证据（复现 stack 的最大绝对误差 {a['reproduces_stack']:.2e}）", "",
              "| 变体 | 全局 Spearman [CI] | − stack [CI] | " + " | ".join(f"残差 {s}" for s in SYSTEMS) + " | 语料级排名 |",
              "|---|---:|---:|" + "---:|" * len(SYSTEMS) + "---|"]
        for n, v in a["variants"].items():
            L.append(f"| {n} | {_fci(v['global_spearman'])} | {_fci(v['minus_stack'], True) if 'minus_stack' in v else '—'} | "
                     + " | ".join(f"{v['rank_bias'][s]:+.3f}" for s in SYSTEMS) + f" | {' > '.join(v['corpus_ranking'])} |")
        L += ["", f"同源偏差判定（残差变化绝对值 > {FAMILY_MARGIN}）：{a['family_bias_flags']}", ""]
        if "length" in r:
            ln = r["length"]
            L += ["## 8.7 文档长度（GUM 空隙数三等分）", "", "| 组 | 文档数 | 空隙数范围 | S1 全局 Spearman | S1 文档内配对 | Stack 识别率（混合，r = 0.10） |",
                  "|---|---:|---|---:|---:|---:|"]
            for k in ("short", "medium", "long"):
                v = ln[k]
                L.append(f"| {k} | {v['documents']} | {v['gap_range'][0]}–{v['gap_range'][1]} | {v['S1_global_spearman']:.3f} | "
                         f"{v['S1_within_doc_pairwise']:.3f} | {v['stack_detect_mixed_r10']:.3f} |")
            L.append("")
    return "\n".join(L) + "\n"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


# ------------------------------------------------------------------ main
def corpus_report(cc: CorpusCtx, ablation_ev: dict, args) -> dict:
    names = [n for n in ("PuncEv", "S0", "S1", "M0", "M3") if n in cc.scores]
    out = {"gold_candidate": gold_candidate_report(cc)}
    print(json.dumps({"status": "A8_STEP", "corpus": cc.name, "step": "8.1"}), flush=True)
    out["perturbation"] = perturbation_report(cc, args.bootstrap)
    print(json.dumps({"status": "A8_STEP", "corpus": cc.name, "step": "8.2"}), flush=True)
    out["error_pricing"] = error_pricing_report(cc, args.bootstrap)
    out["selection"] = selection_report(cc, names, args.bootstrap)
    out["baselines"] = baselines_report(cc, args.bootstrap, args.workers)
    out["ablation"] = ablation_report(cc, ablation_ev, args.bootstrap, args.workers)
    per_doc = out["perturbation"].pop("_per_doc_detect")
    if cc.name == "gum":
        out["length"] = length_report(cc, per_doc)
    print(json.dumps({"status": "A8_CORPUS_DONE", "corpus": cc.name}), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="*", default=["gum", "gentle", "lines"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    gum = gum_corpus()
    dev = json.loads((ROOT / "results/a6/dev_report.json").read_text())
    s1 = statistics(with_scores(gum.items, gum.scores, ("S1",)))["S1"]["global_spearman"]
    report = {"note": "amendment 8; analysis only; GENTLE / LinES are post-test",
              "gum_reproduces_dev": bool(abs(s1 - dev["S"]["point"]["S1"]["global_spearman"]) < 1e-9), "corpora": {}}
    for name in args.corpora:
        cc = gum if name == "gum" else test_corpus(name, final)
        ev = ablated_stack_gum(cc) if name == "gum" else ablated_stack_test(cc, gum)
        report["corpora"][name] = corpus_report(cc, ev, args)
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = _jsonable(report)
    (RESULTS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "summary.md").write_text(render(report))
    print(json.dumps({"status": "A8_COMPLETE", "gum_reproduces_dev": report["gum_reproduces_dev"]}), flush=True)


if __name__ == "__main__":
    main()
