"""Amendment 11: post-hoc analyses for the paper (labels D on GUM LOGO, R/P elsewhere; no decisions).

  evidence  held-out per-gap evidence of every evaluator on the ladder, cached in results/a11/evidence/
  analyze   E1 system-ranking agreement, E2 scoring formula, E3 real-error pricing, perturbation test v2,
            E4 discordant examples -> results/a11/{report.json, summary.md, examples.md}

Ladder (flat):   PuncEv, Probe, Stack, +casing, final (B-qb on ref91 features)
Ladder (layout): Probe, Stack, +casing, +layout, final (B-qb on lower-nl features)
References: consensus (official M0 for E1/E2; gap-level leave-one-out majority for E3 / perturbations),
hard pseudo-reference of the final evaluator, DocAgree (E1 only).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from .a6_combine import FINAL, STACK_LAMBDA, consensus_scores, dev_evidence, e0_evidence, full_boundaries
from .a6_data import CORPORA, ROOT, SEED, SYSTEMS, doc_slices, gold_sets, jsonl, predictions, surfaces
from .a6_signals import fit_logistic_np, predict_logistic_np
from .a8_supplement import BASE_TYPES, NOPUNCT, PUNCT, RATES, REPLICATES, TERMINAL, TYPES, _puncev_gum, apply_op, \
    boot_mean, category_codes, consensus_refs, credit, hard_f1, n_ops, ols_slope, resample_plan, soft_f1
from .a9_casing import A9, logo_stack
from .a9_layout import NL, bootstrap, break_mask, contrast, doc_agree
from .a10_quality import B_VARIANTS, SETTINGS, QualityProblem, fit_quality, load, prob, standardize
from .meta_eval import max_soft_f1
from .quality_agreement import tie_exact

RESULTS = ROOT / "results/a11"
CACHE = RESULTS / "evidence"
NAMES = ("gum", "gentle", "lines", "ewt")
EWT_PUNCEV = ROOT / "data/real_system_ewt/evidence/raw/platt.jsonl"
LADDER = {"flat": ("PuncEv", "Probe", "Stack", "+casing", "final"),
          "nl": ("Probe", "Stack", "+casing", "+layout", "final")}
# evidence key behind each ladder name, per scenario
EVIDENCE_OF = {"flat": {"PuncEv": "PuncEv", "Probe": "Probe", "Stack": "Stack", "+casing": "+casing", "final": "final_flat"},
               "nl": {"Probe": "Probe", "Stack": "Stack", "+casing": "+casing", "+layout": "+layout", "final": "final_layout"}}
SCENARIOS = {"gum": ("flat", "nl"), "gentle": ("flat", "nl"), "lines": ("flat",), "ewt": ("flat", "nl")}
TYPES2 = TYPES + ("merge_other_par", "merge_other_inpar")       # indices 0-6 = amendment 8.2 (same seeds)
LEVELS = tuple(f"{r:.2f}" for r in RATES) + ("k=1",)            # level index 4 = exactly one error
ERRORS = ("miss_terminal", "miss_other_par", "miss_other_inpar", "fp_terminal", "fp_punct", "fp_nopunct")
TOL = 1e-12


# ------------------------------------------------------------------ stage 1: evidence
def _fold_quality(torch, data, weights) -> np.ndarray:
    """Held-out B-qb evidence on GUM: per genre, warm-start from the fold's BCE stack (amendment 10 recipe)."""
    sl = doc_slices(data.docs)
    genres = sorted(set(data.genres.tolist()))
    doc_fold = np.array([genres.index(g) for g in data.genres])
    out = np.zeros(len(data.y))
    for g in range(len(genres)):
        held = data.fold == g
        tr_doc = np.flatnonzero(doc_fold != g)
        tr_gap = np.concatenate([np.arange(sl[i].start, sl[i].stop) for i in tr_doc])
        outer = fit_logistic_np(data.x[~held], data.y[~held], STACK_LAMBDA)
        problem = QualityProblem(torch, standardize(outer, data.x[tr_gap]), data.y[tr_gap], data, tr_doc)
        theta = fit_quality(torch, problem, np.append(outer["w"], outer["b"]), weights)
        out[held] = prob(outer, theta, data.x[held])
    return out


def _final_quality(torch, data, weights):
    model = fit_logistic_np(data.x, data.y, STACK_LAMBDA)
    problem = QualityProblem(torch, standardize(model, data.x), data.y, data, np.arange(len(data.docs)))
    return model, fit_quality(torch, problem, np.append(model["w"], model["b"]), weights)


def build_evidence(args) -> None:
    import torch
    torch.set_num_threads(args.threads)
    started = time.time()
    gum_docs = surfaces(CORPORA["gum"])
    ev_gum, extras = dev_evidence(CORPORA["gum"], gum_docs, gold_sets(CORPORA["gum"]))
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    ctx = {"ev_gum": ev_gum, "extras": extras, "final": final}
    weights = B_VARIANTS["B-qb"]
    CACHE.mkdir(parents=True, exist_ok=True)
    models = {}
    data_by = {}
    for name in NAMES:
        data_by[name] = {t: load(name, SETTINGS[t], ctx) for t in ("T1", "T2", "T3")}
        print(json.dumps({"status": "A11_LOADED", "corpus": name, "seconds": round(time.time() - started, 1)}), flush=True)
    gum = data_by["gum"]
    for t in ("T1", "T2", "T3"):
        models[t] = fit_logistic_np(gum[t].x, gum[t].y, STACK_LAMBDA)
    for t in ("T2", "T3"):
        models[f"q{t}"] = _final_quality(torch, gum[t], weights)
    for name in NAMES:
        d = data_by[name]
        docs = d["T1"].docs
        ev = {}
        if name == "gum":
            ev["PuncEv"] = _puncev_gum(docs)
            ev["Probe"] = ev_gum["E0"]
            ev["Stack"] = logo_stack(d["T1"].x, d["T1"].y, d["T1"].fold)
            ev["+casing"] = logo_stack(d["T2"].x, d["T2"].y, d["T2"].fold)
            ev["+layout"] = logo_stack(d["T3"].x, d["T3"].y, d["T3"].fold)
            ev["final_flat"] = _fold_quality(torch, d["T2"], weights)
            ev["final_layout"] = _fold_quality(torch, d["T3"], weights)
        else:
            if name == "ewt":
                ev["PuncEv"] = e0_evidence(EWT_PUNCEV, docs)
            ev["Probe"] = (e0_evidence(ROOT / "data/real_system_ewt/evidence/raw/probe+lr.jsonl", docs) if name == "ewt" else
                           np.concatenate([q["calibrated_boundary_probability"]
                                           for q in jsonl(CORPORA[name].path("evidence", "E0.jsonl"))]))
            ev["Stack"] = predict_logistic_np(models["T1"], d["T1"].x)
            ev["+casing"] = predict_logistic_np(models["T2"], d["T2"].x)
            ev["+layout"] = predict_logistic_np(models["T3"], d["T3"].x)
            for key, t in (("final_flat", "T2"), ("final_layout", "T3")):
                model, theta = models[f"q{t}"]
                ev[key] = prob(model, theta, d[t].x)
        gold = gold_sets(A9[name])
        per = {"y": np.concatenate([[o in gold[x.doc_id] for o in x.candidate_offsets] for x in docs]).astype(bool),
               "par": break_mask(name, docs).astype(bool),
               "cats": np.concatenate([category_codes(x.text, x.candidate_offsets) for x in docs]),
               "doc": np.array([x.doc_id for x in docs]), "genre": np.array([x.genre for x in docs]),
               "gaps": np.array([len(x.candidate_offsets) for x in docs])}
        for scen, t in (("flat", "T1"), ("nl", "T3")):
            data = d[t]
            per[f"{scen}__cmat"] = np.concatenate(data.cmat, axis=1) if len(data.cmat) else np.zeros((4, 0), bool)
            per[f"{scen}__f1"] = data.f1
            corpus = A9[name] if scen == "flat" else NL[name]
            sdocs = docs if scen == "flat" else surfaces(NL[name])
            preds = predictions(corpus)
            per[f"{scen}__M0"] = consensus_scores(sdocs, full_boundaries(corpus, preds))
            per[f"{scen}__DocAgree"] = doc_agree(sdocs, preds)
        np.savez_compressed(CACHE / f"{name}.npz", **{f"ev__{k}": v for k, v in ev.items()}, **per)
        print(json.dumps({"status": "A11_EVIDENCE", "corpus": name, "keys": sorted(ev),
                          "seconds": round(time.time() - started, 1)}), flush=True)


# ------------------------------------------------------------------ helpers
class Corpus:
    """Cached arrays of one corpus, with per-document views."""

    def __init__(self, name: str):
        z = np.load(CACHE / f"{name}.npz")
        self.name = name
        self.ev = {k[4:]: z[k] for k in z.files if k.startswith("ev__")}
        self.y, self.par, self.cats = z["y"], z["par"], z["cats"]
        self.doc, self.genre, self.gaps = z["doc"], z["genre"], z["gaps"]
        self.sl = [slice(a, a + n) for a, n in zip(np.concatenate([[0], np.cumsum(self.gaps)[:-1]]), self.gaps)]
        self.scen = {s: {"cmat": z[f"{s}__cmat"], "f1": z[f"{s}__f1"], "M0": z[f"{s}__M0"], "DocAgree": z[f"{s}__DocAgree"]}
                     for s in SCENARIOS[name]}

    def docs(self):
        return range(len(self.doc))


def item_scores(c: Corpus, scen: str, p: np.ndarray, kind: str = "norm") -> np.ndarray:
    """Scores of the four systems' outputs in every document (document-major, SYSTEMS order)."""
    cm = c.scen[scen]["cmat"]
    out = []
    for sl in c.sl:
        rows, q = cm[:, sl], p[sl]
        if not len(q):
            out.extend([1.0] * len(SYSTEMS))
        elif kind == "hard":
            out.extend(hard_f1(rows, q >= 0.5))
        elif kind == "norm_end":
            out.extend(soft_f1_end(rows, q))
        else:
            s = soft_f1(rows, q)
            out.extend(s / max_soft_f1(tuple(q)) if kind == "norm" else s)
    return np.array(out)


def soft_f1_end(rows: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Normalized Soft F1 with the document-final boundary counted (always correct for every candidate and the
    evidence): Q_end = 2(y.p + 1) / (|y| + |p| + 2); maximum over top-k prefixes, k = 0..n (k = 0: no split).
    Post-hoc variant (amendment 11 addendum): an empty candidate in a document without any boundary scores ~1."""
    rows = np.atleast_2d(rows).astype(float)
    total = q.sum()
    score = 2 * (rows @ q + 1) / (rows.sum(1) + total + 2)
    srt = np.sort(q)[::-1]
    k = np.arange(1, len(q) + 1)
    best = max(2 / (total + 2), float(np.max(2 * (np.cumsum(srt) + 1) / (k + total + 2))) if len(q) else 0.0)
    return score / best


def ladder_scores(c: Corpus, scen: str) -> dict:
    out = {}
    for name in LADDER[scen]:
        key = EVIDENCE_OF[scen][name]
        if key in c.ev:
            out[name] = item_scores(c, scen, c.ev[key])
    return out


# ------------------------------------------------------------------ E1 system ranking
def tau(a, b) -> float:
    a, b = tie_exact(a), tie_exact(b)
    if len(set(a.tolist())) < 2 or len(set(b.tolist())) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(kendalltau(a, b).statistic)


def system_means(v: np.ndarray, rows: np.ndarray) -> np.ndarray:
    m = v[rows].reshape(-1, len(SYSTEMS))
    return m.mean(0)


def top1_hit(score_means: np.ndarray, gold_means: np.ndarray) -> float:
    """Tie-aware top-1 agreement (review fix 2026-11-20; before: np.argmax, i.e. ties settled by SYSTEMS order):
    the expected hit when a tie among the top-scored systems is broken uniformly at random; any Gold-best
    system counts as a hit."""
    a, b = tie_exact(score_means), tie_exact(gold_means)
    picked = np.flatnonzero(a == a.max())
    return float(np.mean(b[picked] == b.max()))


def ranking_stats(score: np.ndarray, f1: np.ndarray, genre: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(corpus tau, genre-mean tau, corpus top-1 match, genre top-1 rate) on the resampled documents idx."""
    rows = (idx[:, None] * len(SYSTEMS) + np.arange(len(SYSTEMS))).ravel()
    ms, mg = system_means(score, rows), system_means(f1, rows)
    t_c = tau(ms, mg)
    top_c = top1_hit(ms, mg)
    taus, tops = [], []
    for g in sorted(set(genre[idx].tolist())):
        sub = idx[genre[idx] == g]
        r = (sub[:, None] * len(SYSTEMS) + np.arange(len(SYSTEMS))).ravel()
        a, b = system_means(score, r), system_means(f1, r)
        taus.append(tau(a, b))
        tops.append(top1_hit(a, b))
    return np.array([t_c, np.nanmean(taus) if np.any(np.isfinite(taus)) else np.nan, top_c, np.mean(tops)])


def doc_taus(score: np.ndarray, f1: np.ndarray, n_docs: int) -> np.ndarray:
    s, g = score.reshape(n_docs, len(SYSTEMS)), f1.reshape(n_docs, len(SYSTEMS))
    return np.array([tau(a, b) for a, b in zip(s, g)])


def e1(c: Corpus, scen: str, scores: dict, n_boot: int) -> dict:
    f1 = c.scen[scen]["f1"]
    n = len(c.doc)
    plan = resample_plan(c.genre, n_boot)
    allidx = np.arange(n)
    out, draws = {}, {}
    for name, v in scores.items():
        point = ranking_stats(v, f1, c.genre, allidx)
        dt = doc_taus(v, f1, n)
        dr = np.array([ranking_stats(v, f1, c.genre, idx) for idx in plan])
        dd = np.array([np.nanmean(dt[idx]) for idx in plan])
        draws[name] = np.column_stack([dr, dd])
        q = lambda col: [float(np.nanpercentile(draws[name][:, col], 2.5)), float(np.nanpercentile(draws[name][:, col], 97.5))]
        out[name] = {"corpus_tau": [float(point[0]), *q(0)], "genre_tau": [float(point[1]), *q(1)],
                     "corpus_top1": float(point[2]), "genre_top1": [float(point[3]), *q(3)],
                     "doc_tau": [float(np.nanmean(dt)), *q(4)], "doc_tau_defined": int(np.isfinite(dt).sum()),
                     "documents": n}
    pairs = {}
    for a, b in (("final", "Probe"), ("final", "M0"), ("final", "Stack"), ("final (+end)", "M0")):
        if a in draws and b in draws:
            d = draws[a] - draws[b]
            pairs[f"{a} - {b}"] = {m: [float(out[a][m][0] - out[b][m][0]), float(np.nanpercentile(d[:, j], 2.5)),
                                       float(np.nanpercentile(d[:, j], 97.5))]
                                   for m, j in (("genre_tau", 1), ("doc_tau", 4))}
    return {"scorers": out, "paired": pairs}


# ------------------------------------------------------------------ E2 scoring formula
def e2(c: Corpus, scen: str, n_boot: int) -> dict:
    f1 = c.scen[scen]["f1"]
    out = {}
    for name in LADDER[scen]:
        key = EVIDENCE_OF[scen][name]
        if key not in c.ev:
            continue
        sc = {k: item_scores(c, scen, c.ev[key], k) for k in ("norm", "raw", "hard", "norm_end")}
        point, draws, names = bootstrap(sc, f1, c.genre, n_boot)
        out[name] = {k: {"global_spearman": float(point[k][0]), "within_doc_pairwise": float(point[k][1])} for k in sc}
        out[name]["norm_minus_raw"] = contrast(draws, names, point, ["norm"], ["raw"])
        out[name]["norm_minus_hard"] = contrast(draws, names, point, ["norm"], ["hard"])
        out[name]["end_minus_norm"] = contrast(draws, names, point, ["norm_end"], ["norm"])
        out[name]["end_minus_hard"] = contrast(draws, names, point, ["norm_end"], ["hard"])
    return out


# ------------------------------------------------------------------ E3 real-error pricing
def error_masks2(c_row: np.ndarray, gold: np.ndarray, cats: np.ndarray, par: np.ndarray) -> dict:
    miss, fp = gold & ~c_row, c_row & ~gold
    other = cats != TERMINAL
    return {"miss_terminal": miss & ~other, "miss_other_par": miss & other & par, "miss_other_inpar": miss & other & ~par,
            "fp_terminal": fp & (cats == TERMINAL), "fp_punct": fp & (cats == PUNCT), "fp_nopunct": fp & (cats == NOPUNCT)}


def e3(c: Corpus, scen: str, n_boot: int) -> dict:
    cm = c.scen[scen]["cmat"]
    names = [n for n in LADDER[scen] if EVIDENCE_OF[scen][n] in c.ev] + ["Consensus"]
    d_true = np.zeros((len(c.doc), len(ERRORS)))
    d_score = np.zeros((len(names), len(c.doc), len(ERRORS)))
    count = np.zeros((len(c.doc), len(ERRORS)))
    for i, sl in enumerate(c.sl):
        g, cats, par = c.y[sl], c.cats[sl], c.par[sl]
        votes = cm[:, sl]
        refs = consensus_refs(votes)
        for s in range(len(SYSTEMS)):
            row = votes[s]
            masks = error_masks2(row, g, cats, par)
            mat = [row]
            for e in ERRORS:
                fixed = row.copy()
                fixed[masks[e]] = e.startswith("miss")
                mat.append(fixed)
            mat = np.array(mat)
            true = hard_f1(mat, g)
            d_true[i] += true[1:] - true[0]
            count[i] += [masks[e].sum() for e in ERRORS]
            for k, n in enumerate(names):
                if n == "Consensus":
                    sc = hard_f1(mat, refs[s])
                else:
                    q = c.ev[EVIDENCE_OF[scen][n]][sl]
                    sc = soft_f1(mat, q) / (max_soft_f1(tuple(q)) if len(q) else 1.0)
                d_score[k, i] += sc[1:] - sc[0]
    plan = resample_plan(c.genre, n_boot)
    out = {"counts": dict(zip(ERRORS, count.sum(0).astype(int).tolist())),
           "true_gain": dict(zip(ERRORS, d_true.sum(0).tolist())), "ratio": {}}
    for k, n in enumerate(names):
        with np.errstate(divide="ignore", invalid="ignore"):
            point = d_score[k].sum(0) / d_true.sum(0)
            draws = np.array([d_score[k][idx].sum(0) / d_true[idx].sum(0) for idx in plan])
        out["ratio"][n] = {e: [float(point[j]), float(np.nanpercentile(draws[:, j], 2.5)), float(np.nanpercentile(draws[:, j], 97.5))]
                           for j, e in enumerate(ERRORS)}
    return out


# ------------------------------------------------------------------ perturbation test v2
def apply_op2(c: np.ndarray, gold: np.ndarray, cats: np.ndarray, par: np.ndarray, op: str, rng) -> bool:
    if op in ("merge_other_par", "merge_other_inpar"):
        where = (cats != TERMINAL) & (par if op == "merge_other_par" else ~par)
        pool = np.flatnonzero(c & gold & where)
        if not len(pool):
            return False
        c[pool[rng.integers(len(pool))]] = False
        return True
    return apply_op(c, gold, cats, op, rng)


def candidates2(gold: np.ndarray, cats: np.ndarray, par: np.ndarray, doc_index: int):
    """Gold (row 0) + perturbed candidates; types 0-6 and levels 0-3 reproduce amendment 8.2 exactly."""
    rows, t_idx, l_idx = [gold.copy()], [-1], [-1]
    n_gold = int(gold.sum())
    for t, kind in enumerate(TYPES2):
        for lv in range(len(LEVELS)):
            rng = np.random.default_rng([SEED, doc_index, t, lv])
            k = 1 if LEVELS[lv] == "k=1" else n_ops(RATES[lv], n_gold)
            for _ in range(REPLICATES):
                c, done = gold.copy(), 0
                for _ in range(k):
                    op = BASE_TYPES[rng.integers(len(BASE_TYPES))] if kind == "mixed" else kind
                    done += apply_op2(c, gold, cats, par, op, rng)
                if done:
                    rows.append(c)
                    t_idx.append(t)
                    l_idx.append(lv)
    return np.array(rows), np.array(t_idx), np.array(l_idx)


PERTURB_SCORERS = ("PuncEv", "Probe", "Stack", "+casing", "+layout", "final (flat)", "final (layout)",
                   "H(final)", "Consensus")
PERTURB_EVIDENCE = {"PuncEv": "PuncEv", "Probe": "Probe", "Stack": "Stack", "+casing": "+casing", "+layout": "+layout",
                    "final (flat)": "final_flat", "final (layout)": "final_layout"}


def perturbation(c: Corpus, n_boot: int, kind: str = "Q_end") -> dict:
    """Controlled error injection. Soft scorers use ``kind``: "Q_end" (adopted scorer, amendment 11 addendum;
    review fix 2026-11-20 — before, this test always used Q without the document-end term) or "Q"."""
    scen = "nl" if "nl" in c.scen else "flat"
    names = [n for n in PERTURB_SCORERS if n in ("H(final)", "Consensus") or PERTURB_EVIDENCE[n] in c.ev]
    if "lines" == c.name:
        names = [n for n in names if n not in ("+layout", "final (layout)")]
    detect = np.full((len(c.doc), len(names), len(TYPES2), len(LEVELS)), np.nan)
    slope = np.full((len(c.doc), len(names), len(TYPES2)), np.nan)
    hfinal = "final_layout" if scen == "nl" and c.name != "lines" else "final_flat"
    for i, sl in enumerate(c.sl):
        g, cats, par = c.y[sl], c.cats[sl], c.par[sl]
        if not len(g) or not g.any():
            continue
        cand, t_idx, l_idx = candidates2(g, cats, par, i)
        true = hard_f1(cand, g)
        refs = consensus_refs(c.scen[scen]["cmat"][:, sl])
        for k, n in enumerate(names):
            if n == "Consensus":
                s = np.mean([hard_f1(cand, refs[r]) for r in range(len(SYSTEMS))], axis=0)
            elif n == "H(final)":
                s = hard_f1(cand, c.ev[hfinal][sl] >= 0.5)
            else:
                q = c.ev[PERTURB_EVIDENCE[n]][sl]
                s = soft_f1_end(cand, q) if kind == "Q_end" else soft_f1(cand, q) / max_soft_f1(tuple(q))
            s = tie_exact(s)
            for t in range(len(TYPES2)):
                rows_t = np.flatnonzero(t_idx == t)
                if not len(rows_t):
                    continue
                slope[i, k, t] = ols_slope(true[np.concatenate([[0], rows_t])], s[np.concatenate([[0], rows_t])])
                for lv in range(len(LEVELS)):
                    r = rows_t[l_idx[rows_t] == lv]
                    if len(r):
                        detect[i, k, t, lv] = credit(s[0] - s[r]).mean()
    plan = resample_plan(c.genre, n_boot)
    det = boot_mean(detect.reshape(len(c.doc), -1), plan)
    slo = boot_mean(slope.reshape(len(c.doc), -1), plan)
    shape_d, shape_s = (len(names), len(TYPES2), len(LEVELS)), (len(names), len(TYPES2))
    pack = lambda b, shape: {k: np.asarray(b[k]).reshape(shape).tolist() for k in ("point", "lo", "hi")}
    return {"scenario_for_consensus": scen, "soft_scorer": kind, "scorers": names, "types": list(TYPES2), "levels": list(LEVELS),
            "documents_with_type": dict(zip(TYPES2, np.isfinite(slope[:, 0, :]).sum(0).astype(int).tolist())),
            "detect": pack(det, shape_d), "slope": pack(slo, shape_s)}


# ------------------------------------------------------------------ E4 examples
CAT_NAME = {TERMINAL: "terminal", PUNCT: "comma/colon/dash", NOPUNCT: "no punct."}


def examples(c: Corpus, scen: str, limit: int = 25) -> list[dict]:
    key = EVIDENCE_OF[scen]["final"]
    if key not in c.ev:
        return []
    score = tie_exact(item_scores(c, scen, c.ev[key]))
    f1 = c.scen[scen]["f1"]
    texts = {d.doc_id: d.text for d in surfaces(A9[c.name])}
    offsets = {d.doc_id: d.candidate_offsets for d in surfaces(A9[c.name])}
    cand = []
    for i, sl in enumerate(c.sl):
        for a in range(len(SYSTEMS)):
            for b in range(len(SYSTEMS)):
                ia, ib = i * len(SYSTEMS) + a, i * len(SYSTEMS) + b
                if f1[ia] > f1[ib] + 1e-9 and score[ia] < score[ib]:
                    cand.append((f1[ia] - f1[ib], i, a, b))
    cand.sort(reverse=True)
    out = []
    for gap_f1, i, a, b in cand[:limit]:
        sl, doc = c.sl[i], c.doc[i]
        cm = c.scen[scen]["cmat"][:, sl]
        diff = np.flatnonzero(cm[a] != cm[b])
        text, offs = texts[doc], offsets[doc]
        gaps = []
        for j in diff[:8]:
            o = offs[j]
            gaps.append({"left": text[max(0, o - 60):o].replace("\n", " ⏎ "), "right": text[o:o + 60].replace("\n", " ⏎ "),
                         "gold": bool(c.y[sl][j]), "p": round(float(c.ev[key][sl][j]), 3),
                         "left_punct": CAT_NAME[int(c.cats[sl][j])], "paragraph_break": bool(c.par[sl][j]),
                         "cut_by": SYSTEMS[a] if cm[a, j] else SYSTEMS[b]})
        out.append({"doc": str(doc), "genre": str(c.genre[i]), "better_by_gold": SYSTEMS[a], "worse_by_gold": SYSTEMS[b],
                    "gold_f1": [round(float(f1[i * len(SYSTEMS) + a]), 3), round(float(f1[i * len(SYSTEMS) + b]), 3)],
                    "score": [round(float(score[i * len(SYSTEMS) + a]), 4), round(float(score[i * len(SYSTEMS) + b]), 4)],
                    "differing_gaps": int(len(diff)), "gaps": gaps})
    return out


# ------------------------------------------------------------------ stage 2: analysis
def analyze(args) -> None:
    report = {"note": "amendment 11; GUM = LOGO (D), other corpora R/P; no decisions", "corpora": {}}
    ex_all = {}
    for name in args.corpora:
        started = time.time()
        c = Corpus(name)
        out = {"documents": len(c.doc), "scenarios": {}}
        for scen in SCENARIOS[name]:
            sc = ladder_scores(c, scen)
            check = {n: float(np.round(bootstrap({n: v}, c.scen[scen]["f1"], c.genre, 1)[0][n][0], 4)) for n, v in sc.items()}
            sc["M0"] = c.scen[scen]["M0"]
            sc["DocAgree"] = c.scen[scen]["DocAgree"]
            sc["H(final)"] = item_scores(c, scen, c.ev[EVIDENCE_OF[scen]["final"]], "hard")
            sc["final (+end)"] = item_scores(c, scen, c.ev[EVIDENCE_OF[scen]["final"]], "norm_end")
            out["scenarios"][scen] = {"global_spearman_check": check, "E1": e1(c, scen, sc, args.bootstrap),
                                      "E2": e2(c, scen, args.bootstrap), "E3": e3(c, scen, args.bootstrap)}
        out["perturbation"] = perturbation(c, args.bootstrap)
        scen_ex = "nl" if "nl" in c.scen else "flat"
        ex_all[name] = {"scenario": scen_ex, "pairs": examples(c, scen_ex)}
        report["corpora"][name] = out
        print(json.dumps({"status": "A11_CORPUS_DONE", "corpus": name, "seconds": round(time.time() - started, 1),
                          "check": {s: v["global_spearman_check"] for s, v in out["scenarios"].items()}}), flush=True)
        save(report, ex_all)
    save(report, ex_all)
    print(json.dumps({"status": "A11_COMPLETE"}), flush=True)


def save(report: dict, ex_all: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "summary.md").write_text(render(report))
    (RESULTS / "examples.md").write_text(render_examples(ex_all))


def _ci(v, sign=False) -> str:
    f = (lambda x: f"{x:+.3f}") if sign else (lambda x: f"{x:.3f}")
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"


def render(r: dict) -> str:
    L = ["# 修订 11：论文正文的补充分析（GUM = LOGO，D；其余语料 R/P；不做取舍）", ""]
    for name, c in r["corpora"].items():
        L += [f"# {name}（{c['documents']} 篇）", ""]
        for scen, s in c["scenarios"].items():
            L += [f"## 场景：{'带版式' if scen == 'nl' else '无换行'}", "", "全局 Spearman 核对：" + json.dumps(s["global_spearman_check"]), "",
                  "**E1 系统排名一致性**（Kendall τ_b；领域层面为预先指定的整体分数）", "",
                  "| 评分 | 语料 τ | 领域平均 τ | 文档平均 τ | 语料第一名一致 | 领域第一名一致率 |", "|---|---:|---:|---:|---:|---:|"]
            for n, v in s["E1"]["scorers"].items():
                L.append(f"| {n} | {_ci(v['corpus_tau'])} | {_ci(v['genre_tau'])} | {_ci(v['doc_tau'])} | {v['corpus_top1']:.0f} | {_ci(v['genre_top1'])} |")
            for k, v in s["E1"]["paired"].items():
                L.append(f"\n配对 {k}：领域平均 τ {_ci(v['genre_tau'], True)}，文档平均 τ {_ci(v['doc_tau'], True)}")
            L += ["", "**E2 评分公式**（全局 Spearman；配对差；“+末尾”= 计入文档末尾边界的事后变体）", "",
                  "| 证据 | 归一化 | 原始 Soft F1 | 硬伪参考 | 归一化 +末尾 | 归一化 − 原始 | 归一化 − 硬 | +末尾 − 归一化 | +末尾 − 硬 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for n, v in s["E2"].items():
                L.append(f"| {n} | {v['norm']['global_spearman']:.3f} | {v['raw']['global_spearman']:.3f} | {v['hard']['global_spearman']:.3f} | "
                         f"{v['norm_end']['global_spearman']:.3f} | {_ci(v['norm_minus_raw'], True)} | {_ci(v['norm_minus_hard'], True)} | "
                         f"{_ci(v['end_minus_norm'], True)} | {_ci(v['end_minus_hard'], True)} |")
            e = s["E3"]
            L += ["", "**E3 真实错误定价**（只修正这类错误时，分数增量 / 真实 F1 增量；1 = 按真实损失计价）", "",
                  "错误个数：" + json.dumps(e["counts"]), "", "| 评分 | " + " | ".join(ERRORS) + " |", "|---|" + "---:|" * len(ERRORS)]
            for n, v in e["ratio"].items():
                L.append(f"| {n} | " + " | ".join(f"{v[x][0]:.2f}" for x in ERRORS) + " |")
            L.append("")
        p = c["perturbation"]
        L += ["## 人工扰动 v2", "", f"共识用的场景：{p['scenario_for_consensus']}；含该类型的文档数：" + json.dumps(p["documents_with_type"]), "",
              "**检出率（只注入 1 个错误）**", "", "| 评分 | " + " | ".join(p["types"]) + " |", "|---|" + "---:|" * len(p["types"])]
        k1 = p["levels"].index("k=1")
        for i, n in enumerate(p["scorers"]):
            L.append(f"| {n} | " + " | ".join(f"{p['detect']['point'][i][t][k1]:.2f}" for t in range(len(p["types"]))) + " |")
        L += ["", "**斜率（理想 1）**", "", "| 评分 | " + " | ".join(p["types"]) + " |", "|---|" + "---:|" * len(p["types"])]
        for i, n in enumerate(p["scorers"]):
            L.append(f"| {n} | " + " | ".join(f"{p['slope']['point'][i][t]:.2f}" for t in range(len(p["types"]))) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def render_examples(ex: dict) -> str:
    L = ["# E4：最终评估器与 Gold 顺序相反的系统对（供人工挑选正文例子）", "",
         "每条：同一文档中 Gold 认为 A 比 B 好、评估器却给 B 更高分；列出 A、B 不同的空隙（⟂ 为空隙，⏎ 为段落换行）。", ""]
    for name, block in ex.items():
        L += [f"## {name}（场景：{block['scenario']}）", ""]
        for k, e in enumerate(block["pairs"], 1):
            L.append(f"**{k}. {e['doc']}**（{e['genre']}）：Gold 更好 {e['better_by_gold']} F1 {e['gold_f1'][0]} vs {e['worse_by_gold']} "
                     f"{e['gold_f1'][1]}；分数 {e['score'][0]} vs {e['score'][1]}；不同空隙 {e['differing_gaps']} 个")
            for g in e["gaps"]:
                L.append(f"- `…{g['left']}⟂{g['right']}…` — Gold {'切' if g['gold'] else '不切'}，p = {g['p']}，左侧 {g['left_punct']}"
                         f"{'，段落边界' if g['paragraph_break'] else ''}，{g['cut_by']} 切")
            L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["evidence", "analyze"])
    parser.add_argument("--corpora", nargs="*", default=list(NAMES))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    {"evidence": build_evidence, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
