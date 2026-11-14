"""Amendment 6: combine strong evidence with other signals (CPU).

  dev        GUM train, leave-one-genre-out: every evidence variant, entry-level scorers of both
             tracks, the pre-registered selection, dev success / fairness checks, robustness;
             then fits every component once on all of GUM train for the one-shot test.
  freeze     test corpus: all scores from the frozen components, written before any Gold exists
  analyze    test corpus: meta-evaluation after ``a6_test gold`` materialized Gold

Single-system track: S0 (frozen main method), S1 (selected evidence), S2 (QE ridge).
Multi-system track: M0 (leave-one-out consensus), M1/M2 (Dawid-Skene without / with the
evidence prior, leave-one-out posteriors), M3 (QE ridge with the multi-system signals).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from .a6_data import CORPORA, ROOT, SEED, SYSTEMS, doc_slices, gold_sets, jsonl, load_gap_array, logo_folds, \
    predictions, safe_lower, surfaces, vote_matrix, write_evidence, write_jsonl
from .a6_signals import KneserNeyTrigram, NGRAM_NAMES, RULE_NAMES, dawid_skene, fit_logistic_np, fit_ridge, \
    gold_sentences, hsmm_posterior, length_prior, ngram_features, predict_logistic_np, predict_ridge, rule_features
from .analyze_evidence_gaps import left_context_category
from .meta_eval import max_soft_f1
from .quality_agreement import Items, evaluate, load_items, statistics
from .scoring import soft_fbeta


RESULTS = ROOT / "results/a6"
FINAL = CORPORA["gum"].path("final_components.pkl")
E0_DEV = ROOT / "data/real_system/train_cv/boundary_probe/raw/probe+lr_lam0.001/logo.jsonl"
EVIDENCE_ORDER = ("E0", "E0+hsmm", "E1", "E1+hsmm", "E2", "E2+hsmm", "stack", "stack+hsmm")
ANALYSIS_EVIDENCE = ("E1sm", "E2b")
RAW_SIGNALS = {"E1": ("sat12l_raw", "prob"), "E1sm": ("sat12l-sm_raw", "prob"),
               "E2": ("ft-modernbert_logit", "logit"), "E2b": ("ft-bert-uncased_logit", "logit")}
STACK_LAMBDA, PLATT_LAMBDA, RIDGE_LAMBDA, TOLERANCE = 1e-3, 1e-6, 1.0, 0.01
S_TRACK, M_TRACK = ("S0", "S1", "S2"), ("M0", "M1", "M2", "M3")
M_REPORT = M_TRACK + ("M1s",)          # M1s: amendment 6a, analysis only (never selected)
QE_DOC = ("doc_max_soft", "doc_entropy", "doc_mass_per_word", "doc_mass_no_terminal", "doc_log_gaps",
          "doc_terminal_density")
QE_CAND = ("soft", "nsoft", "cand_per_word", "cand_no_terminal", "cut_cost_per_word", "miss_mass_per_word",
           "stability")


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ gap-level evidence
def raw_signal(corpus, docs, key: str) -> np.ndarray:
    name, kind = RAW_SIGNALS[key]
    values = load_gap_array(corpus, name, docs).astype(float)
    return logit(values) if kind == "prob" else values


def e0_evidence(path: Path, docs) -> np.ndarray:
    rows = {r["doc_id"]: r for r in jsonl(path)}
    out = []
    for d in docs:
        r = rows[d.doc_id]
        if r["text_sha256"] != d.text_sha256 or r["candidate_offsets"] != d.candidate_offsets:
            raise RuntimeError(f"E0 evidence does not match {d.doc_id}")
        out.append(r["calibrated_boundary_probability"])
    return np.concatenate(out)


def terminal_mask(docs) -> np.ndarray:
    return np.concatenate([[left_context_category(safe_lower(d.text), o) == "terminal" for o in d.candidate_offsets]
                           for d in docs])


def rule_matrix(docs) -> np.ndarray:
    return np.vstack([rule_features(d.text, d.candidate_offsets) for d in docs])


def ngram_matrix(model, docs) -> np.ndarray:
    return np.vstack([ngram_features(model, d.text, d.candidate_offsets) for d in docs])


def hsmm(p: np.ndarray, docs, prior_len: np.ndarray, base_rate: float) -> np.ndarray:
    return np.concatenate([hsmm_posterior(p[sl], prior_len, base_rate) for sl in doc_slices(docs)])


def stack_features(e0, e1_logit, e2_logit, rules, ngrams) -> np.ndarray:
    return np.column_stack([logit(e0), e1_logit, e2_logit, rules, ngrams])


def dev_evidence(corpus, docs, gold) -> tuple[dict, dict]:
    """Every evidence variant on GUM train, each document's value from folds that never saw its genre."""
    y = np.concatenate([[int(o in gold[d.doc_id]) for o in d.candidate_offsets] for d in docs]).astype(float)
    fold = logo_folds(docs)
    genre_of_doc = np.array([d.genre for d in docs])
    genres = sorted(set(genre_of_doc.tolist()))
    raw = {k: raw_signal(corpus, docs, k) for k in RAW_SIGNALS}
    ev = {"E0": e0_evidence(E0_DEV, docs)}
    for key, z in raw.items():
        out = np.zeros_like(y)
        for g in range(len(genres)):
            held = fold == g
            out[held] = predict_logistic_np(fit_logistic_np(z[~held, None], y[~held], PLATT_LAMBDA), z[held, None])
        ev[key] = out
    rules = rule_matrix(docs)
    ngrams = np.zeros((len(y), len(NGRAM_NAMES)))
    stack = np.zeros_like(y)
    hs = {k: np.zeros_like(y) for k in ("E0", "E1", "E2", "stack")}
    slices = doc_slices(docs)
    for g, genre in enumerate(genres):
        train_docs = [d for d in docs if d.genre != genre]
        held_docs = [d for d in docs if d.genre == genre]
        held = fold == g
        ngrams[held] = ngram_matrix(KneserNeyTrigram(gold_sentences(train_docs, gold)), held_docs)
    x = stack_features(ev["E0"], raw["E1"], raw["E2"], rules, ngrams)
    for g, genre in enumerate(genres):
        held = fold == g
        stack[held] = predict_logistic_np(fit_logistic_np(x[~held], y[~held], STACK_LAMBDA), x[held])
    ev["stack"] = stack
    for g, genre in enumerate(genres):
        train_docs = [d for d in docs if d.genre != genre]
        prior = length_prior(train_docs, gold)
        rate = float(y[fold != g].mean())
        for d, sl in zip(docs, slices):
            if d.genre != genre:
                continue
            for k in hs:
                hs[k][sl] = hsmm_posterior(ev[k][sl], prior, rate)
    for k, v in hs.items():
        ev[f"{k}+hsmm"] = v
    return ev, {"y": y, "rules": rules, "ngrams": ngrams}


def fit_final(corpus, docs, gold, ev: dict, extras: dict) -> dict:
    """Every component fitted once on all of GUM train (labels = GUM-train Gold)."""
    y = extras["y"]
    raw = {k: raw_signal(corpus, docs, k) for k in RAW_SIGNALS}
    final = {"platt": {k: fit_logistic_np(z[:, None], y, PLATT_LAMBDA) for k, z in raw.items()},
             "ngram": KneserNeyTrigram(gold_sentences(docs, gold)),
             "length_prior": length_prior(docs, gold), "base_rate": float(y.mean())}
    x = stack_features(ev["E0"], raw["E1"], raw["E2"], extras["rules"], extras["ngrams"])
    final["stack"] = fit_logistic_np(x, y, STACK_LAMBDA)
    return final


def test_evidence(corpus, docs, final: dict, e0_path: Path) -> dict:
    ev = {"E0": e0_evidence(e0_path, docs)}
    raw = {k: raw_signal(corpus, docs, k) for k in RAW_SIGNALS}
    for k, z in raw.items():
        ev[k] = predict_logistic_np(final["platt"][k], z[:, None])
    x = stack_features(ev["E0"], raw["E1"], raw["E2"], rule_matrix(docs), ngram_matrix(final["ngram"], docs))
    ev["stack"] = predict_logistic_np(final["stack"], x)
    for k in ("E0", "E1", "E2", "stack"):
        ev[f"{k}+hsmm"] = hsmm(ev[k], docs, final["length_prior"], final["base_rate"])
    return ev


# ------------------------------------------------------------------ entry-level scores (gold-free)
def full_boundaries(corpus, preds: dict) -> dict:
    """Every predicted boundary incl. non-gap positions (for consensus, as in amendment 5)."""
    full = {k: set(v) for k, v in preds.items()}
    for failure in jsonl(corpus.failures):
        full[(failure["doc_id"], failure["system"])].add(failure["offset"])
    return full


def boundary_f1(a: set, b: set) -> float:
    return 1.0 if not a and not b else 2 * len(a & b) / (len(a) + len(b))


def entry_keys(docs) -> list[tuple[str, str]]:
    return [(d.doc_id, s) for d in docs for s in SYSTEMS]


def soft_scores(docs, preds, p: np.ndarray | dict) -> tuple[np.ndarray, np.ndarray]:
    """(Soft F1, normalized Soft F1) per entry; ``p`` shared or {system: per-gap array}."""
    soft, norm = [], []
    for d, sl in zip(docs, doc_slices(docs)):
        for s in SYSTEMS:
            q = (p[s] if isinstance(p, dict) else p)[sl]
            c = [int(o in preds[(d.doc_id, s)]) for o in d.candidate_offsets]
            value = soft_fbeta(c, q, 1.0) if len(c) else 1.0
            soft.append(value)
            norm.append(value / max_soft_f1(tuple(q)) if len(c) else 1.0)
    return np.array(soft), np.array(norm)


def stability_scores(corpus, docs, preds) -> np.ndarray:
    rows = defaultdict(list)
    for r in jsonl(corpus.path("stability.jsonl")):
        rows[(r["doc_id"], r["system"])].append(r)
    out = []
    for d in docs:
        gaps = set(d.candidate_offsets)
        for s in SYSTEMS:
            values = []
            for r in rows[(d.doc_id, s)]:
                interior = gaps - set(r["edge_gaps"])
                values.append(boundary_f1(preds[(d.doc_id, s)] & interior, set(r["offsets"]) & interior))
            if len(values) != 2:
                raise RuntimeError(f"Stability rows missing for {d.doc_id}/{s}")
            out.append(float(np.mean(values)))
    return np.array(out)


def qe_features(docs, preds, p: np.ndarray, terminal: np.ndarray, soft, nsoft, stability) -> np.ndarray:
    rows, k = [], 0
    for d, sl in zip(docs, doc_slices(docs)):
        q, term = p[sl], terminal[sl]
        n_words = len(q) + 1
        qc = np.clip(q, 1e-6, 1 - 1e-6)
        doc = [max_soft_f1(tuple(q)) if len(q) else 1.0,
               float(np.mean(-(qc * np.log(qc) + (1 - qc) * np.log(1 - qc)))) if len(q) else 0.0,
               q.sum() / n_words, q[~term].sum() / max(q.sum(), 1e-9), math.log1p(len(q)),
               float(term.mean()) if len(q) else 0.0]
        for s in SYSTEMS:
            c = np.array([o in preds[(d.doc_id, s)] for o in d.candidate_offsets], dtype=bool)
            rows.append(doc + [soft[k], nsoft[k], c.sum() / n_words,
                               float((c & ~term).sum() / max(c.sum(), 1)),
                               (1 - q[c]).sum() / n_words, q[~c].sum() / n_words, stability[k]])
            k += 1
    return np.array(rows)


def consensus_scores(docs, full) -> np.ndarray:
    out = []
    for d in docs:
        for s in SYSTEMS:
            votes = defaultdict(int)
            for o in SYSTEMS:
                if o != s:
                    for b in full[(d.doc_id, o)]:
                        votes[b] += 1
            out.append(boundary_f1(full[(d.doc_id, s)], {b for b, v in votes.items() if v >= 2}))
    return np.array(out)


def ds_scores(docs, preds, terminal, prior, per_stratum_rate: bool = False) -> tuple[np.ndarray, dict]:
    votes = vote_matrix(docs, preds)
    fit = dawid_skene(votes, terminal.astype(int), prior, per_stratum_rate=per_stratum_rate)
    per_system = {s: fit["leave_one_out"][i] for i, s in enumerate(SYSTEMS)}
    soft, _ = soft_scores(docs, preds, per_system)
    info = {"base_rate": fit["base_rate"],
            "sensitivity": {s: fit["sensitivity"][i].tolist() for i, s in enumerate(SYSTEMS)},
            "specificity": {s: fit["specificity"][i].tolist() for i, s in enumerate(SYSTEMS)}}
    return soft, info


def all_scores(corpus, docs, ev: dict, selected: str, qe_models: dict | None = None) -> tuple[dict, dict, dict]:
    """Every entry-level scorer. ``qe_models`` = fitted ridge models (test); None on dev, where the
    QE scorers are cross-fitted by genre (caller passes gold via ``dev_qe``)."""
    preds = predictions(corpus)
    full = full_boundaries(corpus, preds)
    terminal = terminal_mask(docs)
    scores, info = {}, {}
    for name, p in ev.items():
        soft, norm = soft_scores(docs, preds, p)
        scores[f"ev:{name}"] = norm
        scores[f"ev_raw:{name}"] = soft
    scores["S0"] = scores["ev:E0"]
    scores["S1"] = scores[f"ev:{selected}"]
    scores["M0"] = consensus_scores(docs, full)
    scores["M1"], info["M1"] = ds_scores(docs, preds, terminal, None)
    scores["M1s"], info["M1s"] = ds_scores(docs, preds, terminal, None, per_stratum_rate=True)   # 6a, analysis only
    scores["M2"], info["M2"] = ds_scores(docs, preds, terminal, ev[selected])
    stability = stability_scores(corpus, docs, preds)
    soft, norm = scores[f"ev_raw:{selected}"], scores[f"ev:{selected}"]
    x_s = qe_features(docs, preds, ev[selected], terminal, soft, norm, stability)
    x_m = np.column_stack([x_s, scores["M2"], scores["M0"]])
    features = {"S2": x_s, "M3": x_m, "S2-no-stability": x_s[:, :-1]}
    if qe_models is not None:
        for name, x in features.items():
            scores[name] = predict_ridge(qe_models[name], x)
    scores["stability"] = stability
    return scores, features, info


def dev_qe(docs, features: dict, gold_f1: np.ndarray) -> dict:
    genre = np.repeat([d.genre for d in docs], len(SYSTEMS))
    out = {}
    for name, x in features.items():
        pred = np.zeros(len(gold_f1))
        for g in sorted(set(genre.tolist())):
            held = genre == g
            pred[held] = predict_ridge(fit_ridge(x[~held], gold_f1[~held], RIDGE_LAMBDA), x[held])
        out[name] = pred
    return out


# ------------------------------------------------------------------ meta-evaluation helpers
def gold_items(corpus, docs, e0_path: Path, candidates: Path) -> Items:
    items, _ = load_items(e0_path, data_root=corpus.data_root, split=corpus.split, candidates=candidates,
                          failures=corpus.failures)
    if list(zip(items.doc.tolist(), items.system.tolist())) != entry_keys(docs):
        raise RuntimeError("Item order differs from the amendment-6 entry order")
    return items


def with_scores(items: Items, scores: dict, names) -> Items:
    return Items(items.doc, items.domain, items.system, items.gold, {n: np.asarray(scores[n], float) for n in names})


def fairness(items: Items, score: np.ndarray) -> dict:
    pct_s, pct_g = rankdata(score) / len(score), rankdata(items.gold) / len(score)
    bias = {s: float(np.mean((pct_s - pct_g)[items.system == s])) for s in SYSTEMS}
    mean_s = {s: float(score[items.system == s].mean()) for s in SYSTEMS}
    mean_g = {s: float(items.gold[items.system == s].mean()) for s in SYSTEMS}
    return {"rank_bias": bias, "max_abs_bias": max(abs(v) for v in bias.values()),
            "top_by_score": max(mean_s, key=mean_s.get), "top_by_gold": max(mean_g, key=mean_g.get),
            "corpus_ranking": sorted(mean_s, key=mean_s.get, reverse=True)}


def decomposition(items: Items, score: np.ndarray) -> dict:
    docs = sorted(set(items.doc.tolist()))
    ms = np.array([score[items.doc == d].mean() for d in docs])
    mg = np.array([items.gold[items.doc == d].mean() for d in docs])
    rs = score - np.array([ms[docs.index(d)] for d in items.doc])
    rg = items.gold - np.array([mg[docs.index(d)] for d in items.doc])
    return {"between_doc": float(spearmanr(ms, mg).statistic), "within_doc": float(spearmanr(rs, rg).statistic)}


def evidence_quality(ev: dict, y: np.ndarray, terminal: np.ndarray) -> dict:
    out = {}
    for name, p in ev.items():
        d = p >= 0.5
        tp, fp, fn = (d & (y == 1)).sum(), (d & (y == 0)).sum(), (~d & (y == 1)).sum()
        out[name] = {"di_f1": float(2 * tp / (2 * tp + fp + fn)),
                     "mean_p_gold_no_terminal": float(p[(y == 1) & ~terminal].mean()),
                     "mean_p_nongold_terminal": float(p[(y == 0) & terminal].mean()) if ((y == 0) & terminal).any() else None}
    return out


def error_overlap(docs, preds, p: np.ndarray, y: np.ndarray) -> dict:
    wrong_e = (p >= 0.5) != (y == 1)
    votes = vote_matrix(docs, preds)
    out = {}
    for i, s in enumerate(SYSTEMS):
        wrong_s = (votes[i] == 1) != (y == 1)
        out[s] = float(wrong_e[wrong_s].mean()) if wrong_s.any() else float("nan")
    return out


def robustness(docs, preds, gold: dict) -> dict[str, np.ndarray]:
    """Alternative Gold quality per entry on whitespace gaps: -Pk, -WindowDiff, B-cubed F1."""
    out = {"neg_pk": [], "neg_windowdiff": [], "bcubed_f1": []}
    for d in docs:
        g = np.array([o in gold[d.doc_id] for o in d.candidate_offsets], dtype=int)
        n_words = len(g) + 1
        k = max(1, int(round(n_words / (g.sum() + 1) / 2)))
        for s in SYSTEMS:
            c = np.array([o in preds[(d.doc_id, s)] for o in d.candidate_offsets], dtype=int)
            out["neg_pk"].append(-_pk(g, c, k))
            out["neg_windowdiff"].append(-_windowdiff(g, c, k))
            out["bcubed_f1"].append(_bcubed(g, c))
    return {k: np.array(v) for k, v in out.items()}


def _pk(g, c, k):
    if len(g) < k:
        return 0.0
    cg, cc = np.concatenate([[0], np.cumsum(g)]), np.concatenate([[0], np.cumsum(c)])
    i = np.arange(len(g) + 1 - k)
    return float(np.mean(((cg[i + k] - cg[i]) > 0) != ((cc[i + k] - cc[i]) > 0)))


def _windowdiff(g, c, k):
    if len(g) < k:
        return 0.0
    cg, cc = np.concatenate([[0], np.cumsum(g)]), np.concatenate([[0], np.cumsum(c)])
    i = np.arange(len(g) + 1 - k)
    return float(np.mean((cg[i + k] - cg[i]) != (cc[i + k] - cc[i])))


def _bcubed(g, c):
    sg = np.concatenate([[0], np.cumsum(g)])          # sentence id of each word
    sc = np.concatenate([[0], np.cumsum(c)])
    size_g, size_c = np.bincount(sg), np.bincount(sc)
    pair = defaultdict(int)
    for a, b in zip(sg, sc):
        pair[(a, b)] += 1
    overlap = np.array([pair[(a, b)] for a, b in zip(sg, sc)], dtype=float)
    precision = float(np.mean(overlap / size_c[sc]))
    recall = float(np.mean(overlap / size_g[sg]))
    return 2 * precision * recall / (precision + recall)


def ci_positive(result: dict, name: str, metric: str = "global_spearman") -> bool:
    diff = result["paired_vs_baseline"].get(name)
    return bool(diff and diff[metric][0] > 0)


def pick(point: dict, names, eligible=None) -> str:
    """Highest global Spearman; the simplest (earliest in ``names``) within TOLERANCE wins."""
    pool = [n for n in names if eligible is None or n in eligible]
    if not pool:
        return names[0]
    best = max(point[n]["global_spearman"] for n in pool)
    return next(n for n in pool if point[n]["global_spearman"] >= best - TOLERANCE)


# ------------------------------------------------------------------ stages
def dev(args) -> None:
    corpus = CORPORA["gum"]
    docs = surfaces(corpus)
    gold = gold_sets(corpus)
    ev, extras = dev_evidence(corpus, docs, gold)
    terminal = terminal_mask(docs)
    items = gold_items(corpus, docs, E0_DEV, ROOT / "data/real_system/train_cv/frozen_scores_cv.jsonl")

    # 1. evidence selection (normalized Soft F1, paired vs E0)
    ev_names = [f"ev:{n}" for n in EVIDENCE_ORDER]
    preds = predictions(corpus)
    ev_scores = {f"ev:{n}": soft_scores(docs, preds, ev[n])[1] for n in EVIDENCE_ORDER + ANALYSIS_EVIDENCE}
    ev_result = evaluate(with_scores(items, ev_scores, list(ev_scores)), args.bootstrap, SEED, "ev:E0", args.workers)
    eligible = [n for n in ev_names[1:] if ci_positive(ev_result, n)]
    selected = (pick(ev_result["point"], ev_names, eligible) if eligible else "ev:E0").split(":", 1)[1]
    print(json.dumps({"status": "A6_EVIDENCE_SELECTED", "selected": selected, "eligible": eligible}), flush=True)

    # 2. entry-level scorers
    scores, features, info = all_scores(corpus, docs, ev, selected)
    scores.update(dev_qe(docs, features, items.gold))
    alt = robustness(docs, preds, gold)
    s_res = evaluate(with_scores(items, scores, S_TRACK + ("S2-no-stability",)), args.bootstrap, SEED, "S0", args.workers)
    m_res = evaluate(with_scores(items, scores, M_REPORT), args.bootstrap, SEED, "M0", args.workers)
    s_star = "S2" if s_res["point"]["S2"]["global_spearman"] >= s_res["point"]["S1"]["global_spearman"] + TOLERANCE else "S1"
    m_star = pick(m_res["point"], M_TRACK[1:])
    fair = {n: fairness(items, scores[n]) for n in S_TRACK + M_REPORT}
    checks = {
        "S_paired_ci_positive": ci_positive(s_res, s_star) if s_star != "S0" else False,
        "M_paired_ci_positive": ci_positive(m_res, m_star),
        "S_fair_bias": fair[s_star]["max_abs_bias"] <= fair["S0"]["max_abs_bias"] + 0.05,
        "M_fair_bias": fair[m_star]["max_abs_bias"] <= fair["M0"]["max_abs_bias"] + 0.05,
        "S_fair_top": fair[s_star]["top_by_score"] == fair[s_star]["top_by_gold"],
        "M_fair_top": fair[m_star]["top_by_score"] == fair[m_star]["top_by_gold"],
    }
    report = {
        "selected_evidence": selected, "eligible_evidence": [n.split(":", 1)[1] for n in eligible],
        "S_star": s_star, "M_star": m_star, "checks": checks, "dev_success": all(checks.values()),
        "evidence": ev_result, "S": s_res, "M": m_res, "fairness": fair,
        "decomposition": {n: decomposition(items, scores[n]) for n in S_TRACK + M_REPORT},
        "robustness": {n: {k: float(spearmanr(scores[n], v).statistic) for k, v in alt.items()}
                       for n in S_TRACK + M_REPORT},
        "evidence_quality": evidence_quality(ev, extras["y"], terminal),
        "error_overlap": {n: error_overlap(docs, preds, ev[n], extras["y"]) for n in ("E0", selected)},
        "dawid_skene": info, "system_gold_macro_f1": {s: float(items.gold[items.system == s].mean()) for s in SYSTEMS},
        "stability_spearman_vs_gold": float(spearmanr(scores["stability"], items.gold).statistic),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dev_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "dev_summary.md").write_text(render(report, "GUM train（LOGO，开发集）", args.bootstrap))

    # 3. components for the one-shot test, fitted once on all of GUM train
    final = fit_final(corpus, docs, gold, ev, extras)
    final["qe"] = {name: fit_ridge(x, items.gold, RIDGE_LAMBDA) for name, x in features.items()}
    final["selection"] = {"selected_evidence": selected, "S_star": s_star, "M_star": m_star}
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    with FINAL.open("wb") as handle:
        pickle.dump(final, handle)
    print(json.dumps({"status": "A6_DEV_COMPLETE", "selected": selected, "S_star": s_star, "M_star": m_star,
                      "dev_success": report["dev_success"], "checks": checks}), flush=True)


def freeze(args) -> None:
    corpus = CORPORA[args.corpus]
    if corpus.name == "gum":
        raise RuntimeError("freeze is for the test corpora")
    if (corpus.data_root / f"gold_{corpus.split}.jsonl").exists():
        raise RuntimeError("Gold already materialized; refusing to (re)freeze scores")
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    docs = surfaces(corpus)
    ev = test_evidence(corpus, docs, final, corpus.path("e0_probe+lr.jsonl"))
    for name, p in ev.items():
        write_evidence(corpus, name, docs, p)
    selected = final["selection"]["selected_evidence"]
    scores, _, info = all_scores(corpus, docs, ev, selected, final["qe"])
    preds = predictions(corpus)
    rows = []
    for k, (doc_id, system) in enumerate(entry_keys(docs)):
        rows.append({"doc_id": doc_id, "system": system, "candidate_boundary_offsets": sorted(preds[(doc_id, system)]),
                     "scores": {n: float(v[k]) for n, v in scores.items()}})
    write_jsonl(rows, corpus.path("frozen_scores.jsonl"))
    (corpus.path("frozen_info.json")).write_text(json.dumps({"selection": final["selection"], "dawid_skene": info},
                                                            indent=2) + "\n")
    print(json.dumps({"status": "A6_SCORES_FROZEN", "corpus": corpus.name, "rows": len(rows),
                      "gold_file_opened": False}), flush=True)


def analyze(args) -> None:
    corpora = [CORPORA[c] for c in args.corpora]
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    sel = final["selection"]
    out = {"selection": sel, "corpora": {}}
    pooled = []
    for corpus in corpora:
        docs = surfaces(corpus)
        gold = gold_sets(corpus)
        items = gold_items(corpus, docs, corpus.path("evidence", "E0.jsonl"), corpus.path("frozen_scores.jsonl"))
        frozen = jsonl(corpus.path("frozen_scores.jsonl"))
        scores = {n: np.array([r["scores"][n] for r in frozen]) for n in frozen[0]["scores"]}
        preds = predictions(corpus)
        y = np.concatenate([[int(o in gold[d.doc_id]) for o in d.candidate_offsets] for d in docs]).astype(float)
        ev = {n: np.concatenate([r["calibrated_boundary_probability"] for r in jsonl(corpus.path("evidence", f"{n}.jsonl"))])
              for n in EVIDENCE_ORDER + ANALYSIS_EVIDENCE}
        out["corpora"][corpus.name] = corpus_report(corpus, docs, gold, items, scores, preds, y, ev, sel, args)
        pooled.append((items, scores))
    if len(pooled) > 1:
        items = Items(*(np.concatenate([getattr(i, f) for i, _ in pooled]) for f in ("doc", "domain", "system", "gold")))
        names = list(S_TRACK + M_TRACK)
        scores = {n: np.concatenate([s[n] for _, s in pooled]) for n in names}
        out["pooled"] = {"S": evaluate(with_scores(items, scores, S_TRACK), args.bootstrap, SEED, "S0", args.workers),
                         "M": evaluate(with_scores(items, scores, M_TRACK), args.bootstrap, SEED, "M0", args.workers)}
    (RESULTS / "test_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "test_summary.md").write_text(render_test(out, args.bootstrap))
    print(json.dumps({"status": "A6_TEST_ANALYZED"}), flush=True)


def corpus_report(corpus, docs, gold, items, scores, preds, y, ev, sel, args) -> dict:
    terminal = terminal_mask(docs)
    s_res = evaluate(with_scores(items, scores, S_TRACK + ("S2-no-stability",)), args.bootstrap, SEED, "S0", args.workers)
    m_res = evaluate(with_scores(items, scores, M_REPORT), args.bootstrap, SEED, "M0", args.workers)
    ev_names = [f"ev:{n}" for n in EVIDENCE_ORDER + ANALYSIS_EVIDENCE]
    ev_res = evaluate(with_scores(items, scores, ev_names), args.bootstrap, SEED, "ev:E0", args.workers)
    alt = robustness(docs, preds, gold)
    fair = {n: fairness(items, scores[n]) for n in S_TRACK + M_REPORT}
    s_star, m_star = sel["S_star"], sel["M_star"]
    return {
        "items": int(len(items.gold)), "documents": len(docs),
        "test_success": {"S": ci_positive(s_res, s_star) if s_star != "S0" else False, "M": ci_positive(m_res, m_star)},
        "S": s_res, "M": m_res, "evidence": ev_res, "fairness": fair,
        "decomposition": {n: decomposition(items, scores[n]) for n in S_TRACK + M_REPORT},
        "robustness": {n: {k: float(spearmanr(scores[n], v).statistic) for k, v in alt.items()} for n in S_TRACK + M_REPORT},
        "evidence_quality": evidence_quality(ev, y, terminal),
        "error_overlap": {n: error_overlap(docs, preds, ev[n], y) for n in ("E0", sel["selected_evidence"])},
        "system_gold_macro_f1": {s: float(items.gold[items.system == s].mean()) for s in SYSTEMS},
    }


# ------------------------------------------------------------------ rendering
LABELS = {"S0": "S0 冻结主方法（E0 归一化 Soft F1）", "S1": "S1 选中证据（归一化 Soft F1）", "S2": "S2 QE 岭回归（单系统）",
          "S2-no-stability": "S2 去掉稳定性（消融）", "M0": "M0 留一共识", "M1": "M1 Dawid–Skene",
          "M2": "M2 Dawid–Skene + 证据先验", "M1s": "M1s 分层基率（6a，只作分析）", "M3": "M3 QE 岭回归（多系统）"}
METRICS = (("global_spearman", "全局 Spearman"), ("large_gap_pairwise", "大差距配对"),
           ("cell_spearman", "领域×系统 Spearman"), ("within_doc_pairwise", "文档内配对"))


def _table(result: dict, names, labels=None) -> list[str]:
    base = result["baseline"]
    lines = ["| 评分 | " + " | ".join(l for _, l in METRICS) + f" | 全局 Spearman 差 vs {base} [95% CI] |",
             "|---|" + "---:|" * (len(METRICS) + 1)]
    for n in names:
        v = result["point"][n]
        diff = result["paired_vs_baseline"].get(n)
        cell = "—" if diff is None else (f"{v['global_spearman'] - result['point'][base]['global_spearman']:+.3f} "
                                         f"[{diff['global_spearman'][0]:+.3f}, {diff['global_spearman'][1]:+.3f}]")
        ci = result["ci"][n]["global_spearman"]
        lines.append(f"| {(labels or {}).get(n, n)} | {v['global_spearman']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                     + " | ".join(f"{v[k]:.3f}" for k, _ in METRICS[1:]) + f" | {cell} |")
    return lines


def _common(report: dict) -> list[str]:
    lines = ["", "## 公平性（平均秩残差 = 分数百分位秩 − Gold 百分位秩；语料级排名）", "",
             "| 评分 | " + " | ".join(SYSTEMS) + " | 最大绝对值 | 语料级排名 | Gold 第一 |", "|---|" + "---:|" * (len(SYSTEMS) + 1) + "---|---|"]
    for n, f in report["fairness"].items():
        lines.append(f"| {n} | " + " | ".join(f"{f['rank_bias'][s]:+.3f}" for s in SYSTEMS)
                     + f" | {f['max_abs_bias']:.3f} | {' > '.join(f['corpus_ranking'])} | {f['top_by_gold']} |")
    lines += ["", "## 分解与稳健性（Spearman）", "",
              "| 评分 | 文档之间 | 文档之内 | vs −Pk | vs −WindowDiff | vs B-cubed F1 |", "|---|---:|---:|---:|---:|---:|"]
    for n in report["decomposition"]:
        d, r = report["decomposition"][n], report["robustness"][n]
        lines.append(f"| {n} | {d['between_doc']:.3f} | {d['within_doc']:.3f} | {r['neg_pk']:.3f} | "
                     f"{r['neg_windowdiff']:.3f} | {r['bcubed_f1']:.3f} |")
    lines += ["", "## 证据质量（p ≥ 0.5 当作切分器）", "", "| 证据 | 切分 F1 | 无终止标点 Gold 平均 p | 带终止标点非 Gold 平均 p |",
              "|---|---:|---:|---:|"]
    for n, q in report["evidence_quality"].items():
        nt = q["mean_p_nongold_terminal"]
        lines.append(f"| {n} | {q['di_f1']:.3f} | {q['mean_p_gold_no_terminal']:.3f} | {'—' if nt is None else f'{nt:.3f}'} |")
    lines += ["", "证据错误与系统错误的重叠率（系统出错的空隙中证据也出错的比例）：", ""]
    for n, o in report["error_overlap"].items():
        lines.append(f"- {n}：" + "，".join(f"{s} {v:.3f}" for s, v in o.items()))
    lines += ["", "系统 Gold Macro F1：" + "，".join(f"{s} {v:.3f}" for s, v in report["system_gold_macro_f1"].items())]
    return lines


def render(report: dict, title: str, bootstrap: int) -> str:
    lines = [f"# 修订 6：强证据 + 其他信号（{title}）", "",
             f"选中证据：**{report['selected_evidence']}**（CI 全在 0 以上的候选：{', '.join(report['eligible_evidence']) or '无'}）；"
             f"S* = **{report['S_star']}**，M* = **{report['M_star']}**；开发成功：**{report['dev_success']}**。",
             f"CI：领域内分层按文档重采样 {bootstrap} 次。", "", "判据：" + "，".join(f"{k}={v}" for k, v in report["checks"].items()),
             "", "## 证据（归一化 Soft F1）", ""]
    lines += _table(report["evidence"], list(report["evidence"]["point"]))
    lines += ["", "## 单系统线", ""] + _table(report["S"], list(report["S"]["point"]), LABELS)
    lines += ["", "## 多系统线", ""] + _table(report["M"], list(report["M"]["point"]), LABELS)
    lines += _common(report)
    lines += ["", f"扰动稳定性与 Gold F1 的 Spearman：{report['stability_spearman_vs_gold']:.3f}"]
    return "\n".join(lines) + "\n"


def render_test(out: dict, bootstrap: int) -> str:
    sel = out["selection"]
    lines = ["# 修订 6：一次性测试（GENTLE 主、LinES 次）", "",
             f"冻结选择：证据 **{sel['selected_evidence']}**，S* = **{sel['S_star']}**，M* = **{sel['M_star']}**。"
             f"所有组件只在 GUM train 上拟合；分数先冻结、后物化 Gold。CI：领域内分层按文档重采样 {bootstrap} 次。"]
    for name, rep in out["corpora"].items():
        lines += ["", f"# {name}（{rep['documents']} 篇，{rep['items']} 个条目）", "",
                  f"测试成功：S = **{rep['test_success']['S']}**，M = **{rep['test_success']['M']}**", "", "## 单系统线", ""]
        lines += _table(rep["S"], list(rep["S"]["point"]), LABELS)
        lines += ["", "## 多系统线", ""] + _table(rep["M"], list(rep["M"]["point"]), LABELS)
        lines += ["", "## 证据（归一化 Soft F1）", ""] + _table(rep["evidence"], list(rep["evidence"]["point"]))
        lines += _common(rep)
    if "pooled" in out:
        lines += ["", "# 合并（GENTLE + LinES）", "", "## 单系统线", ""] + _table(out["pooled"]["S"], S_TRACK, LABELS)
        lines += ["", "## 多系统线", ""] + _table(out["pooled"]["M"], M_TRACK, LABELS)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["dev", "freeze", "analyze"])
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="gum")
    parser.add_argument("--corpora", nargs="*", default=["gentle", "lines"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    {"dev": dev, "freeze": freeze, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
