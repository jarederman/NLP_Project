"""Amendment 10: quality-oriented evidence fusion vs post-hoc quality regression, controlled.

Within one setting every method shares the per-gap design matrix x (the stack features of the cell),
the 15 GUM LOGO folds, the items (document x 4 systems), the normalized Soft F1 and the evaluation;
only the training objective or the second stage changes.

  S1      L2 logistic stack (BCE), score = Q_norm (the current evaluator)
  S2      amendment-6 ridge QE (13 features incl. perturbation stability); S2-ns without stability
  A       ridge on h = [Q_soft, Q_norm, density, missed mass, mean|y-p|, document features] -> F1
  R       Q_norm + ridge(h) fitted to F1 - Q_norm
  B-*     stack weights trained on L_quality = mean (Q_norm(y, p_theta) - F1)^2, alone or with BCE
          and / or a global pairwise ranking loss; each term divided by its value at the BCE warm start

Settings: T1 = amendment-6 cell (lower, flat; systems flat), T2 = 9.1 ref91 (E1 cased, flat; systems
flat), T3 = 9.2+9.3 lower-nl (systems see newlines). GUM = LOGO (decision); GENTLE / LinES / EWT = R.

  python -m src.a10_quality --setting T3 [--bootstrap 1000]
writes results/a10/report_T*.json and summary_T*.md.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import warnings
from dataclasses import dataclass

import numpy as np

from .a6_combine import FINAL, RIDGE_LAMBDA, STACK_LAMBDA, decomposition, dev_evidence, e0_evidence, fairness, \
    ngram_matrix, rule_matrix, stability_scores, terminal_mask
from .a6_data import CORPORA, ROOT, SYSTEMS, doc_slices, gold_sets, jsonl, logo_folds, predictions, surfaces
from .a6_signals import fit_logistic_np, fit_ridge, predict_logistic_np, predict_ridge
from .a8_supplement import TYPES, DocCtx, candidates, category_codes, hard_f1, ols_slope, soft_f1
from .a9_casing import A9, SLOPE_TYPES, corpus_items
from .a9_layout import EWT_E0, NL, bootstrap, break_mask, contrast, design, e1_logit, e2_logit, scenario_items
from .meta_eval import max_soft_f1

RESULTS = ROOT / "results/a10"
SETTINGS = {"T1": {"e1": "lower", "e2": "lower", "layout": "flat", "label": "修订 6 基线（lower-flat，系统无换行）"},
            "T2": {"e1": "cased", "e2": "lower", "layout": "flat", "label": "9.1 ref91（无换行输入的现行评估器）"},
            "T3": {"e1": "lower", "e2": "lower", "layout": "nl", "label": "9.2+9.3 lower-nl（系统看到换行，带版式输入的现行评估器）"}}
B_VARIANTS = {"B-q": (1.0, 0.0, 0.0), "B-qb": (1.0, 1.0, 0.0), "B-qr": (1.0, 0.0, 1.0), "B-qbr": (1.0, 1.0, 1.0)}
REGRESSORS = ("S2", "S2-ns", "A", "R")
SIMPLICITY = ("R", "A", "B-q", "B-qb", "B-qr", "B-qbr")
TAU = 0.05
MAX_ITER = 200
TOL = 0.01
SLOPE_GAP = 0.05
TEST = ("gentle", "lines", "ewt")
TIE_DECIMALS = 10
CURRENT: dict = {}                  # setting of this run (file names)
ALIGNMENT_ITEMS: dict = {}          # (corpus, layout) -> items whose Gold F1 includes unaligned boundaries


# ------------------------------------------------------------------ item features
def doc_features(q: np.ndarray, term: np.ndarray) -> list[float]:
    """The six amendment-6 document features (``a6_combine.qe_features``)."""
    if not len(q):
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    n_words = len(q) + 1
    qc = np.clip(q, 1e-6, 1 - 1e-6)
    return [max_soft_f1(tuple(q)), float(np.mean(-(qc * np.log(qc) + (1 - qc) * np.log(1 - qc)))),
            q.sum() / n_words, q[~term].sum() / max(q.sum(), 1e-9), math.log1p(len(q)), float(term.mean())]


def item_features(q: np.ndarray, c: np.ndarray, term: np.ndarray, kind: str, stability=None) -> np.ndarray:
    """(m, f) features of the m candidates ``c`` (m, n) given evidence q (n,). kind: S2, S2-ns or h."""
    c = np.atleast_2d(c).astype(bool)
    m, n = c.shape
    doc = doc_features(q, term)
    if n:
        soft = soft_f1(c, q)
        nsoft = soft / doc[0]
    else:
        soft = nsoft = np.ones(m)
    n_words = n + 1
    ncut = c.sum(1)
    miss = np.array([q[~row].sum() for row in c]) / n_words
    if kind == "h":
        dis = np.abs(c - q[None, :]).mean(1) if n else np.zeros(m)
        return np.column_stack([soft, nsoft, ncut / n_words, miss, dis,
                                np.tile([doc[0], doc[1], doc[2], doc[4]], (m, 1))])
    cut_cost = np.array([(1 - q[row]).sum() for row in c]) / n_words
    no_term = (c & ~term[None, :]).sum(1) / np.maximum(ncut, 1)
    x = np.column_stack([np.tile(doc, (m, 1)), soft, nsoft, ncut / n_words, no_term, cut_cost, miss])
    if kind == "S2":
        x = np.column_stack([x, np.broadcast_to(np.asarray(stability, float).ravel(), (m,))])
    return x


# ------------------------------------------------------------------ data
@dataclass
class Data:
    name: str
    docs: list
    x: np.ndarray               # (n_gaps, k) design matrix
    term: np.ndarray            # (n_gaps,) terminal mask
    cmat: list                  # per doc (4, n_d) bool system boundaries on gaps
    f1: np.ndarray              # (D * 4,) Gold F1 per item (= items.gold)
    items: object
    genres: np.ndarray          # (D,)
    stability: np.ndarray | None
    y: np.ndarray | None = None
    fold: np.ndarray | None = None


def load(name: str, setting: dict, ctx: dict) -> Data:
    layout = setting["layout"]
    docs = surfaces(A9[name])
    gold = gold_sets(A9[name])
    if name == "gum":
        e0, rules, ngrams = ctx["ev_gum"]["E0"], ctx["extras"]["rules"], ctx["extras"]["ngrams"]
    else:
        e0 = (e0_evidence(EWT_E0, docs) if name == "ewt" else
              np.concatenate([q["calibrated_boundary_probability"] for q in jsonl(CORPORA[name].path("evidence", "E0.jsonl"))]))
        rules, ngrams = rule_matrix(docs), ngram_matrix(ctx["final"]["ngram"], docs)
    x = design(e0, e1_logit(name, setting["e1"], layout, docs), e2_logit(name, setting["e2"], layout, docs), rules, ngrams,
               break_mask(name, docs) if layout == "nl" else None)
    preds = predictions(A9[name] if layout == "flat" else NL[name])
    items = corpus_items(name, docs) if layout == "flat" else scenario_items(name, "nl", surfaces(NL[name]))
    cmat = [np.array([[o in preds[(d.doc_id, s)] for o in d.candidate_offsets] for s in SYSTEMS], dtype=bool) for d in docs]
    f1_gaps = np.concatenate([hard_f1(c, np.array([o in gold[d.doc_id] for o in d.candidate_offsets])) if len(d.candidate_offsets)
                              else np.ones(len(SYSTEMS)) for d, c in zip(docs, cmat)])
    # the evaluation items also count unaligned (non-gap) system boundaries and non-gap Gold boundaries
    # (e.g. SpaceAfter=No in EWT), so they can differ from the gap-only F1; the items' Gold F1 (the
    # evaluation target of every amendment) is the target everywhere; the count is reported
    ALIGNMENT_ITEMS[(name, layout)] = int(np.sum(np.abs(f1_gaps - items.gold) > 1e-9))
    stab = stability_scores(CORPORA[name], docs, preds) if layout == "flat" and name in CORPORA else None
    out = Data(name, docs, x, terminal_mask(docs), cmat, items.gold.astype(float), items,
               np.array([d.genre for d in docs]), stab)
    if name == "gum":
        out.y, out.fold = ctx["extras"]["y"], logo_folds(docs)
    return out


# ------------------------------------------------------------------ scores from evidence
def qnorm_items(data: Data, p: np.ndarray, doc_idx=None, systems=None) -> np.ndarray:
    """Normalized Soft F1 of the selected documents' items (document-major, systems in order)."""
    sl = doc_slices(data.docs)
    out = []
    for i in (range(len(data.docs)) if doc_idx is None else doc_idx):
        q = p[sl[i]]
        rows = data.cmat[i] if systems is None else data.cmat[i][systems]
        if not len(q):
            out.extend([1.0] * len(rows))
            continue
        out.extend(soft_f1(rows, q) / max_soft_f1(tuple(q)))
    return np.array(out)


def item_matrix(data: Data, p: np.ndarray, kind: str, doc_idx, systems=None) -> np.ndarray:
    sl = doc_slices(data.docs)
    sys_idx = list(range(len(SYSTEMS))) if systems is None else list(systems)
    rows = []
    for i in doc_idx:
        st = None if data.stability is None else data.stability[item_rows([i], sys_idx)]
        rows.append(item_features(p[sl[i]], data.cmat[i][sys_idx], data.term[sl[i]], kind, st))
    return np.vstack(rows)


def item_rows(doc_idx, systems=None) -> np.ndarray:
    sys_idx = list(range(len(SYSTEMS))) if systems is None else list(systems)
    return np.array([i * len(SYSTEMS) + s for i in doc_idx for s in sys_idx], dtype=int)


def regress(kind: str, x_tr, f_tr, q_tr, x_te, q_te) -> np.ndarray:
    if kind == "R":
        return q_te + predict_ridge(fit_ridge(x_tr, f_tr - q_tr, RIDGE_LAMBDA), x_te)
    return predict_ridge(fit_ridge(x_tr, f_tr, RIDGE_LAMBDA), x_te)


FEATURE_KIND = {"S2": "S2", "S2-ns": "S2-ns", "A": "h", "R": "h"}


# ------------------------------------------------------------------ quality-oriented fusion (torch)
class QualityProblem:
    """Normalized Soft F1 of every training item as a differentiable function of the stack weights."""

    def __init__(self, torch, z: np.ndarray, y: np.ndarray, data: Data, doc_idx, systems=None):
        self.torch = torch
        t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt)
        sl = doc_slices(data.docs)
        sys_idx = list(range(len(SYSTEMS))) if systems is None else list(systems)
        lengths = [sl[i].stop - sl[i].start for i in doc_idx]
        if min(lengths) == 0:
            raise ValueError("QualityProblem needs documents with at least one gap")
        L, D, M = max(lengths), len(doc_idx), len(sys_idx)
        # z / y are the training gaps of the documents doc_idx, concatenated in that order
        starts = np.cumsum([0] + lengths[:-1])
        pad = np.full((D, L), len(y), dtype=np.int64)
        cpad = np.zeros((D, M, L))
        for k, (i, n, s0) in enumerate(zip(doc_idx, lengths, starts)):
            pad[k, :n] = np.arange(s0, s0 + n)
            cpad[k, :, :n] = data.cmat[i][sys_idx]
        self.z = t(np.column_stack([z, np.ones(len(z))]))
        self.y = t(y)
        self.pad = torch.as_tensor(pad)
        self.c = t(cpad)
        self.ny = self.c.sum(2)
        self.k = t(np.arange(1, L + 1))
        f = data.f1[item_rows(doc_idx, sys_idx)].reshape(D, M)
        self.f1 = t(f)
        ff = f.ravel()
        i, j = np.nonzero(ff[:, None] > ff[None, :] + 1e-12)
        self.pi, self.pj = torch.as_tensor(i), torch.as_tensor(j)

    def logits(self, theta):
        return self.z @ theta

    def losses(self, theta):
        torch = self.torch
        a = self.logits(theta)
        p = torch.sigmoid(a)
        bce = (torch.nn.functional.softplus(a) - self.y * a).mean()
        P = torch.cat([p, p.new_zeros(1)])[self.pad]                      # (D, L), zero padding
        sp = P.sum(1)
        soft = 2 * torch.einsum("dml,dl->dm", self.c, P) / (self.ny + sp[:, None])
        srt = torch.sort(P, dim=1, descending=True).values
        best = (2 * torch.cumsum(srt, 1) / (self.k[None, :] + sp[:, None])).max(1).values
        q = soft / best[:, None]
        lq = ((q - self.f1) ** 2).mean()
        qf = q.reshape(-1)
        lr = torch.nn.functional.softplus(-(qf[self.pi] - qf[self.pj]) / TAU).mean() if len(self.pi) else q.new_zeros(())
        return lq, bce, lr, q


def fit_quality(torch, problem: QualityProblem, theta0: np.ndarray, weights, lam: float = STACK_LAMBDA) -> np.ndarray:
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        lq0, lb0, lr0, _ = problem.losses(theta)
    cq, cb, cr = weights
    scale = [float(lq0), float(lb0), max(float(lr0), 1e-12)]

    def objective():
        lq, lb, lr, _ = problem.losses(theta)
        j = cq * lq / scale[0] + cb * lb / scale[1] + 0.5 * lam * (theta[:-1] ** 2).sum() / scale[1]
        if cr:
            j = j + cr * lr / scale[2]
        return j

    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=MAX_ITER, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-14, history_size=20)

    def closure():
        opt.zero_grad()
        j = objective()
        j.backward()
        return j

    opt.step(closure)
    return theta.detach().numpy().copy()


def standardize(model: dict, x: np.ndarray) -> np.ndarray:
    return (x - model["mu"]) / model["sd"]


def prob(model: dict, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = standardize(model, x) @ theta[:-1] + theta[-1]
    return 1 / (1 + np.exp(-a))


# ------------------------------------------------------------------ perturbation slopes
def slopes(data: Data, scorers: dict) -> dict:
    """Amendment-8.2 within-document slopes; scorers: name -> f(i, q_unused, C) -> scores of rows of C."""
    gold = gold_sets(A9[data.name])
    out = {n: np.full((len(data.docs), len(SLOPE_TYPES)), np.nan) for n in scorers}
    for i, d in enumerate(data.docs):
        g = np.array([o in gold[d.doc_id] for o in d.candidate_offsets], dtype=bool)
        cats = category_codes(d.text, d.candidate_offsets)
        ctx = DocCtx(d.doc_id, d.genre, g, cats, {}, {}, np.zeros((4, len(g)), bool), np.zeros((4, len(g)), bool),
                     np.zeros((4, len(g))), np.ones(2))
        c, t_idx, _ = candidates(ctx, i)
        true = hard_f1(c, g)
        for n, f in scorers.items():
            s = f(i, c)
            for k, t in enumerate(SLOPE_TYPES):
                rows = np.flatnonzero(t_idx == TYPES.index(t))
                if len(rows):
                    idx = np.concatenate([[0], rows])
                    out[n][i, k] = ols_slope(true[idx], s[idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {n: {t: float(np.nanmean(v[:, j])) for j, t in enumerate(SLOPE_TYPES)} for n, v in out.items()}


# ------------------------------------------------------------------ boundary metrics
def auc(p: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(p)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ------------------------------------------------------------------ evaluation
def tie_exact(v: np.ndarray) -> np.ndarray:
    """Round to TIE_DECIMALS so that mathematically equal scores (e.g. two systems with identical
    boundaries in one document) tie exactly instead of being ordered by floating-point noise."""
    return np.round(np.asarray(v, float), TIE_DECIMALS)


def evaluate(data: Data, scores: dict, n_boot: int, s2_name: str) -> dict:
    scores = {n: tie_exact(v) for n, v in scores.items()}
    np.savez_compressed(RESULTS / f"scores_{CURRENT['setting']}_{data.name}.npz", f1=data.f1,
                        doc=np.asarray(data.items.doc).astype(str), system=np.asarray(data.items.system).astype(str),
                        **{n.replace(":", "__"): v for n, v in scores.items()})
    point, draws, names = bootstrap(scores, data.f1, data.genres, n_boot)
    table = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, n in enumerate(names):
            v = scores[n]
            table[n] = {"global_spearman": [float(point[n][0]), float(np.percentile(draws[:, i, 0], 2.5)),
                                            float(np.percentile(draws[:, i, 0], 97.5))],
                        "within_doc_pairwise": float(point[n][1]), **decomposition(data.items, v),
                        "rank_bias": fairness(data.items, v)["rank_bias"]}
            if not n.startswith(("S1", "B-", "LOSO:B-")):
                table[n]["mae"] = float(np.mean(np.abs(v - data.f1)))
    vs = {}
    for n in names:
        if n == "S1":
            continue
        vs[n] = {"minus_S1": contrast(draws, names, point, [n], ["S1"])}
        for ref in ("S2", "S2-ns"):
            if ref in names and n != ref:
                vs[n][f"minus_{ref}"] = contrast(draws, names, point, [n], [ref])
    return {"table": table, "contrasts": vs, "s2_reference": s2_name}


def decide(gum: dict, s2_name: str) -> dict:
    """Pre-registered amendment-10 rule (GUM LOGO)."""
    c, t, sl = gum["eval"]["contrasts"], gum["eval"]["table"], gum["slopes"]
    mean_slope = {n: float(np.mean(list(v.values()))) for n, v in sl.items()}
    rows = {}
    for n in SIMPLICITY:
        k = c[n]
        loso = c.get(f"LOSO:{n}", {}).get("minus_S1", [float("nan")] * 3)
        ok = {"vs_S1": k["minus_S1"][1] > 0, f"vs_{s2_name}": k[f"minus_{s2_name}"][1] > 0, "loso": loso[0] > 0,
              "slopes": mean_slope[n] >= mean_slope["S1"] - SLOPE_GAP
              and sl[n]["merge_other"] >= sl["S1"]["merge_other"] - SLOPE_GAP}
        rows[n] = {"checks": ok, "eligible": all(ok.values()), "global_spearman": t[n]["global_spearman"][0]}
    eligible = [n for n in SIMPLICITY if rows[n]["eligible"]]
    if not eligible:
        return {"rows": rows, "adopted": "S1 (unchanged)"}
    best = max(rows[n]["global_spearman"] for n in eligible)
    pick = next(n for n in SIMPLICITY if n in eligible and rows[n]["global_spearman"] >= best - TOL)
    return {"rows": rows, "adopted": pick}


# ------------------------------------------------------------------ one LOGO fold (worker process)
_W: dict = {}


def _init_worker(gum: Data, regs, threads: int) -> None:
    import torch
    torch.set_num_threads(threads)
    _W.update(gum=gum, regs=list(regs), torch=torch)


def fold_job(g: int) -> dict:
    """Everything that is trained without genre g and scores genre g."""
    t0 = time.time()
    gum, regs, torch = _W["gum"], _W["regs"], _W["torch"]
    genres = sorted(set(gum.genres.tolist()))
    doc_fold = np.array([genres.index(x) for x in gum.genres])
    sl = doc_slices(gum.docs)
    held_gap, held_doc = gum.fold == g, np.flatnonzero(doc_fold == g)
    tr_doc = np.flatnonzero(doc_fold != g)
    outer = fit_logistic_np(gum.x[~held_gap], gum.y[~held_gap], STACK_LAMBDA)
    p_out = np.zeros(len(gum.y))
    p_out[held_gap] = predict_logistic_np(outer, gum.x[held_gap])
    # nested evidence for the training items of the regressors (inner LOGO over the other genres)
    p_in = np.zeros(len(gum.y))
    for h in range(len(genres)):
        if h == g:
            continue
        tr = (gum.fold != g) & (gum.fold != h)
        p_in[gum.fold == h] = predict_logistic_np(fit_logistic_np(gum.x[tr], gum.y[tr], STACK_LAMBDA), gum.x[gum.fold == h])
    held_rows = item_rows(held_doc)
    out = {"S1": (held_rows, qnorm_items(gum, p_out, held_doc))}
    q_tr, q_te = qnorm_items(gum, p_in, tr_doc), out["S1"][1]
    fmods = {}
    for r in regs:
        kind = FEATURE_KIND[r]
        x_tr, x_te = item_matrix(gum, p_in, kind, tr_doc), item_matrix(gum, p_out, kind, held_doc)
        f_tr = gum.f1[item_rows(tr_doc)]
        out[r] = (held_rows, regress(r, x_tr, f_tr, q_tr, x_te, q_te))
        fmods[r] = fit_ridge(x_tr, f_tr - q_tr if r == "R" else f_tr, RIDGE_LAMBDA)
        rows_all, vals_all = [], []
        for s in range(len(SYSTEMS)):
            keep = [k for k in range(len(SYSTEMS)) if k != s]
            rows_tr = np.flatnonzero(np.isin(item_rows(tr_doc), item_rows(tr_doc, keep)))
            rows_te = np.flatnonzero(np.isin(held_rows, item_rows(held_doc, [s])))
            rows_all.append(held_rows[rows_te])
            vals_all.append(regress(r, x_tr[rows_tr], f_tr[rows_tr], q_tr[rows_tr], x_te[rows_te], q_te[rows_te]))
        out[f"LOSO:{r}"] = (np.concatenate(rows_all), np.concatenate(vals_all))
    # quality-oriented fusion
    tr_gap = np.concatenate([np.arange(sl[i].start, sl[i].stop) for i in tr_doc])
    z_tr = standardize(outer, gum.x[tr_gap])
    theta0 = np.append(outer["w"], outer["b"])
    prob_main = QualityProblem(torch, z_tr, gum.y[tr_gap], gum, tr_doc)
    thetas, p_b = {}, {}
    for b, w in B_VARIANTS.items():
        th = fit_quality(torch, prob_main, theta0, w)
        thetas[b] = th
        pb = np.zeros(len(gum.y))
        pb[held_gap] = prob(outer, th, gum.x[held_gap])
        p_b[b] = pb[held_gap]
        out[b] = (held_rows, qnorm_items(gum, pb, held_doc))
        rows_all, vals_all = [], []
        for s in range(len(SYSTEMS)):
            keep = [k for k in range(len(SYSTEMS)) if k != s]
            th_s = fit_quality(torch, QualityProblem(torch, z_tr, gum.y[tr_gap], gum, tr_doc, keep), theta0, w)
            ps = np.zeros(len(gum.y))
            ps[held_gap] = prob(outer, th_s, gum.x[held_gap])
            rows_all.append(item_rows(held_doc, [s]))
            vals_all.append(qnorm_items(gum, ps, held_doc, [s]))
        out[f"LOSO:{b}"] = (np.concatenate(rows_all), np.concatenate(vals_all))
    return {"g": g, "p_out": p_out[held_gap], "p_b": p_b, "scores": out,
            "models": {"outer": outer, "theta": thetas, "ridge": fmods}, "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ main
def run(args) -> None:
    import torch
    torch.set_num_threads(args.threads)
    setting = SETTINGS[args.setting]
    CURRENT["setting"] = args.setting
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    gum_docs = surfaces(CORPORA["gum"])
    ev_gum, extras = dev_evidence(CORPORA["gum"], gum_docs, gold_sets(CORPORA["gum"]))
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    ctx = {"ev_gum": ev_gum, "extras": extras, "final": final}
    gum = load("gum", setting, ctx)
    s2_name = "S2" if gum.stability is not None else "S2-ns"
    regs = [r for r in REGRESSORS if r != "S2" or gum.stability is not None]
    sl = doc_slices(gum.docs)
    genres = sorted(set(gum.genres.tolist()))
    doc_fold = np.array([genres.index(g) for g in gum.genres])
    n_items = len(gum.f1)
    scores = {"S1": np.zeros(n_items), **{r: np.zeros(n_items) for r in regs}, **{b: np.zeros(n_items) for b in B_VARIANTS}}
    loso = {f"LOSO:{m}": np.zeros(n_items) for m in list(regs) + list(B_VARIANTS)}
    p_out = np.zeros(len(gum.y))
    p_b = {b: np.zeros(len(gum.y)) for b in B_VARIANTS}
    fold_models = {}
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(gum, regs, max(1, args.threads // args.workers))) as pool:
        for res in pool.imap_unordered(fold_job, range(len(genres))):
            g = res["g"]
            held_gap = gum.fold == g
            p_out[held_gap] = res["p_out"]
            for b in B_VARIANTS:
                p_b[b][held_gap] = res["p_b"][b]
            for name, (rows, vals) in res["scores"].items():
                (loso if name.startswith("LOSO:") else scores)[name][rows] = vals
            fold_models[g] = res["models"]
            print(json.dumps({"status": "A10_FOLD", "setting": args.setting, "genre": genres[g],
                              "seconds": res["seconds"]}), flush=True)

    all_scores = {**scores, **loso}
    checks = {}
    if args.setting == "T1":        # the lower-flat cell is the amendment-6 stack
        checks["S1_evidence_max_diff_vs_a6_stack"] = float(np.max(np.abs(p_out - ev_gum["stack"])))
    report = {"setting": args.setting, "label": setting["label"], "s2_reference": s2_name, "checks": checks, "corpora": {}}
    gum_out = {"eval": evaluate(gum, all_scores, args.bootstrap, s2_name)}
    gum_out["boundary"] = {n: {"logloss": logloss(p, gum.y), "auc": auc(p, gum.y)}
                           for n, p in {"S1": p_out, **p_b}.items()}

    def doc_model(i):
        return fold_models[doc_fold[i]]

    scorers = {"S1": lambda i, c: soft_f1(c, p_out[sl[i]]) / max_soft_f1(tuple(p_out[sl[i]]))}
    for b in B_VARIANTS:
        scorers[b] = (lambda b: lambda i, c: soft_f1(c, p_b[b][sl[i]]) / max_soft_f1(tuple(p_b[b][sl[i]])))(b)
    for r in [r for r in regs if r != "S2"]:
        def f(i, c, r=r):
            q = p_out[sl[i]]
            x = item_features(q, c, gum.term[sl[i]], FEATURE_KIND[r])
            base = soft_f1(c, q) / max_soft_f1(tuple(q)) if r == "R" else 0.0
            return base + predict_ridge(doc_model(i)["ridge"][r], x)
        scorers[r] = f
    gum_out["slopes"] = slopes(gum, scorers)
    gum_out["decision"] = decide(gum_out, s2_name) if args.setting in ("T2", "T3") else None
    report["corpora"]["gum"] = gum_out
    save(report)
    print(json.dumps({"status": "A10_GUM_DONE", "setting": args.setting, "seconds": round(time.time() - started, 1),
                      "decision": gum_out["decision"] and gum_out["decision"]["adopted"]}), flush=True)

    # final models on all GUM train, applied to the re-used test corpora (R)
    final_stack = fit_logistic_np(gum.x, gum.y, STACK_LAMBDA)
    theta0 = np.append(final_stack["w"], final_stack["b"])
    all_doc = np.arange(len(gum.docs))
    z_all = standardize(final_stack, gum.x)
    prob_all = QualityProblem(torch, z_all, gum.y, gum, all_doc)
    final_theta = {b: fit_quality(torch, prob_all, theta0, w) for b, w in B_VARIANTS.items()}
    q_gum = qnorm_items(gum, p_out, all_doc)
    for name in args.corpora:
        data = load(name, setting, ctx)
        t_regs = [r for r in regs if r != "S2" or data.stability is not None]
        ref = "S2" if "S2" in t_regs else "S2-ns"
        idx = np.arange(len(data.docs))
        p = predict_logistic_np(final_stack, data.x)
        sc = {"S1": qnorm_items(data, p, idx)}
        for r in t_regs:
            kind = FEATURE_KIND[r]
            sc[r] = regress(r, item_matrix(gum, p_out, kind, all_doc), gum.f1, q_gum, item_matrix(data, p, kind, idx), sc["S1"])
        for b, th in final_theta.items():
            sc[b] = qnorm_items(data, prob(final_stack, th, data.x), idx)
        report["corpora"][name] = {"eval": evaluate(data, sc, args.bootstrap, ref), "s2_reference": ref}
        save(report)
        print(json.dumps({"status": "A10_TEST_DONE", "setting": args.setting, "corpus": name}), flush=True)
    save(report)
    print(json.dumps({"status": "A10_COMPLETE", "setting": args.setting, "seconds": round(time.time() - started, 1)}), flush=True)



def save(report: dict) -> None:
    """Write the report so far (after GUM and after every test corpus)."""
    report["items_with_unaligned_boundaries"] = {f"{k[0]}|{k[1]}": v for k, v in ALIGNMENT_ITEMS.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"report_{report['setting']}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / f"summary_{report['setting']}.md").write_text(render(report))

# ------------------------------------------------------------------ rendering
def _ci(v, sign=True) -> str:
    f = (lambda x: f"{x:+.3f}") if sign else (lambda x: f"{x:.3f}")
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"


LABELS = {"S1": "S1 BCE stack → Q_norm（现行）", "S2": "S2 岭回归 QE（修订 6，含稳定性）", "S2-ns": "S2-ns（无稳定性）",
          "A": "A F1 回归器", "R": "R 残差 QE", "B-q": "B-q 质量损失", "B-qb": "B-qb 质量 + BCE",
          "B-qr": "B-qr 质量 + 排序", "B-qbr": "B-qbr 质量 + BCE + 排序"}


def render(r: dict) -> str:
    L = [f"# 修订 10：面向质量的融合 vs 事后质量回归（{r['setting']}：{r['label']}）", "",
         "控制变量：同一设计矩阵、同一 LOGO 折、同一批条目与同一个 Q_norm；只改训练目标或第二阶段。CI：领域内按文档重采样。"
         "分数在评估前四舍五入到 1e-10（数学上相等的分数严格并列，不由浮点噪声决定先后）。"
         f"与 S2 的比较以 {r['s2_reference']} 为准（换行场景与 EWT 没有稳定性数据）。", "",
         "核对：" + json.dumps(r.get("checks", {})) + "；Gold F1 含未对齐边界的条目数：" + json.dumps(r.get("items_with_unaligned_boundaries", {})), ""]
    for name, c in r["corpora"].items():
        ev = c["eval"]
        title = "GUM train（LOGO，判定）" if name == "gum" else f"{name}（R，全部 GUM train 拟合一次）"
        L += [f"## {title}", "", "| 方法 | 全局 Spearman [CI] | 文档之间 | 文档之内 | 文档内配对 | MAE | − S1 [CI] | − S2/S2-ns [CI] |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        ref = c.get("s2_reference", r["s2_reference"])
        for n, v in ev["table"].items():
            k = ev["contrasts"].get(n, {})
            d1 = _ci(k["minus_S1"]) if "minus_S1" in k else "—"
            d2 = _ci(k[f"minus_{ref}"]) if f"minus_{ref}" in k else "—"
            mae = f"{v['mae']:.3f}" if "mae" in v else "—"
            L.append(f"| {LABELS.get(n, n)} | {_ci(v['global_spearman'], False)} | {v['between_doc']:.3f} | {v['within_doc']:.3f} | "
                     f"{v['within_doc_pairwise']:.3f} | {mae} | {d1} | {d2} |")
        L.append("")
        if name == "gum":
            L += ["边界概率（留出空隙）：" + "，".join(f"{n} log loss {v['logloss']:.4f} / AUC {v['auc']:.4f}" for n, v in c["boundary"].items()), "",
                  "| 扰动斜率（理想 1） | " + " | ".join(SLOPE_TYPES) + " | 平均 |", "|---|" + "---:|" * (len(SLOPE_TYPES) + 1)]
            for n, v in c["slopes"].items():
                L.append(f"| {LABELS.get(n, n)} | " + " | ".join(f"{v[t]:.3f}" for t in SLOPE_TYPES) + f" | {np.mean(list(v.values())):.3f} |")
            L.append("")
            d = c.get("decision")
            if d:
                L += ["**预先写定的判定**：" + "；".join(f"{n} {'✓' if v['eligible'] else '✗'}（" +
                                                 "，".join(f"{k}{'✓' if ok else '✗'}" for k, ok in v["checks"].items()) + "）"
                                                 for n, v in d["rows"].items()) + f" → 采用 **{d['adopted']}**", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=sorted(SETTINGS), required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--corpora", nargs="*", default=list(TEST))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
