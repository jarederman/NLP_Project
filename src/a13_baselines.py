"""Amendment 13 (post-hoc, no decisions): the final evaluator against natural baselines, and a controlled,
null-standardized sensitivity test.

  sat      CPU  SaT-12l-sm (cased) probabilities on EWT dev and LinES train+dev (other corpora: amendment 9.1)
  analyze  CPU  V1 global Spearman + within-document pairwise, V2 system-ranking tau, V3 sensitivity z with a
                quality-neutral control -> results/a13/{report.json, summary.md}

Baselines: B1 punctuation rule, B2 sentence-length prior, B3 masked-LM punctuation probability (PuncEv) + Q_end,
B4 SaT-12l-sm segmentation as pseudo-reference, B5 SaT-12l-sm probabilities + Q_end, B6 consensus M0, B7 DocAgree.
Ours: final evaluator (flat) and (layout). With layout, baselines also see the paragraph breaks.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from types import SimpleNamespace

import numpy as np

from .a6_combine import FINAL, PLATT_LAMBDA, logit
from .a6_data import CORPORA, ROOT, SEED, SYSTEMS, doc_slices, gold_sets, jsonl, load_gap_array, logo_folds, \
    predictions, save_gap_array, surfaces
from .a6_signals import fit_logistic_np, length_prior, predict_logistic_np
from .a8_supplement import RATES, TERMINAL, _puncev_gum, category_codes, consensus_refs, credit, hard_f1, length_score, \
    resample_plan
from .a9_casing import A9, gap_probabilities
from .a9_layout import bootstrap, contrast
from .a11_analysis import LEVELS, TYPES2, Corpus as A11Corpus, candidates2, doc_taus, ranking_stats, soft_f1_end
from .a12_test import TEST, items_for
from .quality_agreement import tie_exact

RESULTS = ROOT / "results/a13"
NAMES = ("gum", "gentle", "lines", "ewt", "ewtdev", "linesx")
LABEL = {"gum": "GUM (D)", "gentle": "GENTLE", "lines": "LinES test", "ewt": "EWT test", "ewtdev": "EWT dev (fresh)",
         "linesx": "LinES train+dev (fresh)"}
FEATURES = {"gentle": ROOT / "data/real_system_gentle/a6/features/raw/test.pt",
            "lines": ROOT / "data/real_system_lines/a6/features/raw/test.pt",
            "ewt": ROOT / "data/real_system_ewt/features/raw/test.pt",
            "ewtdev": ROOT / "data/real_system_ewtdev/a12/features/raw/test.pt",
            "linesx": ROOT / "data/real_system_linesx/a12/features/raw/test.pt"}
PLATT_MODEL = ROOT / "data/real_system_ewt/models/raw_platt.pt"
SAT_THRESHOLD = 0.25
METHODS = ("Punctuation rule", "Length prior", "Masked LM (PuncEv)", "SaT-12l-sm output as reference",
           "SaT-12l-sm probabilities", "Consensus M0", "DocAgree", "Ours")
SENS_TYPES = ("merge_terminal", "merge_other_par", "merge_other_inpar", "split_terminal", "split_punct", "split_nopunct",
              "shift", "control")
REPS = 20


# ------------------------------------------------------------------ stage: SaT-12l-sm on the fresh corpora (CPU)
def run_sat(args) -> None:
    import torch
    from wtpsplit import SaT
    torch.set_num_threads(args.threads)
    sat = SaT("sat-12l-sm")
    for name in ("ewtdev", "linesx"):
        started = time.time()
        corpus = TEST[name]
        docs = surfaces(corpus)
        probs = list(sat.predict_proba([d.text for d in docs], stride=64, block_size=512, batch_size=8))
        out = []
        for d, pr in zip(docs, probs):
            pr = np.asarray(pr, dtype=float)
            if len(pr) != len(d.text):
                raise RuntimeError(f"SaT length mismatch in {d.doc_id}")
            out.append(gap_probabilities(pr, d.candidate_offsets))
        save_gap_array(corpus, "sat12lsm-cased", docs, np.concatenate(out))
        print(json.dumps({"status": "A13_SAT", "corpus": name, "seconds": round(time.time() - started, 1)}), flush=True)


# ------------------------------------------------------------------ data assembly
def puncev_from_features(name: str, docs) -> np.ndarray:
    import torch
    saved = torch.load(FEATURES[name], weights_only=False)
    if list(saved["doc_ids"]) != [d.doc_id for d in docs]:
        raise RuntimeError(f"Feature documents differ for {name}")
    m = torch.load(PLATT_MODEL, weights_only=False)
    x = logit(np.clip(saved["p_punct"].numpy().astype(float), 1e-7, 1 - 1e-7))
    z = (x - float(m["mu"][0])) / float(m["sd"][0])
    return 1 / (1 + np.exp(-(z * float(m["w"][0]) + float(m["b"]))))


def load_corpus(name: str) -> SimpleNamespace:
    """Gold, candidates, items and every evidence array of one corpus, document-major."""
    if name in ("ewtdev", "linesx"):
        corpus = TEST[name]
        docs = surfaces(corpus)
        gold = gold_sets(corpus)
        ev_npz = np.load(corpus.path("evidence_frozen.npz"))
        ev = {"Probe": ev_npz["Probe"], "final_flat": ev_npz["final (flat)"], "final_layout": ev_npz["final (layout)"]}
        par = np.zeros(sum(len(d.candidate_offsets) for d in docs), bool)
        if name == "ewtdev":
            rows = {r["doc_id"]: set(r["paragraph_break_offsets"]) for r in jsonl(TEST["ewtdev-nl"].data_root / "layout.jsonl")}
            par = np.concatenate([[o in rows.get(d.doc_id, set()) for o in d.candidate_offsets] for d in docs]).astype(bool)
        scen = {}
        for s, cname in (("flat", name), ("nl", f"{name}-nl")):
            if cname not in TEST:
                continue
            preds = predictions(TEST[cname])
            frozen = jsonl(TEST[cname].path("frozen_scores.jsonl"))
            scen[s] = {"cmat": np.concatenate([np.array([[o in preds[(d.doc_id, x)] for o in d.candidate_offsets] for x in SYSTEMS],
                                                        dtype=bool) for d in docs], axis=1),
                       "f1": items_for(cname, surfaces(TEST[cname])).gold,
                       "M0": np.array([r["scores"]["M0"] for r in frozen]), "DocAgree": np.array([r["scores"]["DocAgree"] for r in frozen])}
        sat = load_gap_array(corpus, "sat12lsm-cased", docs)
        puncev = puncev_from_features(name, docs)
        y = np.concatenate([[o in gold[d.doc_id] for o in d.candidate_offsets] for d in docs]).astype(bool)
        cats = np.concatenate([category_codes(d.text, d.candidate_offsets) for d in docs])
        genre = np.array([d.genre for d in docs])
    else:
        c = A11Corpus(name)
        docs = surfaces(A9[name])
        ev = {k: c.ev[k] for k in ("Probe", "final_flat", "final_layout")}
        scen = c.scen
        par, y, cats, genre = c.par, c.y, c.cats, c.genre
        sat = load_gap_array(A9[name], "sat12lsm-cased", docs)
        puncev = c.ev["PuncEv"] if "PuncEv" in c.ev else puncev_from_features(name, docs)
    return SimpleNamespace(name=name, docs=docs, sl=doc_slices(docs), ev=ev, scen=scen, par=par, y=y, cats=cats, genre=genre,
                           sat_raw=sat, puncev=puncev)


def sat_platt(gum: SimpleNamespace, target: SimpleNamespace, gum_fold: np.ndarray) -> np.ndarray:
    xg = logit(np.clip(gum.sat_raw, 1e-7, 1 - 1e-7))[:, None]
    if target.name == "gum":
        out = np.zeros(len(xg))
        for g in sorted(set(gum_fold.tolist())):
            held = gum_fold == g
            out[held] = predict_logistic_np(fit_logistic_np(xg[~held], gum.y[~held].astype(float), PLATT_LAMBDA), xg[held])
        return out
    xt = logit(np.clip(target.sat_raw, 1e-7, 1 - 1e-7))[:, None]
    return predict_logistic_np(fit_logistic_np(xg, gum.y.astype(float), PLATT_LAMBDA), xt)


def priors_for(c: SimpleNamespace, gum: SimpleNamespace, final: dict) -> list[np.ndarray]:
    if c.name != "gum":
        return [final["length_prior"]] * len(c.docs)
    gold = gold_sets(A9["gum"])
    by_genre = {g: length_prior([d for d in gum.docs if d.genre != g], gold) for g in sorted(set(gum.genre.tolist()))}
    return [by_genre[d.genre] for d in c.docs]


# ------------------------------------------------------------------ scorers on arbitrary candidate rows
def make_scorers(c: SimpleNamespace, scen: str, priors) -> dict:
    """name -> f(doc index i, rows (m, n)) -> (m,) scores. With layout, references include paragraph breaks."""
    lay = scen == "nl"
    rule = (c.cats == TERMINAL) | (c.par if lay else False)
    sat_ref = (c.sat_raw > SAT_THRESHOLD) | (c.par if lay else False)
    punc = np.where(c.par, 1.0, c.puncev) if lay else c.puncev
    satp = np.where(c.par, 1.0, c.sat_p) if lay else c.sat_p
    refs = [consensus_refs(c.scen[scen]["cmat"][:, sl]) for sl in c.sl]
    ours = c.ev["final_layout" if lay else "final_flat"]
    return {
        "Punctuation rule": lambda i, r: hard_f1(r, rule[c.sl[i]]),
        "Length prior": lambda i, r: length_score(r, priors[i]),
        "Masked LM (PuncEv)": lambda i, r: soft_f1_end(r, punc[c.sl[i]]),
        "SaT-12l-sm output as reference": lambda i, r: hard_f1(r, sat_ref[c.sl[i]]),
        "SaT-12l-sm probabilities": lambda i, r: soft_f1_end(r, satp[c.sl[i]]),
        "Consensus (gap level)": lambda i, r: np.mean([hard_f1(r, refs[i][s]) for s in range(len(SYSTEMS))], axis=0),
        "DocAgree": lambda i, r: np.zeros(len(np.atleast_2d(r))),
        "Ours": lambda i, r: soft_f1_end(r, ours[c.sl[i]]),
    }


def item_scores(c: SimpleNamespace, scen: str, scorers: dict) -> dict:
    cm = c.scen[scen]["cmat"]
    out = {n: np.concatenate([f(i, cm[:, sl]) for i, sl in enumerate(c.sl)]) for n, f in scorers.items()
           if n not in ("Consensus (gap level)", "DocAgree")}
    out["Consensus M0"] = c.scen[scen]["M0"]
    out["DocAgree"] = c.scen[scen]["DocAgree"]
    return {n: tie_exact(out[n]) for n in METHODS}


# ------------------------------------------------------------------ V1 + V2
def ranking(scores: dict, f1: np.ndarray, genre: np.ndarray, n_boot: int) -> dict:
    n = len(genre)
    plan = resample_plan(genre, n_boot)
    out = {}
    for name, v in scores.items():
        point = ranking_stats(v, f1, genre, np.arange(n))
        dt = doc_taus(v, f1, n)
        gd = np.array([ranking_stats(v, f1, genre, idx)[1] for idx in plan])
        dd = np.array([np.nanmean(dt[idx]) for idx in plan])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[name] = {"corpus_tau": float(point[0]), "genre_tau": [float(point[1]), float(np.nanpercentile(gd, 2.5)), float(np.nanpercentile(gd, 97.5))],
                         "doc_tau": [float(np.nanmean(dt)), float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))]}
    return out


def agreement(c: SimpleNamespace, scen: str, scorers: dict, n_boot: int) -> dict:
    scores = item_scores(c, scen, scorers)
    f1 = c.scen[scen]["f1"]
    point, draws, names = bootstrap(scores, f1, c.genre, n_boot)
    table = {n: {"global_spearman": [float(point[n][0]), float(np.percentile(draws[:, i, 0], 2.5)), float(np.percentile(draws[:, i, 0], 97.5))],
                 "within_doc_pairwise": float(point[n][1])} for i, n in enumerate(names)}
    base = [n for n in METHODS if n != "Ours"]
    best = max(base, key=lambda n: table[n]["global_spearman"][0])
    pairs = {f"Ours - {b}": contrast(draws, names, point, ["Ours"], [b]) for b in base}
    return {"table": table, "best_baseline": best, "ours_minus": pairs, "ranking": ranking(scores, f1, c.genre, n_boot)}


# ------------------------------------------------------------------ V3 sensitivity
def control_rows(gold: np.ndarray, doc_index: int) -> tuple[np.ndarray, np.ndarray]:
    """REPS pairs (base, moved): one false boundary at a random non-gold gap, moved to another one."""
    rng = np.random.default_rng([SEED, doc_index, 99])
    free = np.flatnonzero(~gold)
    a, b = [], []
    for _ in range(REPS):
        i, j = rng.choice(free, size=2, replace=False)
        x, y = gold.copy(), gold.copy()
        x[i], y[j] = True, True
        a.append(x)
        b.append(y)
    return np.array(a), np.array(b)


def sensitivity(c: SimpleNamespace, scorers: dict, n_boot: int, k_level: str = "k=1") -> dict:
    lv = LEVELS.index(k_level)
    names = list(scorers)
    n_docs = len(c.docs)
    resp = np.full((n_docs, len(names), len(SENS_TYPES)), np.nan)       # mean response per document
    det = np.full((n_docs, len(names), len(SENS_TYPES)), np.nan)
    ctrl_all = [[] for _ in names]                                       # per method: (doc, value) pairs
    for i, sl in enumerate(c.sl):
        g = c.y[sl]
        if not g.any() or (~g).sum() < 2:
            continue
        cand, t_idx, l_idx = candidates2(g, c.cats[sl], c.par[sl], i)
        base, moved = control_rows(g, i)
        for k, n in enumerate(names):
            s = tie_exact(scorers[n](i, cand))
            for j, t in enumerate(SENS_TYPES[:-1]):
                rows = np.flatnonzero((t_idx == TYPES2.index(t)) & (l_idx == lv))
                if len(rows):
                    d = s[0] - s[rows]
                    resp[i, k, j] = d.mean()
                    det[i, k, j] = credit(d).mean()
            d = tie_exact(scorers[n](i, base)) - tie_exact(scorers[n](i, moved))
            resp[i, k, -1] = d.mean()
            det[i, k, -1] = credit(d).mean()
            ctrl_all[k].extend((i, float(v)) for v in d)
    plan = resample_plan(c.genre, n_boot)
    out = {"level": k_level, "types": list(SENS_TYPES), "scorers": names,
           "documents_with_type": {t: int(np.isfinite(resp[:, 0, j]).sum()) for j, t in enumerate(SENS_TYPES)}, "z": {}, "detect": {}}
    for k, n in enumerate(names):
        ctrl = np.array([v for _, v in ctrl_all[k]])
        ctrl_doc = np.array([i for i, _ in ctrl_all[k]], dtype=int)
        sigma = float(ctrl.std()) if len(ctrl) else 0.0
        cnt = np.bincount(ctrl_doc, minlength=n_docs).astype(float)          # per-document moments of the control
        s1 = np.bincount(ctrl_doc, weights=ctrl, minlength=n_docs)
        s2 = np.bincount(ctrl_doc, weights=ctrl * ctrl, minlength=n_docs)

        def zvals(idx):
            if sigma <= 1e-12:            # the method never reacts to the neutral edit: z undefined (reported as NaN)
                return np.full(len(SENS_TYPES), np.nan)
            N, S, Q = cnt[idx].sum(), s1[idx].sum(), s2[idx].sum()
            var = Q / N - (S / N) ** 2 if N else sigma ** 2
            s0 = np.sqrt(var) if var > 1e-24 else sigma
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(resp[idx, k, :], 0) / s0

        ok_docs = np.flatnonzero(np.isfinite(resp[:, k, -1]))
        point = zvals(ok_docs)
        draws = []
        for idx in plan:
            idx = idx[np.isfinite(resp[idx, k, -1])]
            draws.append(zvals(idx))
        draws = np.array(draws)
        sel = draws - draws[:, [-1]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dpoint = np.nanmean(det[:, k, :], 0)
            ddraws = np.array([np.nanmean(det[idx, k, :], 0) for idx in plan])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sel_lo = np.nanpercentile(sel, 2.5, axis=0) if np.isfinite(sel).any() else np.full(len(SENS_TYPES), np.nan)
            sel_hi = np.nanpercentile(sel, 97.5, axis=0) if np.isfinite(sel).any() else np.full(len(SENS_TYPES), np.nan)
        out["z"][n] = {t: {"z": float(point[j]), "selectivity_ci": [float(sel_lo[j]), float(sel_hi[j])],
                           "detected": bool(sel_lo[j] > 0) if t != "control" and np.isfinite(sel_lo[j]) else None}
                       for j, t in enumerate(SENS_TYPES)}
        out["z"][n]["_sigma0"] = sigma
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["detect"][n] = {t: [float(dpoint[j]), float(np.nanpercentile(ddraws[:, j], 2.5)), float(np.nanpercentile(ddraws[:, j], 97.5))]
                                for j, t in enumerate(SENS_TYPES)}
        # primary criterion (post-hoc revision, see PREREGISTRATION amendment 13 note): detection rate CI above chance (0.5)
        out["detect"][n]["_detected"] = {t: bool(out["detect"][n][t][1] > 0.5) for t in SENS_TYPES if t != "control"}
    return out


# ------------------------------------------------------------------ analysis
def analyze(args) -> None:
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    gum = load_corpus("gum")
    gum_fold = logo_folds(gum.docs)
    report = {"note": "amendment 13 (post-hoc); ours on EWT dev / LinES train+dev is the amendment-12 one-shot test (T)", "corpora": {}}
    for name in args.corpora:
        started = time.time()
        c = gum if name == "gum" else load_corpus(name)
        c.sat_p = sat_platt(gum, c, gum_fold)
        priors = priors_for(c, gum, final)
        out = {"documents": len(c.docs), "scenarios": {}}
        if name == "ewt":            # PuncEv from cached features must equal the amendment-4 evidence file
            out["check_puncev_max_abs_diff"] = float(np.max(np.abs(puncev_from_features("ewt", c.docs) - c.puncev)))
        for scen in c.scen:
            out["scenarios"][scen] = agreement(c, scen, make_scorers(c, scen, priors), args.bootstrap)
        if name in args.sensitivity:
            sc = make_scorers(c, "flat", priors)
            if "nl" in c.scen:
                sc["Ours (layout)"] = make_scorers(c, "nl", priors)["Ours"]
            sc = {("Ours (flat)" if k == "Ours" else ("Consensus" if k == "Consensus (gap level)" else k)): v for k, v in sc.items()}
            out["sensitivity"] = {lv: sensitivity(c, sc, args.bootstrap, lv) for lv in ("k=1", "0.10")}
        report["corpora"][name] = out
        print(json.dumps({"status": "A13_CORPUS_DONE", "corpus": name, "seconds": round(time.time() - started, 1)}), flush=True)
        save(report)
    save(report)
    print(json.dumps({"status": "A13_COMPLETE"}), flush=True)


def save(report: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "summary.md").write_text(render(report))


def _ci(v, sign=False) -> str:
    f = (lambda x: f"{x:+.3f}") if sign else (lambda x: f"{x:.3f}")
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"


def render(r: dict) -> str:
    L = ["# 修订 13：最终评估器 vs 自然基线；带对照组的敏感度测试（事后分析）", "", r["note"], ""]
    for scen in ("flat", "nl"):
        cols = [n for n, c in r["corpora"].items() if scen in c["scenarios"]]
        if not cols:
            continue
        L += [f"## V1 全局 Spearman（{'带版式' if scen == 'nl' else '无换行'}）", "", "| 方法 | " + " | ".join(LABEL[n] for n in cols) + " |",
              "|---|" + "---:|" * len(cols)]
        for m in METHODS:
            L.append(f"| {m} | " + " | ".join(f"{r['corpora'][n]['scenarios'][scen]['table'][m]['global_spearman'][0]:.3f}" for n in cols) + " |")
        L += ["", "Ours − 最强基线：" + "；".join(
            f"{LABEL[n]} vs {r['corpora'][n]['scenarios'][scen]['best_baseline']} "
            f"{_ci(r['corpora'][n]['scenarios'][scen]['ours_minus']['Ours - ' + r['corpora'][n]['scenarios'][scen]['best_baseline']], True)}" for n in cols), ""]
        L += [f"文档内配对准确率 / 领域平均 τ / 文档平均 τ（{'带版式' if scen == 'nl' else '无换行'}）", "", "| 方法 | " + " | ".join(LABEL[n] for n in cols) + " |",
              "|---|" + "---:|" * len(cols)]
        for m in METHODS:
            cells = []
            for n in cols:
                s = r["corpora"][n]["scenarios"][scen]
                cells.append(f"{s['table'][m]['within_doc_pairwise']:.2f} / {s['ranking'][m]['genre_tau'][0]:.2f} / {s['ranking'][m]['doc_tau'][0]:.2f}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    for n, c in r["corpora"].items():
        if "sensitivity" not in c:
            continue
        for lv, s in c["sensitivity"].items():
            L += [f"## V3 敏感度（{LABEL[n]}，{lv}）", "", "z（附录用；σ₀ = 0 时无定义；* = 选择性 Δz 的 95% CI 下界 > 0）：", "", "| 方法 | " + " | ".join(s["types"]) + " | σ₀ |",
                  "|---|" + "---:|" * (len(s["types"]) + 1)]
            for m in s["scorers"]:
                z = s["z"][m]
                L.append(f"| {m} | " + " | ".join(("—" if z[t]['z'] != z[t]['z'] else f"{z[t]['z']:.1f}") + ('*' if z[t]['detected'] else '') for t in s["types"]) + f" | {z['_sigma0']:.2g} |")
            L += ["", "检出率（扰动后分数下降的比例；* = 95% CI 下界 > 0.5，主判据）：", "", "| 方法 | " + " | ".join(s["types"]) + " |", "|---|" + "---:|" * len(s["types"])]
            for m in s["scorers"]:
                dd = s["detect"][m]
                L.append(f"| {m} | " + " | ".join(f"{dd[t][0]:.2f}{'*' if dd['_detected'].get(t) else ''}" for t in s["types"]) + " |")
            L.append("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ post-hoc extras: family bias, fixed ensemble
def rank_bias(score: np.ndarray, f1: np.ndarray) -> dict:
    """Mean percentile-rank residual per system (amendment 6 definition): > 0 = overrated."""
    from scipy.stats import rankdata
    n = len(score)
    res = rankdata(tie_exact(score)) / n - rankdata(tie_exact(f1)) / n
    return {s: float(res[k::len(SYSTEMS)].mean()) for k, s in enumerate(SYSTEMS)}


def extra(args) -> None:
    """Post-hoc (P): (a) rank residual of each system under SaT-based evaluators vs ours (same-family bias);
    (b) a fixed, unfitted ensemble: the average of our final evidence and the SaT-12l-sm probabilities."""
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    gum = load_corpus("gum")
    gum_fold = logo_folds(gum.docs)
    out = {}
    for name in args.corpora:
        c = gum if name == "gum" else load_corpus(name)
        c.sat_p = sat_platt(gum, c, gum_fold)
        priors = priors_for(c, gum, final)
        res = {}
        for scen in c.scen:
            sc = make_scorers(c, scen, priors)
            lay = scen == "nl"
            ours = c.ev["final_layout" if lay else "final_flat"]
            satp = np.where(c.par, 1.0, c.sat_p) if lay else c.sat_p
            ens = (ours + satp) / 2
            sc["Ensemble (ours + SaT-12l-sm, average)"] = lambda i, r, ens=ens: soft_f1_end(r, ens[c.sl[i]])
            cm = c.scen[scen]["cmat"]
            names = ("SaT-12l-sm output as reference", "SaT-12l-sm probabilities", "Ours", "Ensemble (ours + SaT-12l-sm, average)")
            scores = {n: tie_exact(np.concatenate([sc[n](i, cm[:, sl]) for i, sl in enumerate(c.sl)])) for n in names}
            f1 = c.scen[scen]["f1"]
            point, draws, nm = bootstrap(scores, f1, c.genre, args.bootstrap)
            ens_name = names[-1]
            res[scen] = {"global_spearman": {n: [float(point[n][0]), float(np.percentile(draws[:, i, 0], 2.5)), float(np.percentile(draws[:, i, 0], 97.5))]
                                             for i, n in enumerate(nm)},
                         "within_doc_pairwise": {n: float(point[n][1]) for n in nm},
                         "ensemble_minus": {b: contrast(draws, nm, point, [ens_name], [b]) for b in names[:-1]},
                         "rank_bias": {n: rank_bias(scores[n], f1) for n in names}}
            if scen == ("nl" if "nl" in c.scen else "flat") and name in args.sensitivity:
                sub = {k: sc[k] for k in ("SaT-12l-sm probabilities", "Ours", ens_name)}
                res[scen]["sensitivity_k1"] = sensitivity(c, sub, args.bootstrap, "k=1")["detect"]
        out[name] = res
        print(json.dumps({"status": "A13_EXTRA", "corpus": name}), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "extra.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    L = ["# 修订 13 补充（事后分析 P）：同族偏差与不拟合的平均集成", ""]
    for name, res in out.items():
        for scen, r in res.items():
            L += [f"## {LABEL[name]}（{'带版式' if scen == 'nl' else '无换行'}）", "",
                  "| 方法 | 全局 Spearman [CI] | 文档内配对 | " + " | ".join(f"秩残差 {s}" for s in SYSTEMS) + " |", "|---|---:|---:|" + "---:|" * len(SYSTEMS)]
            for n, v in r["global_spearman"].items():
                L.append(f"| {n} | {_ci(v)} | {r['within_doc_pairwise'][n]:.3f} | " + " | ".join(f"{r['rank_bias'][n][s]:+.3f}" for s in SYSTEMS) + " |")
            L += ["", "集成 − 其他：" + "；".join(f"{k} {_ci(v, True)}" for k, v in r["ensemble_minus"].items())]
            if "sensitivity_k1" in r:
                L += ["", "检出率（1 个错误）：" + "；".join(f"{m}: " + ", ".join(f"{t} {v[t][0]:.2f}" for t in SENS_TYPES) for m, v in r["sensitivity_k1"].items())]
            L.append("")
    (RESULTS / "extra.md").write_text("\n".join(L) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["sat", "analyze", "extra"])
    parser.add_argument("--corpora", nargs="*", default=list(NAMES))
    parser.add_argument("--sensitivity", nargs="*", default=["gum", "gentle", "ewtdev", "linesx"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    {"sat": run_sat, "analyze": analyze, "extra": extra}[args.stage](args)


if __name__ == "__main__":
    main()
