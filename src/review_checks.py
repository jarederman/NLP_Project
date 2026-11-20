"""Code review 2026-11-20: diagnostics for findings whose fix leaves the frozen artifacts untouched.

  ds         amendment-6 multi-system track (M1 / M1s / M2 / M3) before and after the leave-one-system-out
             Dawid-Skene fix (finding 2): GUM train LOGO (D) and the amendment-6 one-shot test corpora GENTLE /
             LinES test, recomputed post hoc (P) from the frozen evidence; the frozen scores stay the
             pre-registered result.
  consensus  official M0 (full boundary sets, reference without the scored system) vs the gap-level
             leave-one-out majority on the real systems (minor finding: two consensus definitions).
  sat        SaT-12l-sm on GUM train at fp32 on CPU vs the fp16 GPU evidence (minor finding: B4/B5 precision);
             effect on B4/B5 on GUM and on B5's Platt calibration for EWT dev / LinES train+dev.
Writes results/review_fixes/checks.json and checks.md. Run after the amendment-11 evidence cache exists.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time

import numpy as np
from scipy.stats import spearmanr

from .a6_combine import ANALYSIS_EVIDENCE, E0_DEV, EVIDENCE_ORDER, FINAL, M_REPORT, M_TRACK, RIDGE_LAMBDA, all_scores, \
    dev_evidence, dev_qe, gold_items, pick, terminal_mask, with_scores, soft_scores
from .a6_data import CORPORA, ROOT, SEED, SYSTEMS, gold_sets, jsonl, logo_folds, predictions, surfaces, vote_matrix
from .a6_signals import _ds_em, _sigmoid, _vote_log_odds, dawid_skene, fit_ridge
from .a8_supplement import consensus_refs, hard_f1
from .a9_casing import A9, gap_probabilities
from .a9_layout import fast_stats
from .a11_analysis import soft_f1_end
from .quality_agreement import evaluate, tie_exact

OUT = ROOT / "results/review_fixes"
CACHE = ROOT / "data/lm_cache/review_fixes"


def rho(a, b) -> float:
    return float(spearmanr(tie_exact(np.asarray(a, float)), tie_exact(np.asarray(b, float))).statistic)


# ------------------------------------------------------------------ finding 2: Dawid-Skene
def loo_before_fix(votes, strata, prior, per_stratum_rate=False) -> np.ndarray:
    """The pre-fix leave-one-out posterior: one EM fit on all systems, system s's term dropped afterwards."""
    fit = _ds_em(votes, strata, prior, 200, per_stratum_rate)
    lo = _vote_log_odds(votes, strata, fit["sensitivity"], fit["specificity"])
    total = lo.sum(0)
    return np.array([_sigmoid(fit["base_log_odds"] + total - lo[s]) for s in range(len(votes))])


def m_scores(docs, preds, terminal, prior, per_stratum: bool, before_fix: bool) -> np.ndarray:
    votes = vote_matrix(docs, preds)
    strata = terminal.astype(int)
    loo = (loo_before_fix(votes, strata, prior, per_stratum) if before_fix else
           dawid_skene(votes, strata, prior, per_stratum_rate=per_stratum)["leave_one_out"])
    return soft_scores(docs, preds, {s: loo[i] for i, s in enumerate(SYSTEMS)})[0]


def summarize(res: dict, names) -> dict:
    base = res["point"][res["baseline"]]["global_spearman"]
    out = {}
    for n in names:
        p = res["point"][n]["global_spearman"]
        pair = res["paired_vs_baseline"].get(n, {}).get("global_spearman")
        out[n] = {"global_spearman": p, "ci": res["ci"].get(n, {}).get("global_spearman"),
                  "minus_M0": None if pair is None else [p - base, *pair]}
    return out


def ds_check(args) -> dict:
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    sel = final["selection"]
    selected = sel["selected_evidence"]
    corpus = CORPORA["gum"]
    docs = surfaces(corpus)
    gold = gold_sets(corpus)
    ev, _ = dev_evidence(corpus, docs, gold)
    preds, terminal = predictions(corpus), terminal_mask(docs)
    items = gold_items(corpus, docs, E0_DEV, ROOT / "data/real_system/train_cv/frozen_scores_cv.jsonl")
    new, feats, _ = all_scores(corpus, docs, ev, selected)                       # corrected Dawid-Skene
    new.update(dev_qe(docs, feats, items.gold))
    old = {"M1": m_scores(docs, preds, terminal, None, False, True),
           "M1s": m_scores(docs, preds, terminal, None, True, True),
           "M2": m_scores(docs, preds, terminal, ev[selected], False, True)}
    old_feats = {"M3": np.column_stack([feats["S2"], old["M2"], new["M0"]])}
    old.update(dev_qe(docs, old_feats, items.gold))
    reported = json.loads((ROOT / "results/a6/dev_report.json").read_text())["M"]["point"]
    repro = {n: abs(rho(old[n], items.gold) - reported[n]["global_spearman"]) for n in old}
    scores = {"M0": new["M0"], **{n: new[n] for n in old}, **{f"{n} (before fix)": v for n, v in old.items()}}
    res = evaluate(with_scores(items, scores, list(scores)), args.bootstrap, SEED, "M0", args.workers)
    m_star = pick(res["point"], M_TRACK[1:])
    out = {"gum_dev": {"reproduces_reported_before_fix_max_abs": max(repro.values()), "frozen_M_star": sel["M_star"],
                       "M_star_after_fix": m_star, "scores": summarize(res, list(scores))}}
    # amendment-6 test corpora: post-hoc recomputation from the frozen evidence (frozen scores unchanged)
    qe_new = {name: fit_ridge(x, items.gold, RIDGE_LAMBDA) for name, x in feats.items()}
    for cname in ("gentle", "lines"):
        c = CORPORA[cname]
        tdocs = surfaces(c)
        tev = {n: np.concatenate([r["calibrated_boundary_probability"] for r in jsonl(c.path("evidence", f"{n}.jsonl"))])
               for n in EVIDENCE_ORDER + ANALYSIS_EVIDENCE}
        t_new, _, _ = all_scores(c, tdocs, tev, selected, qe_new)
        frozen = jsonl(c.path("frozen_scores.jsonl"))
        t_old = {n: np.array([r["scores"][n] for r in frozen]) for n in ("M1", "M1s", "M2", "M3") if n in frozen[0]["scores"]}
        tpreds, tterm = predictions(c), terminal_mask(tdocs)
        check = float(np.max(np.abs(m_scores(tdocs, tpreds, tterm, None, False, True) - t_old["M1"])))
        titems = gold_items(c, tdocs, c.path("evidence", "E0.jsonl"), c.path("frozen_scores.jsonl"))
        sc = {"M0": t_new["M0"], **{n: t_new[n] for n in t_old}, **{f"{n} (frozen, before fix)": v for n, v in t_old.items()}}
        r = evaluate(with_scores(titems, sc, list(sc)), args.bootstrap, SEED, "M0", args.workers)
        out[cname] = {"frozen_M1_reproduced_by_pre_fix_code_max_abs": check, "scores": summarize(r, list(sc))}
        print(json.dumps({"status": "REVIEW_DS", "corpus": cname}), flush=True)
    return out


# ------------------------------------------------------------------ minor: consensus definitions
def consensus_check() -> dict:
    from .a13_baselines import load_corpus
    out = {}
    for name in ("gum", "gentle", "lines", "ewt", "ewtdev", "linesx"):
        c = load_corpus(name)
        for scen, s in c.scen.items():
            cm, f1, m0 = s["cmat"], s["f1"], s["M0"]
            gap = []
            for sl in c.sl:
                refs = consensus_refs(cm[:, sl])
                gap.extend(float(hard_f1(cm[k:k + 1, sl], refs[k])[0]) for k in range(len(SYSTEMS)))
            gap = np.array(gap)
            out[f"{name}/{scen}"] = {"M0_vs_gold": rho(m0, f1), "gap_level_vs_gold": rho(gap, f1),
                                     "M0_vs_gap_level": rho(m0, gap), "items_differing": int(np.sum(np.abs(m0 - gap) > 1e-9)),
                                     "items": int(len(gap))}
    return out


# ------------------------------------------------------------------ minor: SaT precision
def sat_check(args) -> dict:
    import torch
    from wtpsplit import SaT
    from .a13_baselines import SAT_THRESHOLD, load_corpus, sat_platt
    torch.set_num_threads(args.threads)
    docs = surfaces(A9["gum"])
    path = CACHE / "sat12lsm_cased_fp32_cpu_gum.npy"
    if path.exists():
        p32 = np.load(path)
    else:
        started = time.time()
        sat = SaT("sat-12l-sm")                       # same call as a13_baselines.run_sat (CPU, fp32)
        probs = list(sat.predict_proba([d.text for d in docs], stride=64, block_size=512, batch_size=8))
        p32 = np.concatenate([gap_probabilities(np.asarray(pr, float), d.candidate_offsets) for d, pr in zip(docs, probs)])
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, p32)
        print(json.dumps({"status": "REVIEW_SAT_FP32", "seconds": round(time.time() - started, 1)}), flush=True)
    gum16 = load_corpus("gum")
    if [d.doc_id for d in gum16.docs] != [d.doc_id for d in docs]:
        raise RuntimeError("GUM document order differs")
    p16 = gum16.sat_raw
    gum32 = copy.copy(gum16)
    gum32.sat_raw = p32
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    out = {"gum_gaps": int(len(p32)), "max_abs_diff": float(np.max(np.abs(p32 - p16))),
           "mean_abs_diff": float(np.mean(np.abs(p32 - p16))), "max_abs_logit_diff": float(np.max(np.abs(lg(p32) - lg(p16)))),
           "spearman": rho(p32, p16),
           "flips_at_0.25": int(np.sum((p32 > SAT_THRESHOLD) != (p16 > SAT_THRESHOLD))),
           "flips_at_0.5": int(np.sum((p32 >= 0.5) != (p16 >= 0.5))), "agreement": {}}
    fold = logo_folds(gum16.docs)
    for name in ("gum", "ewtdev", "linesx"):
        tgt = None if name == "gum" else load_corpus(name)
        row = {}
        for tag, g in (("fp16_gum", gum16), ("fp32_gum", gum32)):
            c = g if tgt is None else tgt
            sat_p = sat_platt(g, c, fold)
            cm, f1 = c.scen["flat"]["cmat"], c.scen["flat"]["f1"]
            b4 = np.concatenate([hard_f1(cm[:, sl], c.sat_raw[sl] > SAT_THRESHOLD) for sl in c.sl])
            b5 = np.concatenate([soft_f1_end(cm[:, sl], sat_p[sl]) for sl in c.sl])
            st = fast_stats(np.array([b4, b5]), f1)
            row[tag] = {"B4_global_spearman": float(st[0, 0]), "B5_global_spearman": float(st[1, 0])}
        out["agreement"][name] = row
    return out


# ------------------------------------------------------------------ report
def render(r: dict) -> str:
    L = ["# 代码审查（2026-11-20）：诊断", ""]
    if "ds" in r:
        d = r["ds"]
        L += ["## 发现 2：Dawid–Skene 的留一系统（修订 6 的 M 线）", "",
              f"GUM（LOGO，D）：修复前代码复现已报告值的最大误差 {d['gum_dev']['reproduces_reported_before_fix_max_abs']:.1e}；"
              f"冻结的 M* = {d['gum_dev']['frozen_M_star']}，修复后按同一规则选 M* = {d['gum_dev']['M_star_after_fix']}。", "",
              "| 语料 | 评分器 | 全局 Spearman | − M0 |", "|---|---|---:|---:|"]
        for corpus in ("gum_dev", "gentle", "lines"):
            for n, v in d[corpus]["scores"].items():
                diff = v["minus_M0"]
                L.append(f"| {corpus} | {n} | {v['global_spearman']:.3f} | "
                         f"{'' if diff is None else f'{diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}]'} |")
        L += ["", "GENTLE / LinES test 的修复后数值是事后重算（P）：冻结分数不变，仍是预注册结果。", ""]
    if "consensus" in r:
        L += ["## 次要：两种共识定义（真实系统条目）", "", "| 语料/场景 | M0 vs Gold | 空隙级 vs Gold | M0 vs 空隙级 | 不同条目 |",
              "|---|---:|---:|---:|---:|"]
        L += [f"| {k} | {v['M0_vs_gold']:.3f} | {v['gap_level_vs_gold']:.3f} | {v['M0_vs_gap_level']:.3f} | "
              f"{v['items_differing']}/{v['items']} |" for k, v in r["consensus"].items()] + [""]
    if "sat" in r:
        s = r["sat"]
        L += ["## 次要：SaT-12l-sm 的精度（GUM train，CPU fp32 vs GPU fp16）", "",
              f"{s['gum_gaps']} 个空隙：最大 |Δp| = {s['max_abs_diff']:.2e}，平均 {s['mean_abs_diff']:.2e}，"
              f"Spearman {s['spearman']:.6f}；阈值 0.25 处翻转 {s['flips_at_0.25']} 个，0.5 处 {s['flips_at_0.5']} 个。", "",
              "| 语料 | GUM 概率 | B4 全局 Spearman | B5 全局 Spearman |", "|---|---|---:|---:|"]
        for name, row in s["agreement"].items():
            L += [f"| {name} | {tag} | {v['B4_global_spearman']:.4f} | {v['B5_global_spearman']:.4f} |" for tag, v in row.items()]
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", nargs="*", default=["ds", "consensus", "sat"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "checks.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for name in args.checks:
        started = time.time()
        report[name] = {"ds": ds_check, "consensus": lambda a: consensus_check(), "sat": sat_check}[name](args)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        (OUT / "checks.md").write_text(render(report))
        print(json.dumps({"status": "REVIEW_CHECK_DONE", "check": name, "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
