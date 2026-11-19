"""Amendment 9.1: does seeing letter case make SaT evidence a better *evaluator* component?

  gpu       per-gap SaT probabilities for {sat-3l-sm, sat-12l-sm, sat-12l} x {cased, lower} on
            GUM train, GUM dev, GENTLE, LinES and EWT test (inference only)
  analyze   CPU: reproduction check (sat-3l-sm cased = the evaluated system), Platt calibration
            on GUM train, evaluator agreement (normalized Soft F1), Stack vs Stack-cased,
            same-family residuals, perturbation slopes, evidence-as-segmenter F1 (diagnostic)

sat-3l-sm is the evaluated SaT system and is never used as evidence. GUM train numbers are
development (LOGO); every other corpus is a re-used test (label R) or GUM dev (supplementary dev).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np

from .a6_combine import E0_DEV, FINAL, PLATT_LAMBDA, STACK_LAMBDA, decomposition, dev_evidence, evaluate, \
    fairness, gold_items, logit, ngram_matrix, raw_signal, rule_matrix, soft_scores, stack_features, with_scores
from .a6_data import CORPORA, ROOT, SEED, SYSTEMS, Corpus, doc_slices, gold_sets, jsonl, load_gap_array, \
    logo_folds, predictions, safe_lower, save_gap_array, surfaces, write_evidence
from .a6_signals import fit_logistic_np, predict_logistic_np
from .a8_supplement import TYPES, DocCtx, candidates, category_codes, hard_f1, ols_slope, resample_plan, soft_f1
from .meta_eval import max_soft_f1
from .quality_agreement import load_items


RESULTS = ROOT / "results/a9"
MODELS = {"sat3lsm": "sat-3l-sm", "sat12lsm": "sat-12l-sm", "sat12l": "sat-12l"}
CASES = ("cased", "lower")
EVIDENCE = ("sat12l", "sat12lsm")                 # sat3lsm is the evaluated system: never evidence
SYSTEM_THRESHOLD = 0.25                           # wtpsplit default for "-sm" models
SLOPE_TYPES = ("merge_terminal", "merge_other", "split_terminal", "mixed")

A9 = {
    "gum": Corpus("gum", ROOT / "data/real_system", "train", ROOT / "data/real_system/train_cv/predictions",
                  ROOT / "data/real_system/train_cv/alignment_failures.jsonl", ROOT / "data/real_system/train_cv/a9"),
    "gumdev": Corpus("gumdev", ROOT / "data/real_system", "dev", ROOT / "data/real_system/predictions",
                     ROOT / "results/alignment_failures.jsonl", ROOT / "data/real_system/a9_dev"),
    "gentle": Corpus("gentle", ROOT / "data/real_system_gentle", "test", ROOT / "data/real_system_gentle/predictions",
                     ROOT / "data/real_system_gentle/alignment_failures.jsonl", ROOT / "data/real_system_gentle/a9"),
    "lines": Corpus("lines", ROOT / "data/real_system_lines", "test", ROOT / "data/real_system_lines/predictions",
                    ROOT / "data/real_system_lines/alignment_failures.jsonl", ROOT / "data/real_system_lines/a9"),
    "ewt": Corpus("ewt", ROOT / "data/real_system_ewt", "test", ROOT / "data/real_system_ewt/predictions",
                  ROOT / "data/real_system_ewt/alignment_failures.jsonl", ROOT / "data/real_system_ewt/a9"),
}
CANDIDATES = {"gum": ROOT / "data/real_system/train_cv/frozen_scores_cv.jsonl",
              "gumdev": ROOT / "data/real_system/frozen_scores.jsonl",
              "gentle": ROOT / "data/real_system_gentle/a6/frozen_scores.jsonl",
              "lines": ROOT / "data/real_system_lines/a6/frozen_scores.jsonl",
              "ewt": ROOT / "data/real_system_ewt/frozen/raw.jsonl"}
STACK_CORPORA = ("gum", "gentle", "lines")        # corpora with every stack component cached
LABEL = {"gum": "GUM train（LOGO，开发）", "gumdev": "GUM dev（补充开发）", "gentle": "GENTLE（R）",
         "lines": "LinES（R）", "ewt": "EWT test（R）"}


def gap_name(model: str, case: str) -> str:
    return f"{model}-{case}"


# ------------------------------------------------------------------ GPU stage
def gap_probabilities(prob: np.ndarray, offsets) -> np.ndarray:
    """Probability of a boundary at each whitespace gap: max(last char of the word, the whitespace)."""
    o = np.asarray(offsets, dtype=int)
    return np.maximum(prob[o - 1], prob[o]) if len(o) else np.zeros(0)


def run_gpu(args) -> None:
    import torch
    from wtpsplit import SaT

    docs_of = {n: surfaces(A9[n]) for n in args.corpora}
    for key, model_name in MODELS.items():
        sat = SaT(model_name)
        sat.half().to("cuda")
        for name, docs in docs_of.items():
            for case in CASES:
                started = time.time()
                texts = [d.text if case == "cased" else safe_lower(d.text) for d in docs]
                probs = list(sat.predict_proba(texts, stride=64, block_size=512, batch_size=64))
                out = []
                for d, prob in zip(docs, probs):
                    prob = np.asarray(prob, dtype=float)
                    if len(prob) != len(d.text):
                        raise RuntimeError(f"SaT returned {len(prob)} probabilities for {len(d.text)} chars in {d.doc_id}")
                    out.append(gap_probabilities(prob, d.candidate_offsets))
                save_gap_array(A9[name], gap_name(key, case), docs, np.concatenate(out))
                print(json.dumps({"status": "A9_SAT", "corpus": name, "model": model_name, "case": case,
                                  "seconds": round(time.time() - started, 1)}), flush=True)
        del sat
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ analysis helpers
def macro_f1(docs, chosen: list[set], gold: dict) -> np.ndarray:
    out = []
    for d, c in zip(docs, chosen):
        g = gold[d.doc_id] & set(d.candidate_offsets)
        out.append(1.0 if not c and not g else 2 * len(c & g) / (len(c) + len(g)))
    return np.array(out)


def thresholded(docs, p: np.ndarray, threshold: float, strict: bool = False) -> list[set]:
    return [{o for o, q in zip(d.candidate_offsets, p[sl]) if (q > threshold if strict else q >= threshold)}
            for d, sl in zip(docs, doc_slices(docs))]


def paired_mean_ci(diff: np.ndarray, genres: np.ndarray, n_boot: int) -> list[float]:
    plan = resample_plan(genres, n_boot)
    draws = [diff[idx].mean() for idx in plan]
    return [float(diff.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def platt(gum_raw: np.ndarray, gum_y: np.ndarray, fold: np.ndarray | None, target_raw: np.ndarray | None) -> np.ndarray:
    """LOGO-calibrated GUM probabilities (target_raw None) or a GUM-fitted map applied to a target corpus."""
    z = logit(gum_raw)
    if target_raw is None:
        out = np.zeros_like(z)
        for g in sorted(set(fold.tolist())):
            held = fold == g
            out[held] = predict_logistic_np(fit_logistic_np(z[~held, None], gum_y[~held], PLATT_LAMBDA), z[held, None])
        return out
    return predict_logistic_np(fit_logistic_np(z[:, None], gum_y, PLATT_LAMBDA), logit(target_raw)[:, None])


def corpus_items(name: str, docs):
    corpus = A9[name]
    if name in ("gum", "gentle", "lines"):
        e0 = E0_DEV if name == "gum" else CORPORA[name].path("evidence", "E0.jsonl")
        return gold_items(CORPORA[name], docs, e0, CANDIDATES[name])
    # any shared evidence file works for the Gold side of the items; write a neutral one
    path = write_evidence(corpus, "items_placeholder", docs, np.full(sum(len(d.candidate_offsets) for d in docs), 0.5))
    items, _ = load_items(path, data_root=corpus.data_root, split=corpus.split, candidates=CANDIDATES[name],
                          failures=corpus.failures)
    if [(a, b) for a, b in zip(items.doc.tolist(), items.system.tolist())] != [(d.doc_id, s) for d in docs for s in SYSTEMS]:
        raise RuntimeError(f"Item order mismatch in {name}")
    return items


def slopes(docs, gold: dict, evidence: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per document, the within-document slope of normalized Soft F1 on true F1 for each perturbation type
    (amendment 8.2 generator and seeds)."""
    out = {n: np.full((len(docs), len(SLOPE_TYPES)), np.nan) for n in evidence}
    for i, (d, sl) in enumerate(zip(docs, doc_slices(docs))):
        g = np.array([o in gold[d.doc_id] for o in d.candidate_offsets], dtype=bool)
        cats = category_codes(d.text, d.candidate_offsets)
        ctx = DocCtx(d.doc_id, d.genre, g, cats, {}, {}, np.zeros((4, len(g)), bool), np.zeros((4, len(g)), bool),
                     np.zeros((4, len(g))), np.ones(2))
        c, t_idx, _ = candidates(ctx, i)
        true = hard_f1(c, g)
        for n, p in evidence.items():
            q = p[sl]
            s = soft_f1(c, q) / (max_soft_f1(tuple(q)) if len(q) else 1.0)
            for k, t in enumerate(SLOPE_TYPES):
                rows = np.flatnonzero(t_idx == TYPES.index(t))
                if len(rows):
                    idx = np.concatenate([[0], rows])
                    out[n][i, k] = ols_slope(true[idx], s[idx])
    return out


# ------------------------------------------------------------------ analysis
def analyze(args) -> None:
    gum_c = CORPORA["gum"]
    gum_docs = surfaces(gum_c)
    gum_gold = gold_sets(gum_c)
    ev_gum, extras = dev_evidence(gum_c, gum_docs, gum_gold)
    y, fold = extras["y"], logo_folds(gum_docs)
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    names = list(dict.fromkeys(["gum", *args.corpora]))
    docs_of = {name: surfaces(A9[name]) for name in names}
    raw = {name: {gap_name(m, c): load_gap_array(A9[name], gap_name(m, c), docs_of[name]) for m in MODELS for c in CASES}
           for name in names}
    # stack design matrices (GUM: LOGO components; column 1 = E1 logit)
    x_gum = stack_features(ev_gum["E0"], raw_signal(gum_c, gum_docs, "E1"), raw_signal(gum_c, gum_docs, "E2"),
                           extras["rules"], extras["ngrams"])
    x_gum_cased = x_gum.copy()
    x_gum_cased[:, 1] = logit(raw["gum"]["sat12l-cased"])
    report = {"note": "amendment 9.1; GUM = development (LOGO); other corpora re-used (R) or GUM dev", "corpora": {}}
    for name in args.corpora:
        corpus = A9[name]
        docs = docs_of[name]
        gold = gold_sets(corpus)
        genres = np.array([d.genre for d in docs])
        preds = predictions(corpus)
        r = raw[name]
        # 1. reproduction: sat-3l-sm cased at the default threshold vs the evaluated SaT system
        repro = thresholded(docs, r["sat3lsm-cased"], SYSTEM_THRESHOLD, strict=True)
        system = [preds[(d.doc_id, "sat")] & set(d.candidate_offsets) for d in docs]
        n_gaps = sum(len(d.candidate_offsets) for d in docs)
        mismatch = sum(len(a ^ b) for a, b in zip(repro, system))
        out = {"documents": len(docs), "reproduction": {"gap_mismatch": int(mismatch), "gaps": int(n_gaps),
                                                         "agreement": 1 - mismatch / n_gaps}}
        # 2. calibrated evidence
        ev = {}
        for m in EVIDENCE:
            for c in CASES:
                k = gap_name(m, c)
                ev[k] = platt(raw["gum"][k], y, fold, None) if name == "gum" else platt(raw["gum"][k], y, None, r[k])
        if name in STACK_CORPORA:
            if name == "gum":
                ev["stack"], ev["stack-cased"] = ev_gum["stack"], logo_stack(x_gum_cased, y, fold)
                out["stack_reproduces_a6"] = float(np.max(np.abs(logo_stack(x_gum, y, fold) - ev_gum["stack"])))
            else:
                cc = CORPORA[name]
                e0 = np.concatenate([q["calibrated_boundary_probability"] for q in jsonl(cc.path("evidence", "E0.jsonl"))])
                x = stack_features(e0, raw_signal(cc, docs, "E1"), raw_signal(cc, docs, "E2"), rule_matrix(docs),
                                   ngram_matrix(final["ngram"], docs))
                frozen = np.concatenate([q["calibrated_boundary_probability"]
                                         for q in jsonl(cc.path("evidence", "stack.jsonl"))])
                ev["stack"] = predict_logistic_np(fit_logistic_np(x_gum, y, STACK_LAMBDA), x)
                out["stack_reproduces_a6"] = float(np.max(np.abs(ev["stack"] - frozen)))
                xc = x.copy()
                xc[:, 1] = logit(r["sat12l-cased"])
                ev["stack-cased"] = predict_logistic_np(fit_logistic_np(x_gum_cased, y, STACK_LAMBDA), xc)
        # 3. evaluator agreement
        items = corpus_items(name, docs)
        scores = {k: soft_scores(docs, preds, p)[1] for k, p in ev.items()}
        res = evaluate(with_scores(items, scores, list(scores)), args.bootstrap, SEED, list(scores)[0], args.workers)
        pairs = [(gap_name(m, "cased"), gap_name(m, "lower")) for m in EVIDENCE]
        if "stack" in ev:
            pairs.append(("stack-cased", "stack"))
        diffs = {}
        for a, b in pairs:
            rr = evaluate(with_scores(items, scores, (b, a)), args.bootstrap, SEED, b, args.workers)
            diffs[f"{a} - {b}"] = {m: [rr["point"][a][m] - rr["point"][b][m], *rr["paired_vs_baseline"][a][m]]
                                   for m in ("global_spearman", "within_doc_pairwise")}
        out["evaluator"] = {k: {"global_spearman": [res["point"][k]["global_spearman"], *res["ci"][k]["global_spearman"]],
                                "within_doc_pairwise": res["point"][k]["within_doc_pairwise"],
                                **decomposition(items, scores[k]),
                                "rank_bias": fairness(items, scores[k])["rank_bias"]} for k in scores}
        out["paired"] = diffs
        # 4. evidence as a segmenter (diagnostic) and the raw SaT models at their default thresholds
        seg = {}
        for k, p in ev.items():
            seg[f"{k} (p>=0.5)"] = macro_f1(docs, thresholded(docs, p, 0.5), gold)
        for m, thr in (("sat3lsm", 0.25), ("sat12lsm", 0.25), ("sat12l", 0.025)):
            for c in CASES:
                seg[f"{gap_name(m, c)} raw > {thr}"] = macro_f1(docs, thresholded(docs, r[gap_name(m, c)], thr, True), gold)
        for s in SYSTEMS:
            seg[f"system {s}"] = macro_f1(docs, [preds[(d.doc_id, s)] & set(d.candidate_offsets) for d in docs], gold)
        out["segmenter_macro_f1"] = {k: float(v.mean()) for k, v in seg.items()}
        out["segmenter_paired"] = {
            f"{a} - {b}": paired_mean_ci(seg[a] - seg[b], genres, args.bootstrap)
            for a, b in [("sat12lsm-cased raw > 0.25", "system sat"), ("sat12l-cased (p>=0.5)", "sat12l-lower (p>=0.5)"),
                         ("sat12lsm-cased (p>=0.5)", "sat12lsm-lower (p>=0.5)")]
            + ([("stack-cased (p>=0.5)", "stack (p>=0.5)"), ("stack-cased (p>=0.5)", "system sat")] if "stack" in ev else [])}
        # 5. perturbation slopes (blind spots)
        if name in STACK_CORPORA or name == "gumdev":
            sl = slopes(docs, gold, ev)
            plan = resample_plan(genres, args.bootstrap)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["slopes"] = {k: {t: float(np.nanmean(v[:, j])) for j, t in enumerate(SLOPE_TYPES)} for k, v in sl.items()}
                out["slopes_paired"] = {}
                for a, b in pairs:
                    d = sl[a] - sl[b]
                    out["slopes_paired"][f"{a} - {b}"] = {
                        t: [float(np.nanmean(d[:, j])), *np.percentile([np.nanmean(d[idx, j]) for idx in plan], [2.5, 97.5]).tolist()]
                        for j, t in enumerate(SLOPE_TYPES)}
        report["corpora"][name] = out
        print(json.dumps({"status": "A9_CORPUS_DONE", "corpus": name}), flush=True)
    g = report["corpora"].get("gum", {})
    if "paired" in g:
        sp = g["paired"]["stack-cased - stack"]["global_spearman"]
        mo = g["slopes_paired"]["stack-cased - stack"]["merge_other"]
        report["adopt_cased_evidence"] = bool(sp[1] > 0 or mo[1] > 0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "casing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "casing_summary.md").write_text(render(report))
    print(json.dumps({"status": "A9_COMPLETE", "adopt_cased_evidence": report.get("adopt_cased_evidence")}), flush=True)


def logo_stack(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y))
    for g in sorted(set(fold.tolist())):
        held = fold == g
        out[held] = predict_logistic_np(fit_logistic_np(x[~held], y[~held], STACK_LAMBDA), x[held])
    return out


# ------------------------------------------------------------------ rendering
def _ci(v, sign=True) -> str:
    f = (lambda x: f"{x:+.3f}") if sign else (lambda x: f"{x:.3f}")
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"


def render(r: dict) -> str:
    L = ["# 修订 9.1：证据看到大小写的效果（为评估器服务）", "",
         "GUM = GUM train 的 LOGO（开发）；GUM dev 为补充开发；GENTLE / LinES / EWT 为复用的测试集（R），只报告。"
         "sat-3l-sm 是被评系统，不作证据。CI：领域内按文档重采样。", "",
         f"按预先写定的判定采用 cased 证据：**{r.get('adopt_cased_evidence')}**", ""]
    for name, c in r["corpora"].items():
        rep = c["reproduction"]
        L += [f"## {LABEL[name]}（{c['documents']} 篇）", "",
              f"复现核对：sat-3l-sm（原文大小写，阈值 0.25）与被评 SaT 系统在 {rep['gaps']} 个空隙中不一致 {rep['gap_mismatch']} 个"
              f"（一致率 {rep['agreement']:.4f}）" + (f"；stack 复现修订 6 的最大误差 {c['stack_reproduces_a6']:.2e}" if "stack_reproduces_a6" in c else ""),
              "", "### 评估器（归一化 Soft F1）", "",
              "| 证据 | 全局 Spearman [CI] | 文档之间 | 文档之内 | 文档内配对 | " + " | ".join(f"残差 {s}" for s in SYSTEMS) + " |",
              "|---|---:|---:|---:|---:|" + "---:|" * len(SYSTEMS)]
        for k, v in c["evaluator"].items():
            L.append(f"| {k} | {_ci(v['global_spearman'], False)} | {v['between_doc']:.3f} | {v['within_doc']:.3f} | "
                     f"{v['within_doc_pairwise']:.3f} | " + " | ".join(f"{v['rank_bias'][s]:+.3f}" for s in SYSTEMS) + " |")
        L += ["", "| 配对差 | 全局 Spearman [CI] | 文档内配对 [CI] |", "|---|---:|---:|"]
        for k, v in c["paired"].items():
            L.append(f"| {k} | {_ci(v['global_spearman'])} | {_ci(v['within_doc_pairwise'])} |")
        if "slopes" in c:
            L += ["", "### 扰动斜率（文档内分数对真实 F1，理想 = 1；修订 8.2 的生成器与种子）", "",
                  "| 证据 | " + " | ".join(SLOPE_TYPES) + " |", "|---|" + "---:|" * len(SLOPE_TYPES)]
            for k, v in c["slopes"].items():
                L.append(f"| {k} | " + " | ".join(f"{v[t]:.3f}" for t in SLOPE_TYPES) + " |")
            L += ["", "| 斜率配对差 | " + " | ".join(SLOPE_TYPES) + " |", "|---|" + "---:|" * len(SLOPE_TYPES)]
            for k, v in c["slopes_paired"].items():
                L.append(f"| {k} | " + " | ".join(_ci(v[t]) for t in SLOPE_TYPES) + " |")
        L += ["", "### 诊断：当切分器用的 macro F1（只看空白空隙）", "", "| 切分 | macro F1 |", "|---|---:|"]
        for k, v in c["segmenter_macro_f1"].items():
            L.append(f"| {k} | {v:.3f} |")
        L += ["", "| 配对差 | macro F1 差 [CI] |", "|---|---:|"]
        for k, v in c["segmenter_paired"].items():
            L.append(f"| {k} | {_ci(v)} |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["gpu", "analyze"])
    parser.add_argument("--corpora", nargs="*", default=list(A9))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    {"gpu": run_gpu, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
