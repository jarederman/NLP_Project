"""Code review 2026-11-20, finding 1: size of the second-order leak in GUM leave-one-genre-out (LOGO).

The fusion model that scores genre g is trained on base features of the other genres h. Those features came
from base models trained on every genre except h, which includes g, so g's Gold reaches its own scorer
indirectly. This diagnostic refits two of the three learned base features in a nested way (for every outer
genre g and inner genre h != g, the base model is trained without both g and h):

  gpu  E0 probe+lr (frozen-BERT [MASK] states, lambda 1e-3; same features and solver as boundary_probe):
       15 LOGO refits (integrity check against the evidence used everywhere) + 15 x 14 nested refits.
  cpu  Kneser-Ney n-gram features, nested the same way; then the BCE stacks and the final B-qb evaluators are
       refitted per outer genre on the nested training features and scored on genre g with its unchanged
       (clean) features. Reports GUM global Spearman (Q_end) of nested vs the current (un-nested) LOGO.

The ModernBERT tagger (E2) is not nested (it would need 210 fine-tuning runs), and SaT-12l-sm was itself
trained on GUM train; GUM numbers stay development numbers (D). Test corpora are unaffected: every test-time
model is fitted on all of GUM train. Writes results/review_fixes/nested.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import time

import numpy as np

from .a6_data import CORPORA, ROOT, SYSTEMS, doc_slices, gold_sets, jsonl, surfaces

CACHE = ROOT / "data/lm_cache/review_fixes"
OUT = ROOT / "results/review_fixes"
FEATURE_ROOT = ROOT / "data/lm_cache/bert_probe_v1.1"
PROBE_LAMBDA = 1e-3                                   # E0 = probe+lr_lam0.001 (a6_combine.E0_DEV)


def genre_folds(docs) -> tuple[list[str], np.ndarray]:
    genres = sorted({d.genre for d in docs})
    return genres, np.concatenate([[genres.index(d.genre)] * len(d.candidate_offsets) for d in docs]).astype(int)


# ------------------------------------------------------------------ GPU: nested probe
def gpu(args) -> None:
    import torch
    from .a6_combine import E0_DEV
    from .boundary_probe import fit_logistic, load_features, predict_logistic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)                 # CPU fallback when the GPU queue is congested
    started = time.time()
    docs = surfaces(CORPORA["gum"])
    gold = gold_sets(CORPORA["gum"])
    features, _ = load_features(torch, FEATURE_ROOT, docs, variants=("raw",))
    f = features["raw"]
    clipped = np.clip(f["p"], 1e-7, 1 - 1e-7)
    x = torch.cat([torch.as_tensor((np.log(clipped) - np.log1p(-clipped))[:, None], dtype=torch.float32, device=device),
                   torch.as_tensor(f["hidden"], device=device).float(),
                   torch.as_tensor(f["lr"], dtype=torch.float32, device=device)], dim=1)   # CONFIGS["probe+lr"] order
    y_np = np.concatenate([[int(o in gold[d.doc_id]) for o in d.candidate_offsets] for d in docs]).astype(float)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    genres, fold = genre_folds(docs)
    ft = torch.as_tensor(fold, device=device)

    def fit_predict(train_mask, test_mask) -> np.ndarray:
        m = fit_logistic(torch, x[train_mask], y[train_mask], PROBE_LAMBDA)
        return predict_logistic(torch, m, x[test_mask]).double().cpu().numpy()

    logo = np.zeros(len(y_np))
    for g in range(len(genres)):
        logo[fold == g] = fit_predict(ft != g, ft == g)
    ref = np.concatenate([r["calibrated_boundary_probability"] for r in
                          sorted(jsonl(E0_DEV), key=lambda r: r["doc_id"])])
    integrity = float(np.max(np.abs(logo - ref)))
    print(json.dumps({"status": "NESTED_PROBE_LOGO", "integrity_max_abs_diff": integrity,
                      "seconds": round(time.time() - started, 1)}), flush=True)
    nested = np.tile(logo, (len(genres), 1))                     # row g: genre g keeps its clean LOGO value
    for g in range(len(genres)):
        for h in range(len(genres)):
            if h != g:
                nested[g, fold == h] = fit_predict((ft != g) & (ft != h), ft == h)
        print(json.dumps({"status": "NESTED_PROBE_OUTER", "genre": genres[g], "seconds": round(time.time() - started, 1)}),
              flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "e0_nested.npz", logo=logo, nested=nested, reference=ref, genres=np.array(genres),
                        doc_ids=np.array([d.doc_id for d in docs]), integrity=integrity)
    print(json.dumps({"status": "NESTED_PROBE_DONE", "seconds": round(time.time() - started, 1)}), flush=True)


# ------------------------------------------------------------------ CPU: nested n-grams, refits, comparison
_W: dict = {}


def _ngram_outer(g: int) -> tuple[int, np.ndarray]:
    from .a6_combine import ngram_matrix
    from .a6_signals import KneserNeyTrigram, NGRAM_NAMES, gold_sentences
    docs, gold, genres, fold = _W["docs"], _W["gold"], _W["genres"], _W["fold"]
    out = np.full((len(fold), len(NGRAM_NAMES)), np.nan)
    for h in range(len(genres)):
        if h == g:
            continue
        train = [d for d in docs if d.genre not in (genres[g], genres[h])]
        held = [d for d in docs if d.genre == genres[h]]
        out[fold == h] = ngram_matrix(KneserNeyTrigram(gold_sentences(train, gold)), held)
    return g, out


def q_end_items(data, p: np.ndarray) -> np.ndarray:
    from .a11_analysis import soft_f1_end
    return np.concatenate([soft_f1_end(data.cmat[i], p[s]) for i, s in enumerate(doc_slices(data.docs))])


def cpu(args) -> None:
    import torch
    from .a6_combine import FINAL, STACK_LAMBDA, dev_evidence, logit
    from .a6_signals import NGRAM_NAMES, RULE_NAMES, fit_logistic_np, predict_logistic_np
    from .a9_casing import logo_stack
    from .a9_layout import bootstrap, contrast
    from .a10_quality import B_VARIANTS, SETTINGS, QualityProblem, fit_quality, load, prob, standardize
    from .a11_analysis import _fold_quality
    torch.set_num_threads(args.threads)
    started = time.time()
    z = np.load(CACHE / "e0_nested.npz")
    docs = surfaces(CORPORA["gum"])
    gold = gold_sets(CORPORA["gum"])
    if z["doc_ids"].tolist() != [d.doc_id for d in docs]:
        raise RuntimeError("Nested probe cache does not match the GUM documents")
    ev_gum, extras = dev_evidence(CORPORA["gum"], docs, gold)
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    ctx = {"ev_gum": ev_gum, "extras": extras, "final": final}
    data = {t: load("gum", SETTINGS[t], ctx) for t in ("T1", "T2", "T3")}
    genres, fold = genre_folds(docs)
    ng = slice(3 + len(RULE_NAMES), 3 + len(RULE_NAMES) + len(NGRAM_NAMES))
    checks = {"probe_logo_vs_reference": float(z["integrity"]),
              "probe_logo_vs_dev_evidence": float(np.max(np.abs(z["logo"] - ev_gum["E0"]))),
              "design_e0_column": max(float(np.max(np.abs(d.x[:, 0] - logit(ev_gum["E0"])))) for d in data.values()),
              "design_ngram_columns": max(float(np.max(np.abs(d.x[:, ng] - extras["ngrams"]))) for d in data.values())}
    if not all(np.array_equal(d.fold, fold) and len(d.y) == len(fold) for d in data.values()):
        raise RuntimeError("Gap order of the amendment-10 design differs from the nested cache")
    print(json.dumps({"status": "NESTED_LOADED", **checks, "seconds": round(time.time() - started, 1)}), flush=True)
    _W.update(docs=docs, gold=gold, genres=genres, fold=fold)
    with mp.get_context("fork").Pool(args.workers) as pool:
        ngram_nested = dict(pool.map(_ngram_outer, range(len(genres))))
    print(json.dumps({"status": "NESTED_NGRAMS", "seconds": round(time.time() - started, 1)}), flush=True)

    def x_nested(d, g: int, parts: tuple[str, ...]) -> np.ndarray:
        x = d.x.copy()
        tr = d.fold != g
        if "probe" in parts:
            x[tr, 0] = logit(z["nested"][g][tr])
        if "ngram" in parts:
            x[tr, ng] = ngram_nested[g][tr]
        return x

    def nested_stack(d, parts) -> np.ndarray:
        out = np.zeros(len(d.y))
        for g in range(len(genres)):
            held = d.fold == g
            xg = x_nested(d, g, parts)
            out[held] = predict_logistic_np(fit_logistic_np(xg[~held], d.y[~held], STACK_LAMBDA), d.x[held])
        return out

    def nested_quality(d, parts) -> np.ndarray:                 # a11_analysis._fold_quality on nested features
        sl = doc_slices(d.docs)
        doc_fold = np.array([genres.index(x) for x in d.genres])
        out = np.zeros(len(d.y))
        for g in range(len(genres)):
            held = d.fold == g
            xg = x_nested(d, g, parts)
            tr_doc = np.flatnonzero(doc_fold != g)
            tr_gap = np.concatenate([np.arange(sl[i].start, sl[i].stop) for i in tr_doc])
            outer = fit_logistic_np(xg[~held], d.y[~held], STACK_LAMBDA)
            problem = QualityProblem(torch, standardize(outer, xg[tr_gap]), d.y[tr_gap], d, tr_doc)
            theta = fit_quality(torch, problem, np.append(outer["w"], outer["b"]), B_VARIANTS["B-qb"])
            out[held] = prob(outer, theta, d.x[held])
        return out

    variants = {"nested probe": ("probe",), "nested n-gram": ("ngram",), "nested probe + n-gram": ("probe", "ngram")}
    evaluators = {"Stack (flat, lower)": ("T1", "stack"), "+casing (flat)": ("T2", "stack"), "+layout": ("T3", "stack"),
                  "final (flat)": ("T2", "bqb"), "final (layout)": ("T3", "bqb")}
    report = {"checks": checks, "note": "GUM train LOGO (D); global Spearman of Q_end vs item Gold F1; "
                                        "paired difference nested - current, genre-stratified document bootstrap",
              "evaluators": {}}
    for name, (t, kind) in evaluators.items():
        d = data[t]
        current = logo_stack(d.x, d.y, d.fold) if kind == "stack" else _fold_quality(torch, d, B_VARIANTS["B-qb"])
        scores = {"current": q_end_items(d, current)}
        for v, parts in variants.items():
            p = nested_stack(d, parts) if kind == "stack" else nested_quality(d, parts)
            scores[v] = q_end_items(d, p)
        point, draws, names = bootstrap(scores, d.f1, d.genres, args.bootstrap)
        report["evaluators"][name] = {
            "global_spearman": {n: float(point[n][0]) for n in names},
            "nested_minus_current": {v: contrast(draws, names, point, [v], ["current"]) for v in variants}}
        print(json.dumps({"status": "NESTED_EVALUATOR", "evaluator": name, **report["evaluators"][name]["global_spearman"],
                          "seconds": round(time.time() - started, 1)}), flush=True)
        save(report)
    save(report)


def save(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nested.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    c = report["checks"]
    L = ["# 代码审查发现 1：GUM 留一领域的二阶泄漏有多大（嵌套重拟合）", "",
         "GUM train LOGO（D）。“当前” = 论文里的 LOGO 做法：融合模型在其他领域 h 的特征上训练，而这些特征来自"
         "“除 h 以外所有领域”（包括被打分的 g）训练的基础模型。“嵌套” = 对每个外层 g、内层 h，基础模型在去掉 g 和 h 的领域上训练。"
         "Probe 与 n-gram 做了嵌套；ModernBERT（E2）没有嵌套（需要 210 次微调）。被打分领域 g 的特征两种做法完全相同。", "",
         f"完整性检查：重拟合的 LOGO Probe 与论文所用证据的最大差 {c['probe_logo_vs_reference']:.1e}；"
         f"设计矩阵 E0 列 {c['design_e0_column']:.1e}，n-gram 列 {c['design_ngram_columns']:.1e}。", "",
         "| 评估器 | 当前 | 嵌套 Probe | 嵌套 n-gram | 嵌套 Probe+n-gram | Δ（Probe+n-gram − 当前，95% CI） |",
         "|---|---:|---:|---:|---:|---:|"]
    for name, e in report["evaluators"].items():
        g, d = e["global_spearman"], e["nested_minus_current"]["nested probe + n-gram"]
        L.append(f"| {name} | {g['current']:.4f} | {g['nested probe']:.4f} | {g['nested n-gram']:.4f} | "
                 f"{g['nested probe + n-gram']:.4f} | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] |")
    (OUT / "nested.md").write_text("\n".join(L) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["gpu", "cpu"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    {"gpu": gpu, "cpu": cpu}[args.stage](args)


if __name__ == "__main__":
    main()
