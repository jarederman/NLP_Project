"""Post-hoc diagnostic for amendments 9.2 + 9.3 (written after the factorial results; label P / R):
each stack cell used as a segmenter (p >= 0.5) against the SaT system (sat-3l-sm) in the matching
layout scenario, with paired, genre-stratified document-bootstrap CIs of the macro-F1 difference.
Flat cells are compared with SaT on flat text, newline cells with SaT on newline text.
GUM numbers are LOGO but SaT-sm was trained on GUM train, so GUM is contaminated (reported only).
Writes results/a9/segmenter_ci.json and segmenter_ci.md.
"""

from __future__ import annotations

import json
import pickle

import numpy as np

from .a6_combine import FINAL, STACK_LAMBDA, dev_evidence, e0_evidence, ngram_matrix, rule_matrix
from .a6_data import CORPORA, gold_sets, jsonl, logo_folds, predictions, surfaces
from .a6_signals import fit_logistic_np, predict_logistic_np
from .a9_casing import A9, logo_stack, macro_f1, paired_mean_ci, thresholded
from .a9_layout import CELLS, EWT_E0, NAMES, NL, RESULTS, break_mask, cell_name, design, e1_logit, e2_logit

VARIANTS = list(CELLS) + ["ref91"]


def spec(v):
    """(label, E1 case, E2 case, layout) of a stack variant."""
    if v == "ref91":
        return "ref91", "cased", "lower", "flat"
    return cell_name(*v), v[0], v[0], v[1]


def main() -> None:
    gum_docs = surfaces(CORPORA["gum"])
    ev_gum, extras = dev_evidence(CORPORA["gum"], gum_docs, gold_sets(CORPORA["gum"]))
    y, fold = extras["y"], logo_folds(gum_docs)
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    gum_nl = break_mask("gum", gum_docs)
    x_gum = {v: design(ev_gum["E0"], e1_logit("gum", c1, layout, gum_docs), e2_logit("gum", c2, layout, gum_docs),
                       extras["rules"], extras["ngrams"], gum_nl if layout == "nl" else None)
             for v in VARIANTS for _, c1, c2, layout in [spec(v)]}
    report = {"note": "post-hoc (P/R); stack p >= 0.5 vs SaT system in the matching layout scenario", "corpora": {}}
    lines = ["# 证据当切分器 vs SaT 系统（修订 9.2+9.3 的事后诊断，P/R）", "",
             "stack 证据 p ≥ 0.5 即切，与同一版式场景下的 SaT 系统（sat-3l-sm）比 macro F1；配对差的 CI 为领域内按文档重采样 1000 次。"
             "GUM 为 LOGO，但 SaT-sm 在 GUM train 上训练过（被污染），只作参考。", "",
             "| 语料 | stack 格 | 场景 | stack F1 | SaT F1 | 差 [95% CI] | 更好 / 更差文档 |", "|---|---|---|---:|---:|---:|---:|"]
    for name in NAMES:
        docs = surfaces(A9[name])
        gold = gold_sets(A9[name])
        genres = np.array([d.genre for d in docs])
        nl = break_mask(name, docs)
        if name != "gum":
            e0 = (e0_evidence(EWT_E0, docs) if name == "ewt" else
                  np.concatenate([q["calibrated_boundary_probability"] for q in jsonl(CORPORA[name].path("evidence", "E0.jsonl"))]))
            rules, ngrams = rule_matrix(docs), ngram_matrix(final["ngram"], docs)
        sat = {"flat": predictions(A9[name]), "nl": predictions(NL[name])}
        out = {}
        for v in VARIANTS:
            label, c1, c2, layout = spec(v)
            if name == "gum":
                p = logo_stack(x_gum[v], y, fold)
            else:
                x = design(e0, e1_logit(name, c1, layout, docs), e2_logit(name, c2, layout, docs), rules, ngrams,
                           nl if layout == "nl" else None)
                p = predict_logistic_np(fit_logistic_np(x_gum[v], y, STACK_LAMBDA), x)
            f_stack = macro_f1(docs, thresholded(docs, p, 0.5), gold)
            f_sat = macro_f1(docs, [sat[layout][(d.doc_id, "sat")] & set(d.candidate_offsets) for d in docs], gold)
            diff = f_stack - f_sat
            out[label] = {"scenario": layout, "stack_f1": float(f_stack.mean()), "sat_f1": float(f_sat.mean()),
                          "diff": paired_mean_ci(diff, genres, 1000),
                          "better": int((diff > 1e-12).sum()), "worse": int((diff < -1e-12).sum())}
            r = out[label]
            lines.append(f"| {name} | {label} | {layout} | {r['stack_f1']:.3f} | {r['sat_f1']:.3f} | "
                         f"{r['diff'][0]:+.3f} [{r['diff'][1]:+.3f}, {r['diff'][2]:+.3f}] | {r['better']} / {r['worse']} |")
        report["corpora"][name] = out
        print(json.dumps({"status": "A9_SEGMENTER_CI", "corpus": name}), flush=True)
    (RESULTS / "segmenter_ci.json").write_text(json.dumps(report, indent=2) + "\n")
    (RESULTS / "segmenter_ci.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
