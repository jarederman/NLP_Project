"""Post-hoc diagnostic for amendments 9.2 + 9.3 (written after the factorial results; label P / R):
each stack cell used as a segmenter (p >= 0.5) against SaT segmenters in the matching layout
scenario, with paired, genre-stratified document-bootstrap CIs of the macro-F1 difference.
Flat cells are compared with SaT on flat text, newline cells with SaT on newline text.

SaT opponents (all on cased text, wtpsplit default thresholds, strict ">"):
  sat-3l-sm   the evaluated system (its actual outputs)
  sat-12l-sm  threshold 0.25; supervised on UD incl. GUM train (GUM is contaminated)
  sat-12l     threshold 0.025; newline pre-training only (no UD supervision)
SaT on newline text = model boundaries on the same probabilities + a hard boundary at every input
newline (``split_on_input_newlines``): the model treats newlines as spaces. For sat-12l this is
checked against its probabilities computed on the newline text itself.
Writes results/a9/segmenter_ci.json and segmenter_ci.md.
"""

from __future__ import annotations

import json
import pickle

import numpy as np

from .a6_combine import FINAL, STACK_LAMBDA, dev_evidence, e0_evidence, ngram_matrix, rule_matrix
from .a6_data import CORPORA, gold_sets, jsonl, logo_folds, predictions, surfaces
from .a6_signals import fit_logistic_np, predict_logistic_np
from .a6_data import load_gap_array
from .a9_casing import A9, logo_stack, macro_f1, paired_mean_ci, thresholded
from .a9_layout import CELLS, EWT_E0, NAMES, NL, RESULTS, break_mask, cell_name, design, e1_logit, e2_logit

VARIANTS = list(CELLS) + ["ref91"]
SAT12 = {"sat-12l-sm": ("sat12lsm-cased", 0.25), "sat-12l": ("sat12l-cased", 0.025)}


def with_breaks(docs, chosen: list[set], breaks: np.ndarray) -> list[set]:
    """Add every paragraph-newline gap (``breaks`` is the per-gap 0/1 mask) to each document's set."""
    out, k = [], 0
    for d, c in zip(docs, chosen):
        n = len(d.candidate_offsets)
        out.append(set(c) | {o for o, b in zip(d.candidate_offsets, breaks[k:k + n]) if b})
        k += n
    return out


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
    report = {"note": "post-hoc (P/R); stack p >= 0.5 vs SaT segmenters (3l-sm system, 12l-sm, 12l) in the matching layout scenario", "corpora": {}}
    lines = ["# 证据当切分器 vs SaT 系统（修订 9.2+9.3 的事后诊断，P/R）", "",
             "stack 证据 p ≥ 0.5 即切，与同一版式场景下的 SaT 切分器比 macro F1；配对差的 CI 为领域内按文档重采样 1000 次。"
             "GUM 为 LOGO，但 SaT-sm 在 GUM train 上训练过（被污染），只作参考。", "",
             "SaT 对手都读原文大小写、用 wtpsplit 默认阈值（sat-3l-sm / sat-12l-sm 0.25，sat-12l 0.025）；看到换行时在每个换行处强制切开"
             "（SaT 模型本身把换行当空格）。sat-12l-sm 在 UD（含 GUM train）上有监督训练；sat-12l 只有换行预训练。", ""]
    rows = ["| 语料 | stack 格 | 对手 | 场景 | stack F1 | 对手 F1 | 差 [95% CI] | 更好 / 更差文档 |", "|---|---|---|---|---:|---:|---:|---:|"]
    checks = []
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
        f_opp = {("sat-3l-sm", lay): macro_f1(docs, [sat[lay][(d.doc_id, "sat")] & set(d.candidate_offsets) for d in docs], gold)
                 for lay in ("flat", "nl")}
        check = {}
        for model, (key, thr) in SAT12.items():
            flat_sets = thresholded(docs, load_gap_array(A9[name], key, docs), thr, strict=True)
            f_opp[(model, "flat")] = macro_f1(docs, flat_sets, gold)
            f_opp[(model, "nl")] = macro_f1(docs, with_breaks(docs, flat_sets, nl), gold)
        # check: sat-12l probabilities computed on the newline text give the same decisions
        direct = with_breaks(docs, thresholded(docs, load_gap_array(NL[name], "sat12l-cased", docs), 0.025, strict=True), nl)
        approx = with_breaks(docs, thresholded(docs, load_gap_array(A9[name], "sat12l-cased", docs), 0.025, strict=True), nl)
        check["sat12l_nl_gap_mismatch"] = int(sum(len(a ^ b) for a, b in zip(direct, approx)))
        check["gaps"] = int(sum(len(d.candidate_offsets) for d in docs))
        out = {"check": check, "opponents": {f"{m}|{lay}": float(v.mean()) for (m, lay), v in f_opp.items()}}
        for v in VARIANTS:
            label, c1, c2, layout = spec(v)
            if name == "gum":
                p = logo_stack(x_gum[v], y, fold)
            else:
                x = design(e0, e1_logit(name, c1, layout, docs), e2_logit(name, c2, layout, docs), rules, ngrams,
                           nl if layout == "nl" else None)
                p = predict_logistic_np(fit_logistic_np(x_gum[v], y, STACK_LAMBDA), x)
            f_stack = macro_f1(docs, thresholded(docs, p, 0.5), gold)
            for opp in ("sat-3l-sm", "sat-12l-sm", "sat-12l"):
                f_sat = f_opp[(opp, layout)]
                diff = f_stack - f_sat
                key = label if opp == "sat-3l-sm" else f"{label} vs {opp}"
                out[key] = {"scenario": layout, "opponent": opp, "stack_f1": float(f_stack.mean()), "sat_f1": float(f_sat.mean()),
                            "diff": paired_mean_ci(diff, genres, 1000),
                            "better": int((diff > 1e-12).sum()), "worse": int((diff < -1e-12).sum())}
                r = out[key]
                rows.append(f"| {name} | {label} | {opp} | {layout} | {r['stack_f1']:.3f} | {r['sat_f1']:.3f} | "
                            f"{r['diff'][0]:+.3f} [{r['diff'][1]:+.3f}, {r['diff'][2]:+.3f}] | {r['better']} / {r['worse']} |")
        report["corpora"][name] = out
        checks.append(f"- {name}：sat-12l 在换行文本上直接算的概率与“无换行概率 + 换行处切开”相比，{check['gaps']} 个空隙中决定不同的有 "
                      f"{check['sat12l_nl_gap_mismatch']} 个；对手 macro F1：" + "，".join(f"{k} {v:.3f}" for k, v in out["opponents"].items()))
        print(json.dumps({"status": "A9_SEGMENTER_CI", "corpus": name, "check": check}), flush=True)
    lines += ["核对与对手 F1：", ""] + checks + [""] + rows
    (RESULTS / "segmenter_ci.json").write_text(json.dumps(report, indent=2) + "\n")
    (RESULTS / "segmenter_ci.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
