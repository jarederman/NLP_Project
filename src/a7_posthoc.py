"""Amendment 7: post-hoc robustness analyses of amendment 6 (analysis only; nothing is refitted
for the test corpora and no amendment-6 artifact is overwritten).

  posthoc   7.1 soft evidence vs hard pseudo-reference; 7.2 same-family bias per single evidence;
            7.3 small-sample sensitivity (unstratified bootstrap, leave-one-genre-out, S1 vs M0)
  seeds     7.4 fine-tuning seed variance of E2 on GUM train (LOGO), selection re-run per seed

GENTLE / LinES parts are post-test analyses and are labelled as such in the output.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import a6_combine as a6
from .a6_combine import ANALYSIS_EVIDENCE, E0_DEV, EVIDENCE_ORDER, RAW_SIGNALS, all_scores, boundary_f1, \
    ci_positive, dev_evidence, dev_qe, error_overlap, evaluate, fairness, gold_items, pick, soft_scores, with_scores
from .a6_data import CORPORA, ROOT, SEED, SYSTEMS, doc_slices, gold_sets, jsonl, predictions, surfaces
from .quality_agreement import Items, statistics


RESULTS = ROOT / "results/a7"
SINGLE_EVIDENCE = ("E0", "E1", "E1sm", "E2", "E2b", "stack")
SEED_OFFSETS = (0, 1, 2, 3)
FAMILY_MARGIN = 0.05


# ------------------------------------------------------------------ helpers
def hard_reference_scores(docs, preds, p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Boundary F1 of each entry against the evidence thresholded at ``threshold`` (whitespace gaps only)."""
    out = []
    for d, sl in zip(docs, doc_slices(docs)):
        ref = {o for o, q in zip(d.candidate_offsets, p[sl]) if q >= threshold}
        for s in SYSTEMS:
            cand = {o for o in d.candidate_offsets if o in preds[(d.doc_id, s)]}
            out.append(boundary_f1(cand, ref))
    return np.array(out)


def unstratified(items: Items) -> Items:
    """Same items with a single resampling stratum (all documents in one pool)."""
    return Items(items.doc, np.full(len(items.doc), "all"), items.system, items.gold, dict(items.scores))


def diff_ci(result: dict, name: str) -> dict:
    point = result["point"][name]["global_spearman"] - result["point"][result["baseline"]]["global_spearman"]
    return {"diff": point, "ci": result["paired_vs_baseline"][name]["global_spearman"]}


def gum_context():
    corpus = CORPORA["gum"]
    docs = a6.surfaces(corpus)
    gold = gold_sets(corpus)
    ev, extras = dev_evidence(corpus, docs, gold)
    items = gold_items(corpus, docs, E0_DEV, ROOT / "data/real_system/train_cv/frozen_scores_cv.jsonl")
    scores, features, _ = all_scores(corpus, docs, ev, "stack")
    scores.update(dev_qe(docs, features, items.gold))
    return corpus, docs, ev, items, scores, extras["y"]


def test_context(name: str):
    corpus = CORPORA[name]
    docs = surfaces(corpus)
    gold = gold_sets(corpus)
    items = gold_items(corpus, docs, corpus.path("evidence", "E0.jsonl"), corpus.path("frozen_scores.jsonl"))
    frozen = jsonl(corpus.path("frozen_scores.jsonl"))
    scores = {n: np.array([r["scores"][n] for r in frozen]) for n in frozen[0]["scores"]}
    ev = {n: np.concatenate([r["calibrated_boundary_probability"] for r in jsonl(corpus.path("evidence", f"{n}.jsonl"))])
          for n in EVIDENCE_ORDER + ANALYSIS_EVIDENCE}
    y = np.concatenate([[int(o in gold[d.doc_id]) for o in d.candidate_offsets] for d in docs]).astype(float)
    return corpus, docs, ev, items, scores, y


# ------------------------------------------------------------------ 7.1 - 7.3
def corpus_posthoc(name: str, args) -> tuple[dict, Items, dict]:
    corpus, docs, ev, items, scores, y = gum_context() if name == "gum" else test_context(name)
    preds = predictions(corpus)
    scores["H(E0)"] = hard_reference_scores(docs, preds, ev["E0"])
    scores["H(stack)"] = hard_reference_scores(docs, preds, ev["stack"])
    out = {"items": int(len(items.gold)), "domains": sorted(set(items.domain.tolist()))}

    # 7.1 soft evidence vs hard pseudo-reference
    hard_stack = evaluate(with_scores(items, scores, ("H(stack)", "S1")), args.bootstrap, SEED, "H(stack)", args.workers)
    hard_e0 = evaluate(with_scores(items, scores, ("H(E0)", "S0")), args.bootstrap, SEED, "H(E0)", args.workers)
    out["hard_reference"] = {
        "global_spearman": {n: r["point"][n]["global_spearman"] for r in (hard_stack, hard_e0) for n in r["point"]},
        "S1_minus_H(stack)": diff_ci(hard_stack, "S1"), "S0_minus_H(E0)": diff_ci(hard_e0, "S0"),
        "within_doc_pairwise": {n: r["point"][n]["within_doc_pairwise"] for r in (hard_stack, hard_e0) for n in r["point"]},
    }

    # 7.2 same-family bias per single evidence
    fam = {}
    for n in SINGLE_EVIDENCE:
        f = fairness(items, scores[f"ev:{n}"])
        fam[n] = {"rank_bias": f["rank_bias"], "corpus_ranking": f["corpus_ranking"], "top_by_gold": f["top_by_gold"],
                  "error_overlap": error_overlap(docs, preds, ev[n], y)}
    bias = {n: fam[n]["rank_bias"] for n in SINGLE_EVIDENCE}
    out["same_family"] = {
        "per_evidence": fam,
        "sat_bias_E1_minus_E0": bias["E1"]["sat"] - bias["E0"]["sat"],
        "sat_bias_E1sm_minus_E0": bias["E1sm"]["sat"] - bias["E0"]["sat"],
        "stanza_bias_E2_minus_E0": bias["E2"]["stanza"] - bias["E0"]["stanza"],
    }
    if name != "gum":
        out["same_family"]["flags"] = {k: v > FAMILY_MARGIN for k, v in out["same_family"].items()
                                       if k.endswith("_minus_E0")}

    # 7.3 sensitivity (all corpora get (a) and (c); (b) only for multi-genre test corpora)
    out["sensitivity"] = sensitivity(items, scores, args)
    if name == "gentle":
        out["sensitivity"]["leave_one_genre_out"] = leave_one_genre_out(items, scores)
    return out, items, {n: scores[n] for n in ("S0", "S1", "M0", "M3")}


def sensitivity(items: Items, scores: dict, args) -> dict:
    out = {}
    for label, it in (("stratified", items), ("unstratified", unstratified(items))):
        s = evaluate(with_scores(it, scores, ("S0", "S1")), args.bootstrap, SEED, "S0", args.workers)
        m = evaluate(with_scores(it, scores, ("M0", "M3", "S1")), args.bootstrap, SEED, "M0", args.workers)
        out[label] = {"S1_minus_S0": diff_ci(s, "S1"), "M3_minus_M0": diff_ci(m, "M3"),
                      "S1_minus_M0": diff_ci(m, "S1")}
    return out


def leave_one_genre_out(items: Items, scores: dict) -> dict:
    scored = with_scores(items, scores, ("S0", "S1", "M0", "M3"))
    out = {}
    for genre in sorted(set(items.domain.tolist())):
        idx = np.flatnonzero(items.domain != genre)
        st = statistics(scored, idx)
        g = {n: st[n]["global_spearman"] for n in st}
        out[genre] = {"S1_minus_S0": g["S1"] - g["S0"], "M3_minus_M0": g["M3"] - g["M0"], **g}
    out["all_positive"] = {k: all(v[k] > 0 for v in out.values()) for k in ("S1_minus_S0", "M3_minus_M0")}
    return out


def posthoc(args) -> None:
    report, pooled = {"note": "amendment 7.1-7.3; GENTLE / LinES parts are post-test analyses"}, []
    for name in ("gum", "gentle", "lines"):
        report[name], items, scores = corpus_posthoc(name, args)
        if name != "gum":
            pooled.append((items, scores))
        print(json.dumps({"status": "A7_POSTHOC", "corpus": name}), flush=True)
    items = Items(*(np.concatenate([getattr(i, f) for i, _ in pooled]) for f in ("doc", "domain", "system", "gold")))
    scores = {n: np.concatenate([s[n] for _, s in pooled]) for n in ("S0", "S1", "M0", "M3")}
    report["pooled_test"] = {"sensitivity": sensitivity(items, scores, args)}
    g = report["gentle"]["sensitivity"]
    report["gentle_robust"] = bool(g["unstratified"]["S1_minus_S0"]["ci"][0] > 0
                                   and g["unstratified"]["M3_minus_M0"]["ci"][0] > 0
                                   and all(g["leave_one_genre_out"]["all_positive"].values()))
    dev = json.loads((ROOT / "results/a6/dev_report.json").read_text())
    report["gum_reproduces_dev"] = abs(report["gum"]["hard_reference"]["global_spearman"]["S1"]
                                       - dev["S"]["point"]["S1"]["global_spearman"]) < 1e-9
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "posthoc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "posthoc_summary.md").write_text(render_posthoc(report))
    print(json.dumps({"status": "A7_POSTHOC_COMPLETE", "gentle_robust": report["gentle_robust"],
                      "gum_reproduces_dev": report["gum_reproduces_dev"]}), flush=True)


# ------------------------------------------------------------------ 7.4
def seeds(args) -> None:
    corpus = CORPORA["gum"]
    docs = a6.surfaces(corpus)
    gold = gold_sets(corpus)
    preds = predictions(corpus)
    items = gold_items(corpus, docs, E0_DEV, ROOT / "data/real_system/train_cv/frozen_scores_cv.jsonl")
    original = RAW_SIGNALS["E2"]
    ev_names = [f"ev:{n}" for n in EVIDENCE_ORDER]
    runs = {}
    try:
        for k in SEED_OFFSETS:
            RAW_SIGNALS["E2"] = (f"ft-modernbert-seed{k}_logit" if k else "ft-modernbert_logit", "logit")
            ev, _ = dev_evidence(corpus, docs, gold)
            ev_scores = {f"ev:{n}": soft_scores(docs, preds, ev[n])[1] for n in EVIDENCE_ORDER}
            ev_res = evaluate(with_scores(items, ev_scores, ev_names), args.bootstrap, SEED, "ev:E0", args.workers)
            eligible = [n for n in ev_names[1:] if ci_positive(ev_res, n)]
            selected = (pick(ev_res["point"], ev_names, eligible) if eligible else "ev:E0").split(":", 1)[1]
            scores, features, _ = all_scores(corpus, docs, ev, selected)
            scores.update(dev_qe(docs, features, items.gold))
            s = evaluate(with_scores(items, scores, ("S0", "S1")), args.bootstrap, SEED, "S0", args.workers)
            m = evaluate(with_scores(items, scores, ("M0", "M3")), args.bootstrap, SEED, "M0", args.workers)
            runs[k] = {"fold_seed_base": SEED + 100 * k, "selected": selected, "eligible": eligible,
                       "evidence": {n: {"global_spearman": ev_res["point"][f"ev:{n}"]["global_spearman"],
                                        **({"diff_vs_E0": diff_ci(ev_res, f"ev:{n}")} if n != "E0" else {})}
                                    for n in EVIDENCE_ORDER},
                       "S1": s["point"]["S1"]["global_spearman"], "M3": m["point"]["M3"]["global_spearman"],
                       "S1_minus_S0": diff_ci(s, "S1"), "M3_minus_M0": diff_ci(m, "M3")}
            print(json.dumps({"status": "A7_SEED", "k": k, "selected": selected, "S1": runs[k]["S1"],
                              "M3": runs[k]["M3"]}), flush=True)
    finally:
        RAW_SIGNALS["E2"] = original
    summary = {}
    for key, get in (("E2", lambda r: r["evidence"]["E2"]["global_spearman"]),
                     ("stack", lambda r: r["evidence"]["stack"]["global_spearman"]),
                     ("S1", lambda r: r["S1"]), ("M3", lambda r: r["M3"])):
        v = np.array([get(r) for r in runs.values()])
        summary[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}
    stable = all(r["selected"] == "stack" and r["S1_minus_S0"]["ci"][0] > 0 and r["M3_minus_M0"]["ci"][0] > 0
                 for r in runs.values())
    report = {"note": "amendment 7.4; GUM train LOGO only", "runs": runs, "summary": summary,
              "selection_stable_over_seeds": stable}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "seeds_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "seeds_summary.md").write_text(render_seeds(report))
    print(json.dumps({"status": "A7_SEEDS_COMPLETE", "stable": stable}), flush=True)


# ------------------------------------------------------------------ rendering
def _ci(d: dict) -> str:
    return f"{d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"


CORPUS_LABEL = {"gum": "GUM 开发（LOGO）", "gentle": "GENTLE（测试后）", "lines": "LinES（测试后）"}


def render_posthoc(r: dict) -> str:
    names = ("gum", "gentle", "lines")
    lines = ["# 修订 7：测试后稳健性分析（只作分析）", "",
             "GENTLE / LinES 上的数字都是测试后分析；修订 6 的选型、判据与结论不变。CI 为 1000 次 bootstrap。", "",
             f"GUM 重算与开发报告一致：{r['gum_reproduces_dev']}", "",
             "## 7.1 软证据 vs 硬伪参考（证据 p ≥ 0.5 当参考答案）", "",
             "| 全局 Spearman | " + " | ".join(CORPUS_LABEL[n] for n in names) + " |", "|---|" + "---:|" * len(names)]
    for key, label in (("S0", "S0 软（E0）"), ("H(E0)", "H(E0) 硬"), ("S1", "S1 软（stack）"), ("H(stack)", "H(stack) 硬")):
        lines.append(f"| {label} | " + " | ".join(f"{r[n]['hard_reference']['global_spearman'][key]:.3f}" for n in names) + " |")
    for key, label in (("S1_minus_H(stack)", "S1 − H(stack) [CI]"), ("S0_minus_H(E0)", "S0 − H(E0) [CI]")):
        lines.append(f"| {label} | " + " | ".join(_ci(r[n]["hard_reference"][key]) for n in names) + " |")
    lines.append("| 文档内配对：S1 / H(stack) | " + " | ".join(
        f"{r[n]['hard_reference']['within_doc_pairwise']['S1']:.3f} / {r[n]['hard_reference']['within_doc_pairwise']['H(stack)']:.3f}"
        for n in names) + " |")
    lines += ["", "## 7.2 同源偏差：各单一证据下每个系统的平均秩残差（正 = 被高估）", ""]
    for n in names:
        fam = r[n]["same_family"]
        lines += [f"### {CORPUS_LABEL[n]}", "", "| 证据 | " + " | ".join(SYSTEMS) + " | 语料级排名 | 与各系统错误重叠率 |",
                  "|---|" + "---:|" * len(SYSTEMS) + "---|---|"]
        for e in SINGLE_EVIDENCE:
            v = fam["per_evidence"][e]
            overlap = ", ".join(f"{s} {v['error_overlap'][s]:.2f}" for s in SYSTEMS)
            lines.append(f"| {e} | " + " | ".join(f"{v['rank_bias'][s]:+.3f}" for s in SYSTEMS)
                         + f" | {' > '.join(v['corpus_ranking'])} | {overlap} |")
        lines += ["", f"SaT 残差 E1 − E0：{fam['sat_bias_E1_minus_E0']:+.3f}；E1sm − E0：{fam['sat_bias_E1sm_minus_E0']:+.3f}；"
                  f"Stanza 残差 E2 − E0：{fam['stanza_bias_E2_minus_E0']:+.3f}"
                  + (f"；判定（> {FAMILY_MARGIN}）：{fam['flags']}" if "flags" in fam else "（GUM 被污染，只作参考）"), ""]
    lines += ["## 7.3 小样本敏感性", "", "| 配对差 [CI] | 语料 | 分层 bootstrap | 不分层 bootstrap |", "|---|---|---:|---:|"]
    for n, label in (("gum", "GUM 开发"), ("gentle", "GENTLE"), ("lines", "LinES"), ("pooled_test", "GENTLE + LinES")):
        sens = r[n]["sensitivity"]
        for key, kl in (("S1_minus_S0", "S1 − S0"), ("M3_minus_M0", "M3 − M0"), ("S1_minus_M0", "S1 − M0")):
            lines.append(f"| {kl} | {label} | {_ci(sens['stratified'][key])} | {_ci(sens['unstratified'][key])} |")
    logo = r["gentle"]["sensitivity"]["leave_one_genre_out"]
    lines += ["", "GENTLE 逐一去掉一个领域（点估计）：", "", "| 去掉的领域 | S1 − S0 | M3 − M0 | S1 | M0 |", "|---|---:|---:|---:|---:|"]
    for g, v in logo.items():
        if g != "all_positive":
            lines.append(f"| {g} | {v['S1_minus_S0']:+.3f} | {v['M3_minus_M0']:+.3f} | {v['S1']:.3f} | {v['M0']:.3f} |")
    lines += ["", f"全部为正：{logo['all_positive']}；GENTLE 结论按 7.3 规则“稳健”：**{r['gentle_robust']}**", ""]
    return "\n".join(lines)


def render_seeds(r: dict) -> str:
    lines = ["# 修订 7.4：微调种子方差（GUM train LOGO）", "",
             f"开发选型对种子稳定：**{r['selection_stable_over_seeds']}**", "",
             "| k（折种子基数） | 选中证据 | E2 | stack | stack − E0 [CI] | S1 − S0 [CI] | M3 − M0 [CI] |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for k, v in r["runs"].items():
        e = v["evidence"]
        lines.append(f"| {k}（{v['fold_seed_base']}） | {v['selected']} | {e['E2']['global_spearman']:.3f} | "
                     f"{e['stack']['global_spearman']:.3f} | {_ci(e['stack']['diff_vs_E0'])} | {_ci(v['S1_minus_S0'])} | "
                     f"{_ci(v['M3_minus_M0'])} |")
    lines += ["", "| 量 | 均值 | 标准差 | 最小 | 最大 |", "|---|---:|---:|---:|---:|"]
    for key, v in r["summary"].items():
        lines.append(f"| {key} | {v['mean']:.3f} | {v['sd']:.3f} | {v['min']:.3f} | {v['max']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["posthoc", "seeds"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    {"posthoc": posthoc, "seeds": seeds}[args.stage](args)


if __name__ == "__main__":
    main()
