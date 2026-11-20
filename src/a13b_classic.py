"""Amendment 13 addendum (post-hoc): two classic baselines added to the comparison.

  B8 GPT-2 perplexity   mean token log-likelihood of the text rendered one sentence per line (gen-PPL style)
  B9 Dawid-Skene        EM truth inference from the four segmenters' votes (one confusion matrix per system,
                        learned base rate, leave-one-system-out posterior) scored with Q_end

  python -m src.a13b_classic --bootstrap 1000      (CPU; downloads gpt2 once)
writes results/a13/classic.json and classic.md (V1 + V2 with every baseline; V3 detection rates).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings

import numpy as np

from .a6_combine import FINAL
from .a6_data import SYSTEMS, logo_folds, surfaces
from .a6_signals import dawid_skene
from .a8_supplement import credit, resample_plan
from .a9_layout import bootstrap, contrast
from .a11_analysis import LEVELS, TYPES2, candidates2, soft_f1_end
from .a12_test import TEST
from .a13_baselines import LABEL, METHODS, NAMES, RESULTS, SENS_TYPES, control_rows, item_scores, load_corpus, \
    make_scorers, priors_for, ranking, sat_platt, _ci
from .a9_casing import A9
from .quality_agreement import tie_exact

CLASSIC = ("GPT-2 perplexity", "Dawid-Skene")
ALL = METHODS[:-1] + CLASSIC + ("Ours",)


# ------------------------------------------------------------------ GPT-2 perplexity
class Perplexity:
    def __init__(self, threads: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        self.max_len, self.stride = 1024, 512

    def mean_loglik(self, text: str) -> float:
        torch = self.torch
        ids = self.tok(text, return_tensors="pt").input_ids[0]
        if len(ids) < 2:
            return 0.0
        nll, count, prev_end = 0.0, 0, 0
        for start in range(0, len(ids), self.stride):
            end = min(start + self.max_len, len(ids))
            target = end - prev_end                       # tokens not scored by an earlier window
            window = ids[start:end][None, :]
            labels = window.clone()
            labels[:, :-target] = -100
            with torch.inference_mode():
                out = self.model(window, labels=labels)
            n = int((labels[:, 1:] != -100).sum())
            nll += float(out.loss) * n
            count += n
            prev_end = end
            if end == len(ids):
                break
        return -nll / max(count, 1)


def render(text: str, offsets, row: np.ndarray) -> str:
    """One sentence per line: cut at the gaps marked in ``row`` (gap offset = the whitespace after a sentence)."""
    cuts = [o for o, b in zip(offsets, row) if b]
    pieces, start = [], 0
    for o in cuts:
        pieces.append(text[start:o].strip())
        start = o
    pieces.append(text[start:].strip())
    return "\n".join(p for p in pieces if p)


def ppl_scorer(ppl: Perplexity, texts: list[str], offsets: list[list[int]]):
    cache: dict[str, float] = {}

    def score(i: int, rows: np.ndarray) -> np.ndarray:
        out = []
        for r in np.atleast_2d(rows):
            t = render(texts[i], offsets[i], r)
            if t not in cache:
                cache[t] = ppl.mean_loglik(t)
            out.append(cache[t])
        return np.array(out)
    return score


# ------------------------------------------------------------------ Dawid-Skene
def ds_fit(c, scen: str) -> dict:
    votes = c.scen[scen]["cmat"].astype(float)
    return dawid_skene(votes, np.zeros(votes.shape[1], dtype=int), None)


def ds_item_scores(c, scen: str, fit: dict) -> np.ndarray:
    cm = c.scen[scen]["cmat"]
    out = []
    for sl in c.sl:
        for s in range(len(SYSTEMS)):
            out.append(soft_f1_end(cm[s:s + 1, sl], fit["leave_one_out"][s][sl])[0])
    return np.array(out)


# ------------------------------------------------------------------ V3 at one level (only the rows that are needed)
def sensitivity_k1(c, scorers: dict, n_boot: int) -> dict:
    lv = LEVELS.index("k=1")
    keep_types = [TYPES2.index(t) for t in SENS_TYPES[:-1]]
    names = list(scorers)
    det = np.full((len(c.docs), len(names), len(SENS_TYPES)), np.nan)
    for i, sl in enumerate(c.sl):
        g = c.y[sl]
        if not g.any() or (~g).sum() < 2:
            continue
        cand, t_idx, l_idx = candidates2(g, c.cats[sl], c.par[sl], i)
        sel = np.flatnonzero((l_idx == lv) & np.isin(t_idx, keep_types))
        rows = np.vstack([cand[:1], cand[sel]])
        types = np.concatenate([[-1], t_idx[sel]])
        base, moved = control_rows(g, i)
        for k, n in enumerate(names):
            s = tie_exact(scorers[n](i, rows))
            for j, t in enumerate(SENS_TYPES[:-1]):
                r = np.flatnonzero(types == TYPES2.index(t))
                if len(r):
                    det[i, k, j] = credit(s[0] - s[r]).mean()
            d = tie_exact(scorers[n](i, base)) - tie_exact(scorers[n](i, moved))
            det[i, k, -1] = credit(d).mean()
        if i % 50 == 0:
            print(json.dumps({"status": "A13B_SENS_PROGRESS", "corpus": c.name, "doc": i}), flush=True)
    plan = resample_plan(c.genre, n_boot)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, n in enumerate(names):
            point = np.nanmean(det[:, k, :], 0)
            draws = np.array([np.nanmean(det[idx, k, :], 0) for idx in plan])
            out[n] = {t: [float(point[j]), float(np.nanpercentile(draws[:, j], 2.5)), float(np.nanpercentile(draws[:, j], 97.5))]
                      for j, t in enumerate(SENS_TYPES)}
            out[n]["_detected"] = {t: bool(out[n][t][1] > 0.5) for t in SENS_TYPES[:-1]}
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="*", default=list(NAMES))
    parser.add_argument("--ds-sensitivity", nargs="*", default=["gum", "gentle", "ewtdev", "linesx"])
    parser.add_argument("--ppl-sensitivity", nargs="*", default=["ewtdev"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    ppl = Perplexity(args.threads)
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    gum = load_corpus("gum")
    gum_fold = logo_folds(gum.docs)
    report = {"note": "amendment 13 addendum (post-hoc): classic baselines B8 GPT-2 perplexity, B9 Dawid-Skene", "corpora": {}}
    for name in args.corpora:
        started = time.time()
        c = gum if name == "gum" else load_corpus(name)
        c.sat_p = sat_platt(gum, c, gum_fold)
        priors = priors_for(c, gum, final)
        offsets = [d.candidate_offsets for d in c.docs]
        out = {"scenarios": {}}
        for scen in c.scen:
            texts = [d.text for d in (c.docs if scen == "flat" else surfaces(TEST[f"{name}-nl"] if name in ("ewtdev",) else
                                                                                 _nl_corpus(name)))]
            scorers = make_scorers(c, scen, priors)
            scores = item_scores(c, scen, scorers)
            fit = ds_fit(c, scen)
            scores["Dawid-Skene"] = tie_exact(ds_item_scores(c, scen, fit))
            pscore = ppl_scorer(ppl, texts, offsets)
            cm = c.scen[scen]["cmat"]
            scores["GPT-2 perplexity"] = tie_exact(np.concatenate([pscore(i, cm[:, sl]) for i, sl in enumerate(c.sl)]))
            scores = {n: scores[n] for n in ALL}
            f1 = c.scen[scen]["f1"]
            point, draws, names = bootstrap(scores, f1, c.genre, args.bootstrap)
            table = {n: {"global_spearman": [float(point[n][0]), float(np.percentile(draws[:, i, 0], 2.5)), float(np.percentile(draws[:, i, 0], 97.5))],
                         "within_doc_pairwise": float(point[n][1])} for i, n in enumerate(names)}
            out["scenarios"][scen] = {"table": table,
                                      "ours_minus": {f"Ours - {b}": contrast(draws, names, point, ["Ours"], [b]) for b in CLASSIC},
                                      "ranking": ranking({n: scores[n] for n in CLASSIC + ("Ours",)}, f1, c.genre, args.bootstrap)}
            print(json.dumps({"status": "A13B_V1", "corpus": name, "scenario": scen, "seconds": round(time.time() - started, 1)}), flush=True)
        if name in args.ds_sensitivity or name in args.ppl_sensitivity:
            sc = make_scorers(c, "flat", priors)
            sc = {("Ours (flat)" if k == "Ours" else ("Consensus" if k == "Consensus (gap level)" else k)): v for k, v in sc.items()}
            fit = ds_fit(c, "flat")
            sc["Dawid-Skene"] = lambda i, r, fit=fit: soft_f1_end(r, fit["posterior"][c.sl[i]])
            if name in args.ppl_sensitivity:
                sc["GPT-2 perplexity"] = ppl_scorer(ppl, [d.text for d in c.docs], offsets)
            if "nl" in c.scen:
                sc["Ours (layout)"] = make_scorers(c, "nl", priors)["Ours"]
            out["sensitivity_k1"] = sensitivity_k1(c, sc, args.bootstrap)
        report["corpora"][name] = out
        print(json.dumps({"status": "A13B_CORPUS_DONE", "corpus": name, "seconds": round(time.time() - started, 1)}), flush=True)
        save(report)
    save(report)
    print(json.dumps({"status": "A13B_COMPLETE"}), flush=True)


def _nl_corpus(name: str):
    from .a9_layout import NL
    return NL[name]


def save(report: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "classic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    L = ["# 修订 13 补充基线（事后分析 P）：B8 GPT-2 困惑度、B9 Dawid–Skene", ""]
    for scen in ("flat", "nl"):
        cols = [n for n, c in report["corpora"].items() if scen in c["scenarios"]]
        if not cols:
            continue
        L += [f"## 全局 Spearman（{'带版式' if scen == 'nl' else '无换行'}；括号内为文档内配对准确率）", "",
              "| 方法 | " + " | ".join(LABEL[n] for n in cols) + " |", "|---|" + "---:|" * len(cols)]
        for m in ALL:
            L.append(f"| {m} | " + " | ".join(f"{report['corpora'][n]['scenarios'][scen]['table'][m]['global_spearman'][0]:.3f} "
                                             f"({report['corpora'][n]['scenarios'][scen]['table'][m]['within_doc_pairwise']:.2f})" for n in cols) + " |")
        L += ["", "Ours − 经典基线：" + "；".join(f"{LABEL[n]}: " + ", ".join(f"{k[7:]} {_ci(v, True)}" for k, v in
                                                                         report['corpora'][n]['scenarios'][scen]['ours_minus'].items()) for n in cols),
              "", "系统排名（领域平均 τ / 文档平均 τ）：" + "；".join(
                  f"{LABEL[n]}: " + ", ".join(f"{m} {v['genre_tau'][0]:.2f}/{v['doc_tau'][0]:.2f}" for m, v in
                                              report['corpora'][n]['scenarios'][scen]['ranking'].items()) for n in cols), ""]
    for n, c in report["corpora"].items():
        if "sensitivity_k1" not in c:
            continue
        s = c["sensitivity_k1"]
        L += [f"## 敏感度：注入 1 个错误的检出率（{LABEL[n]}；* = 95% CI 下界 > 0.5）", "", "| 方法 | " + " | ".join(SENS_TYPES) + " |",
              "|---|" + "---:|" * len(SENS_TYPES)]
        for m, d in s.items():
            L.append(f"| {m} | " + " | ".join(f"{d[t][0]:.2f}{'*' if d['_detected'].get(t) else ''}" for t in SENS_TYPES) + " |")
        L.append("")
    (RESULTS / "classic.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
