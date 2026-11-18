"""Between- vs within-document decomposition of cross-document agreement (amendment 5, post-hoc (7)).

Tests the explanation that agreement-based baselines win the global Spearman by detecting document
difficulty: for each scorer, Spearman of document means (score vs Gold F1) and Spearman of the
document-centred residuals, with stratified paired bootstrap CIs (same resampling as
``src.quality_agreement``), plus the share of Gold-F1 variance that lies between documents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .baselines import CORPORA, OURS, PLATT, corpus_scores, gum_median_sentence_length
from .quality_agreement import _resample_plan


ROOT = Path(__file__).resolve().parents[1]
SCORERS = (OURS, PLATT, "B2 consensus (leave-one-out)", "ours + consensus (rank average)",
           "B3a punctuation-rule reference", "B1 pseudo-reference = sat")


def decompose(score: np.ndarray, gold: np.ndarray, keys: np.ndarray) -> dict[str, float]:
    _, group = np.unique(keys, return_inverse=True)
    counts = np.bincount(group)
    mean_s = np.bincount(group, weights=score) / counts
    mean_g = np.bincount(group, weights=gold) / counts
    return {"global": float(spearmanr(score, gold).statistic),
            "between_documents": float(spearmanr(mean_s, mean_g).statistic),
            "within_documents": float(spearmanr(score - mean_s[group], gold - mean_g[group]).statistic)}


def gold_variance_share(gold: np.ndarray, keys: np.ndarray) -> float:
    _, group = np.unique(keys, return_inverse=True)
    mean_g = np.bincount(group, weights=gold) / np.bincount(group)
    total = float(((gold - gold.mean()) ** 2).sum())
    return float(((mean_g[group] - gold.mean()) ** 2).sum() / total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/baselines")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    seed = int(json.loads((ROOT / "configs/real_splitters.json").read_text())["fixed_seed"])
    length0 = gum_median_sentence_length()
    report, lines = {}, ["# 跨文档一致性分解：文档之间 vs 文档之内（修订 5 补充 (7)，事后分析）", "",
                         "文档之间 = 文档平均分与文档平均 Gold F1 的 Spearman；文档之内 = 减去文档均值后的 Spearman。"
                         f"CI：领域内分层按文档重采样 {args.bootstrap} 次；配对差 = 评分 − 主方法。"
                         "B1 = SaT 含 SaT 自身条目（恒为 1）。", ""]
    for name, spec in CORPORA.items():
        items, _ = corpus_scores(name, spec, length0)
        point = {n: decompose(items.scores[n], items.gold, items.doc) for n in SCORERS}
        draws = {n: {k: [] for k in point[n]} for n in SCORERS}
        share_draws = []
        for idx, keys in _resample_plan(items, args.bootstrap, seed):
            share_draws.append(gold_variance_share(items.gold[idx], keys))
            for n in SCORERS:
                for k, v in decompose(items.scores[n][idx], items.gold[idx], keys).items():
                    draws[n][k].append(v)
        ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        entry = {"gold_variance_between_documents": gold_variance_share(items.gold, items.doc),
                 "gold_variance_between_documents_ci": ci(share_draws), "point": point,
                 "ci": {n: {k: ci(v) for k, v in d.items()} for n, d in draws.items()},
                 "paired_vs_ours": {n: {k: ci(np.array(draws[n][k]) - np.array(draws[OURS][k])) for k in point[n]}
                                    for n in SCORERS if n != OURS}}
        report[name] = entry
        lines += [f"## {spec['label']}", "",
                  f"Gold F1 方差中文档间差异占 **{entry['gold_variance_between_documents']:.1%}** "
                  f"[{entry['gold_variance_between_documents_ci'][0]:.1%}, {entry['gold_variance_between_documents_ci'][1]:.1%}]。", "",
                  "| 评分 | 全局 | 文档之间 | 文档之内 | 文档之间 差 vs 主方法 | 文档之内 差 vs 主方法 |", "|---|---:|---:|---:|---:|---:|"]
        for n in SCORERS:
            p = point[n]
            if n == OURS:
                diffs = ["—", "—"]
            else:
                diffs = [f"{p[k] - point[OURS][k]:+.3f} [{entry['paired_vs_ours'][n][k][0]:+.3f}, {entry['paired_vs_ours'][n][k][1]:+.3f}]"
                         for k in ("between_documents", "within_documents")]
            lines.append(f"| {n} | {p['global']:.3f} | {p['between_documents']:.3f} | {p['within_documents']:.3f} | "
                         f"{diffs[0]} | {diffs[1]} |")
        lines.append("")
        print(json.dumps({"status": "DECOMPOSITION", "corpus": name}), flush=True)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "decomposition.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.results / "decomposition.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "DECOMPOSITION_COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
