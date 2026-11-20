"""Amendment 13 addendum (post-hoc): a controlled soft-vs-hard comparison.

Same calibrated boundary probabilities, same document-end convention; only the scoring differs:
  soft  Q_end-normalized Soft F1 against p                                   (B5 for SaT, ours for ours)
  hard  boundary F1 with the document end counted, against h = [p >= 0.5]
For SaT also its own segmentation (raw probability > 0.25, B4) with the same document-end convention.
Writes results/a13/softhard.json and softhard.md.
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from .a6_combine import FINAL
from .a6_data import logo_folds
from .a9_layout import bootstrap, contrast
from .a11_analysis import soft_f1_end
from .a13_baselines import LABEL, NAMES, RESULTS, SAT_THRESHOLD, _ci, load_corpus, sat_platt
from .quality_agreement import tie_exact


def hard_end(rows: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Boundary F1 with the document end counted for both sides: 2(|y & r| + 1) / (|y| + |r| + 2)."""
    rows = np.atleast_2d(rows).astype(float)
    ref = np.asarray(ref, float)
    return 2 * (rows @ ref + 1) / (rows.sum(1) + ref.sum() + 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="*", default=[n for n in NAMES if n != "gum"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    with FINAL.open("rb") as handle:
        pickle.load(handle)
    gum = load_corpus("gum")
    gum_fold = logo_folds(gum.docs)
    out = {}
    for name in args.corpora:
        c = load_corpus(name)
        c.sat_p = sat_platt(gum, c, gum_fold)
        res = {}
        for scen in c.scen:
            lay = scen == "nl"
            sat_p = np.where(c.par, 1.0, c.sat_p) if lay else c.sat_p
            sat_own = (c.sat_raw > SAT_THRESHOLD) | (c.par if lay else False)
            ours = c.ev["final_layout" if lay else "final_flat"]
            cm = c.scen[scen]["cmat"]
            scorers = {
                "SaT probabilities, soft (B5)": lambda r, sl: soft_f1_end(r, sat_p[sl]),
                "SaT probabilities, hard (p >= 0.5)": lambda r, sl: hard_end(r, sat_p[sl] >= 0.5),
                "SaT own segmentation, hard (B4, end counted)": lambda r, sl: hard_end(r, sat_own[sl]),
                "Ours, soft": lambda r, sl: soft_f1_end(r, ours[sl]),
                "Ours, hard (p >= 0.5)": lambda r, sl: hard_end(r, ours[sl] >= 0.5),
            }
            scores = {n: tie_exact(np.concatenate([f(cm[:, sl], sl) for sl in c.sl])) for n, f in scorers.items()}
            point, draws, names = bootstrap(scores, c.scen[scen]["f1"], c.genre, args.bootstrap)
            res[scen] = {"global_spearman": {n: [float(point[n][0]), float(np.percentile(draws[:, i, 0], 2.5)),
                                                 float(np.percentile(draws[:, i, 0], 97.5))] for i, n in enumerate(names)},
                         "soft_minus_hard_sat": contrast(draws, names, point, ["SaT probabilities, soft (B5)"], ["SaT probabilities, hard (p >= 0.5)"]),
                         "hard_calibrated_minus_own_sat": contrast(draws, names, point, ["SaT probabilities, hard (p >= 0.5)"],
                                                                   ["SaT own segmentation, hard (B4, end counted)"]),
                         "soft_minus_hard_ours": contrast(draws, names, point, ["Ours, soft"], ["Ours, hard (p >= 0.5)"])}
        out[name] = res
        print(json.dumps({"status": "A13C_DONE", "corpus": name}), flush=True)
    (RESULTS / "softhard.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    L = ["# 受控对照：同一组校准概率、同样的文档末尾约定，只比较软评分与硬伪参考（事后分析 P）", "",
         "| 语料 | 场景 | SaT 软（B5） | SaT 硬（p ≥ 0.5） | SaT 自身切分（B4，计末尾） | 我们 软 | 我们 硬 | SaT 软 − 硬 | SaT 硬（校准）− 自身切分 | 我们 软 − 硬 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, res in out.items():
        for scen, r in res.items():
            g = r["global_spearman"]
            L.append(f"| {LABEL[name]} | {'带版式' if scen == 'nl' else '无换行'} | " + " | ".join(f"{g[k][0]:.3f}" for k in g) + " | "
                     f"{_ci(r['soft_minus_hard_sat'], True)} | {_ci(r['hard_calibrated_minus_own_sat'], True)} | {_ci(r['soft_minus_hard_ours'], True)} |")
    (RESULTS / "softhard.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
