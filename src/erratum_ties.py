"""Erratum 2026-11-19: compare the untied (archived) and tie-exact reports of amendments 6-9.

Every numeric leaf whose path contains a rank statistic is compared; every non-numeric leaf
(selections, decisions, flags) must be unchanged or is listed. Writes results/erratum_ties/summary.md
and changes.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/erratum_ties"
REPORTS = ("a6/dev_report.json", "a6/test_report.json", "a7/posthoc_report.json", "a7/seeds_report.json",
           "a8/report.json", "a9/casing_report.json", "a9/layout_report.json")
RANK = re.compile(r"spearman|between_doc|within_doc|rank_bias|pairwise|kendall|robustness")
HEADLINE = re.compile(r"(global_spearman)(/0)?$")


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}/{i}")
    else:
        yield path, obj


def compare(old: dict, new: dict) -> dict:
    a, b = dict(leaves(old)), dict(leaves(new))
    num, cat, missing = [], [], sorted(set(a) ^ set(b))
    for p in sorted(set(a) & set(b)):
        x, y = a[p], b[p]
        if isinstance(x, bool) or isinstance(y, bool) or isinstance(x, str) or isinstance(y, str) or x is None or y is None:
            if x != y:
                cat.append((p, x, y))
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if x == x and y == y and abs(x - y) > 1e-12:
                num.append((p, float(x), float(y)))
    return {"numeric": num, "categorical": cat, "missing": missing}


def main() -> None:
    lines = ["# 勘误（2026-11-19）：平局精确化后修订 6–9 的变化", "",
             "旧值 = `results/erratum_ties/untied/`（浮点噪声决定并列分数的先后）；新值 = 现在的 `results/a6`…`a9`（分数四舍五入到 1e-10 后计算）。"
             "只重跑 CPU 分析阶段；冻结的组件、证据和测试推理都没有重跑。", ""]
    changes = {}
    for rel in REPORTS:
        old = json.loads((OUT / "untied" / rel).read_text())
        new = json.loads((ROOT / "results" / rel).read_text())
        c = compare(old, new)
        rank = [r for r in c["numeric"] if RANK.search(r[0])]
        other = [r for r in c["numeric"] if not RANK.search(r[0])]
        head = [r for r in rank if HEADLINE.search(r[0])]
        changes[rel] = {"rank_changed": len(rank), "other_numeric_changed": len(other), "categorical_changed": c["categorical"],
                        "missing": c["missing"], "max_abs_rank_change": max((abs(y - x) for _, x, y in rank), default=0.0),
                        "headline": head}
        lines += [f"## {rel}", "",
                  f"排序类数值变化 {len(rank)} 个，最大 |Δ| = {changes[rel]['max_abs_rank_change']:.4f}；其他数值变化 {len(other)} 个；"
                  f"非数值字段（选择、判定、标记）变化 {len(c['categorical'])} 个；路径不一致 {len(c['missing'])} 个。", ""]
        if c["categorical"]:
            lines += ["**非数值字段的变化**：", ""] + [f"- `{p}`：{x} → {y}" for p, x, y in c["categorical"]] + [""]
        if other:
            lines += ["其他数值变化（前 10 个）：", ""] + [f"- `{p}`：{x:.6g} → {y:.6g}" for p, x, y in other[:10]] + [""]
        big = sorted(head, key=lambda r: -abs(r[2] - r[1]))[:15]
        if big:
            lines += ["全局 Spearman 点估计变化最大的 15 个：", "", "| 路径 | 旧 | 新 | Δ |", "|---|---:|---:|---:|"]
            lines += [f"| `{p}` | {x:.4f} | {y:.4f} | {y - x:+.4f} |" for p, x, y in big] + [""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({rel: {k: (v if not isinstance(v, list) else len(v)) for k, v in c.items() if k != "headline"}
                      for rel, c in changes.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
