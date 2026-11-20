"""Code review 2026-11-20: compare the archived reports (results/review_fixes/before/) with the rerun ones.

Every numeric and non-numeric leaf is compared (erratum_ties.compare). Writes results/review_fixes/changes.{json,md}.
"""

from __future__ import annotations

import json

from .erratum_ties import compare
from .a6_data import ROOT

OUT = ROOT / "results/review_fixes"
REPORTS = ("a8/report.json", "a11/report.json", "a12/report.json", "a13/report.json", "a13/extra.json",
           "a13/classic.json", "a13/softhard.json")


def main() -> None:
    lines = ["# 代码审查（2026-11-20）修复后重跑：与修复前报告的差异", "",
             "旧值 = `results/review_fixes/before/`；新值 = 重跑后的 `results/`。只重跑 CPU 分析阶段；冻结的模型、"
             "证据和一次性测试的冻结分数都没有改动。", ""]
    changes = {}
    for rel in REPORTS:
        old_path, new_path = OUT / "before" / rel, ROOT / "results" / rel
        if not old_path.exists() or not new_path.exists():
            lines += [f"## {rel}", "", "（缺少旧或新报告，跳过）", ""]
            continue
        c = compare(json.loads(old_path.read_text()), json.loads(new_path.read_text()))
        big = sorted(c["numeric"], key=lambda r: -abs(r[2] - r[1]))
        changes[rel] = {"numeric_changed": len(c["numeric"]), "max_abs_change": max((abs(y - x) for _, x, y in big), default=0.0),
                        "categorical": c["categorical"], "missing": c["missing"], "largest": big[:40]}
        lines += [f"## {rel}", "", f"数值变化 {len(c['numeric'])} 个（最大 |Δ| = {changes[rel]['max_abs_change']:.4f}）；"
                  f"非数值字段变化 {len(c['categorical'])} 个；路径不一致 {len(c['missing'])} 个。", ""]
        if c["categorical"]:
            lines += ["**非数值字段的变化**：", ""] + [f"- `{p}`：{x} → {y}" for p, x, y in c["categorical"][:30]] + [""]
        if c["missing"]:
            lines += ["路径不一致（前 10 个）：", ""] + [f"- `{p}`" for p in c["missing"][:10]] + [""]
        if big:
            lines += ["变化最大的 25 个：", "", "| 路径 | 旧 | 新 | Δ |", "|---|---:|---:|---:|"]
            lines += [f"| `{p}` | {x:.4f} | {y:.4f} | {y - x:+.4f} |" for p, x, y in big[:25]] + [""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n")
    (OUT / "changes.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: {"numeric": v["numeric_changed"], "max": round(v["max_abs_change"], 4),
                          "categorical": len(v["categorical"])} for k, v in changes.items()}, indent=1))


if __name__ == "__main__":
    main()
