"""Evaluate the boundary-probe evidence (PREREGISTRATION amendment 3) on GUM train.

For each cross-fitting scheme (LOGO = primary, P1 5 folds) and each scorer (normalized Soft F1
= primary, original Soft F1), every (variant, configuration) at the primary lambda is compared
with ``raw/platt`` by paired stratified bootstrap. The lambda-sensitivity runs get point
estimates only. Applies the pre-registered selection rule to the ``raw`` configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .boundary_probe import CONFIGS, PRIMARY_LAMBDA, SENSITIVITY_LAMBDAS, VARIANTS
from .quality_agreement import Items, evaluate, load_items, statistics


ROOT = Path(__file__).resolve().parents[1]
SCHEMES = ("logo", "fold5")
SCORERS = ("normalized_soft_f1", "soft_f1")
METRICS = (("global_spearman", "全局 Spearman"), ("large_gap_pairwise", "大差距配对"),
           ("cell_spearman", "领域×系统 Spearman"), ("within_doc_pairwise", "文档内配对"))
BASELINE = "raw/platt"
SIMPLICITY = ("platt", "lr", "probe", "probe+lr")


def merged_items(paths: dict[str, Path], scorer: str, args) -> Items:
    merged = None
    for name, path in paths.items():
        items, _ = load_items(path, args.config, args.data_root, args.cv_root)
        if merged is None:
            merged = Items(items.doc, items.domain, items.system, items.gold, {})
        elif not ((items.doc == merged.doc).all() and (items.system == merged.system).all()
                  and np.allclose(items.gold, merged.gold)):
            raise RuntimeError(f"Item order differs for {name}")
        merged.scores[name] = items.scores[scorer]
    return merged


def select(result: dict) -> dict:
    """Pre-registered rule on LOGO / normalized Soft F1, raw variant only."""
    effective = []
    for config in SIMPLICITY[1:]:
        name = f"raw/{config}"
        low = result["paired_vs_baseline"][name]["global_spearman"][0]
        if low > 0:
            effective.append((config, result["point"][name]["global_spearman"]))
    if not effective:
        return {"selected": "platt", "reason": "no method's paired 95% CI is above 0"}
    best = max(v for _, v in effective)
    for config, value in sorted(effective, key=lambda cv: SIMPLICITY.index(cv[0])):
        if best - value < 0.01:
            return {"selected": config, "reason": f"effective: {[c for c, _ in effective]}; "
                                                  f"simplest within 0.01 of the best ({best:.3f})"}
    raise AssertionError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_splitters.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--cv-root", type=Path, default=ROOT / "data/real_system/train_cv")
    parser.add_argument("--probe-root", type=Path, default=ROOT / "data/real_system/train_cv/boundary_probe")
    parser.add_argument("--results", type=Path, default=ROOT / "results/boundary_probe")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-log", type=Path, default=None)
    args = parser.parse_args()
    seed = int(json.loads(args.config.read_text())["fixed_seed"])
    run_info = json.loads((args.probe_root / "probe_run.json").read_text())

    def path(variant, config, lam, scheme):
        return args.probe_root / variant / f"{config}_lam{lam:g}" / f"{scheme}.jsonl"

    report = {"results": {}, "sensitivity": {}}
    for scheme in SCHEMES:
        primary = {f"{v}/{c}": path(v, c, lams[0], scheme) for v in VARIANTS for c, (_, lams) in CONFIGS.items()}
        for scorer in SCORERS:
            items = merged_items(primary, scorer, args)
            report["results"][f"{scheme}/{scorer}"] = evaluate(items, args.bootstrap, seed, BASELINE, args.workers)
            print(json.dumps({"status": "PROBE_EVAL", "scheme": scheme, "scorer": scorer}), flush=True)
        extra = {f"{v}/{c}/lam={lam:g}": path(v, c, lam, scheme)
                 for v in VARIANTS for c, (_, lams) in CONFIGS.items() for lam in lams[1:]}
        for scorer in SCORERS:
            report["sensitivity"][f"{scheme}/{scorer}"] = statistics(merged_items(extra, scorer, args))
    report["selection"] = select(report["results"]["logo/normalized_soft_f1"])
    report["diagnostics"] = run_info["diagnostics"]
    report["timings"] = run_info["timings"]
    report["corpus"] = run_info["corpus"]
    report["max_fit_gradient"] = max(f["max_grad"] for f in run_info["fits"])
    if args.gpu_log and args.gpu_log.exists():
        util = [float(line.split(",")[0]) for line in args.gpu_log.read_text().splitlines()
                if line.strip() and line.split(",")[0].strip().replace(".", "").isdigit()]
        busy = [u for u in util if u > 0]
        report["gpu_utilization"] = {"samples": len(util), "mean_all": float(np.mean(util)) if util else None,
                                     "mean_nonidle": float(np.mean(busy)) if busy else None,
                                     "share_samples_ge_90": float(np.mean([u >= 90 for u in util])) if util else None}

    lines = ["# 边界证据：探针 / 似然比 / 版面控制变量（GUM train，修订 3）", "",
             f"基线 `{BASELINE}`：原文上下文 + Platt（本作业内重算）。`para` = 段落分隔以 [SEP] 写入 BERT 输入（控制变量，"
             "只作分析）。配对差为相对基线的 95% CI（领域内分层按文档重采样 "
             f"{args.bootstrap} 次）。probe 类 λ = {PRIMARY_LAMBDA:g}。", ""]
    for scheme in SCHEMES:
        for scorer in SCORERS:
            result = report["results"][f"{scheme}/{scorer}"]
            tag = "（主）" if (scheme, scorer) == ("logo", "normalized_soft_f1") else ""
            lines += [f"## {scheme} · {scorer}{tag}", "",
                      "| 配置 | " + " | ".join(label for _, label in METRICS) + " | 全局 Spearman 配对差 |",
                      "|---|" + "---:|" * (len(METRICS) + 1)]
            for name, values in result["point"].items():
                diff = result["paired_vs_baseline"].get(name)
                cell = "—" if diff is None else (f"{values['global_spearman'] - result['point'][BASELINE]['global_spearman']:+.3f} "
                                                 f"[{diff['global_spearman'][0]:+.3f}, {diff['global_spearman'][1]:+.3f}]")
                lines.append(f"| {name} | " + " | ".join(f"{values[m]:.3f}" for m, _ in METRICS) + f" | {cell} |")
            lines.append("")
    lines += [f"**按预注册规则选中（raw，LOGO，归一化）：{report['selection']['selected']}**（{report['selection']['reason']}）", "",
              "## 诊断：校准后平均 p（LOGO，主 λ）", "",
              "| 配置 | 无终止标点 Gold | 其中 >0.5 | 无标点 Gold（非段落分隔） | 有终止标点 Gold | 有终止标点非 Gold | 无标点非 Gold | held-out BCE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        for c, (_, lams) in CONFIGS.items():
            d = report["diagnostics"][f"{v}/{c}/lam={lams[0]:g}/logo"]
            lines.append(f"| {v}/{c} | {d['mean_p_gold_no_terminal']:.3f} | {d['share_gold_no_terminal_above_half']:.1%} | "
                         f"{d['mean_p_gold_no_terminal_not_paragraph']:.3f} | {d['mean_p_gold_terminal']:.3f} | "
                         f"{d['mean_p_nongold_terminal']:.3f} | {d['mean_p_nongold_no_terminal']:.4f} | {d['heldout_bce']:.4f} |")
    lines += ["", "## λ 敏感性（只报点估计，LOGO · 归一化 · 全局 Spearman）", "", "| 配置 | 全局 Spearman | 大差距配对 |", "|---|---:|---:|"]
    for name, values in report["sensitivity"]["logo/normalized_soft_f1"].items():
        lines.append(f"| {name} | {values['global_spearman']:.3f} | {values['large_gap_pairwise']:.3f} |")
    c = report["corpus"]
    lines += ["", "## 语料与成本", "",
              f"- 空隙 {c['gaps']}，Gold {c['gold']}，其中无终止标点 {c['gold_no_terminal']}；段落分隔 {c['paragraph_breaks']}"
              f"（全是 Gold，占 Gold {c['gold_paragraph'] / c['gold']:.1%}，占无终止标点 Gold {c['gold_no_terminal_paragraph'] / c['gold_no_terminal']:.1%}）。",
              f"- 计时：`{json.dumps(report['timings'], ensure_ascii=False)}`",
              f"- 拟合最大梯度：{report['max_fit_gradient']:.2e}"]
    if "gpu_utilization" in report:
        g = report["gpu_utilization"]
        lines.append(f"- GPU 利用率（nvidia-smi 采样 {g['samples']} 次）：全程平均 {g['mean_all']:.1f}%，非空闲平均 "
                     f"{g['mean_nonidle']:.1f}%，≥90% 的采样占 {g['share_samples_ge_90']:.1%}。")
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.results / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "BOUNDARY_PROBE_EVAL_COMPLETE", "selected": report["selection"]["selected"]}), flush=True)


if __name__ == "__main__":
    main()
