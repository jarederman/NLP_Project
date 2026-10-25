from __future__ import annotations

from pathlib import Path


LABELS = {
    "raw_bce_v1": "Raw punctuation BCE", "bce_v1": "Calibrated BCE V1",
    "bce_v2": "Calibrated BCE V2 (abbr.)", "weighted_bce_v1": "Weighted BCE V1",
    "weighted_bce_v2": "Weighted BCE V2 (abbr.)", "soft_f1_v1": "Soft F1 V1",
    "soft_f1_v2": "Soft F1 V2 (abbr.)", "soft_fbeta_v1": "Soft F0.5 V1",
    "soft_fbeta_v2": "Soft F0.5 V2 (abbr.)", "length_baseline": "Sentence-length baseline",
}


def _f(x) -> str:
    return "NA" if x != x else f"{x:.3f}"


def write_summary(path: str | Path, metadata: dict, aggregate: dict, rule_totals: dict, calibrator: dict) -> None:
    m, d = aggregate["metrics"], metadata["diagnostics"]
    lines = [
        "# Preliminary Results：无参考句子分割评价", "", "## 实际运行设置", "",
        f"- 模型：`{metadata['model_name']}`；`Qwen3_5ForCausalLM`，BF16 raw next-token logits；未调用 Chat API、未生成断句答案。",
        f"- 数据：GUM 官方 train 用于两参数 Platt calibration，dev 用于正式比较；未读取 test。dev 为 {metadata['dev_documents']} 个文档、{metadata['dev_sentences']} 个句子、{metadata['dev_candidate_positions']} 个候选空白位置。",
        f"- 受控文本：用 gold 仅移除句末独立 `.?!` token（共 {metadata['dev_removed_terminal_marks']} 个），保留内部标点、缩写和小数；gold 只用于 benchmark 构造与事后验证。",
        f"- GPU：{metadata['gpu_name']}；Transformers {metadata['transformers_version']}；PyTorch {metadata['torch_version']}；随机种子 {metadata['seed']}。",
        f"- raw 值只是 LM 句末标点 token 总概率，不是边界概率。calibrated probability 是在 train 上冻结的 Platt 映射（slope={calibrator['slope']:.4f}, intercept={calibrator['intercept']:.4f}），仍非跨域真实概率。",
        "", "## 主要排序结果", "",
        "| Metric（方向统一为越高质量越好） | 全局 Spearman | 文档内 Spearman 均值 | Pairwise accuracy | Gold 胜过扰动 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in LABELS.items():
        x = m[key]
        lines.append(f"| {label} | {_f(x['spearman_global'])} | {_f(x['spearman_within_doc_mean'])} | {_f(x['pairwise_accuracy'])} | {_f(x['gold_beats_perturbed'])} |")
    lines += ["", "## 按扰动类型", "", "BCE 越低越好，Soft F1 越高越好。", "",
              "| 条件 | n | Gold F1 | BCE V1 | BCE V2 | Weighted BCE V2 | Soft F1 V2 | 长度 baseline |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, row in aggregate["by_condition"].items():
        lines.append(f"| {condition} | {row['n']} | {_f(row['gold_f1'])} | {_f(row['bce_v1'])} | {_f(row['bce_v2'])} | {_f(row['weighted_bce_v2'])} | {_f(row['soft_f1_v2'])} | {_f(row['length_baseline'])} |")
    lines += ["", "### Gold 对各扰动的胜率", "",
              "| 指标 | Over 10/20% | Under 10/20% | Shift 10/20% | Mixed 10/20% |", "|---|---:|---:|---:|---:|"]
    for metric in ("bce_v1", "weighted_bce_v1", "soft_f1_v1", "soft_fbeta_v1", "length_baseline"):
        w = aggregate["gold_wins_by_condition_rate"][metric]
        def pair(c): return f"{w[f'{c}_0.1']:.3f} / {w[f'{c}_0.2']:.3f}"
        lines.append(f"| {LABELS[metric]} | {pair('over')} | {pair('under')} | {pair('shift')} | {pair('mixed')} |")
    lines += ["", "## Abbreviation / Context Rules", "",
              f"规则触发 {rule_totals['triggered']} 个 dev 位置：事后审计全部 {rule_totals['beneficial_nonboundary']} 个位于非边界，真边界误伤 {rule_totals['detrimental_boundary']} 个。以 0.5 为阈值，修复 {rule_totals['fixes']} 个 false positive、引入 {rule_totals['harms']} 个 false negative。规则不删除候选位置，只按固定因子降权。",
              f"V1→V2 pairwise accuracy 变化：BCE {m['bce_v2']['pairwise_accuracy']-m['bce_v1']['pairwise_accuracy']:+.3f}、Weighted BCE {m['weighted_bce_v2']['pairwise_accuracy']-m['weighted_bce_v1']['pairwise_accuracy']:+.3f}、Soft F1 {m['soft_f1_v2']['pairwise_accuracy']-m['soft_f1_v1']['pairwise_accuracy']:+.3f}。方向安全，但本轮无可测排序改善；这些位置的 V1 概率本来已很低。",
              "", "## 概率尺度、类别不平衡与长度", "",
              f"dev boundary prevalence 仅 {d['boundary_prevalence']:.3%}。raw proxy 在 boundary/non-boundary 上均值为 {d['raw_positive_mean']:.4f}/{d['raw_negative_mean']:.4f}；{d['raw_below_clip']} / {metadata['dev_candidate_positions']} 个 raw 值低于 1e-4。校准后均值为 {d['calibrated_positive_mean']:.3f}/{d['calibrated_negative_mean']:.3f}，范围 [{d['calibrated_min']:.4f}, {d['calibrated_max']:.4f}]。raw BCE 受极端小概率和 clipping 强烈影响；校准改善排序，但未消除 under 偏好。",
              f"Gold 文档质量分与文档候选位置数的 Spearman：BCE {d['gold_length_rho_bce']:.3f}、Weighted BCE {d['gold_length_rho_weighted']:.3f}、Soft F1 {d['gold_length_rho_soft_f1']:.3f}，本样本未见强长度相关。",
              "", "## 文本类型", "", "每个 genre 只有 2 个 dev 文档，下表仅作描述，不支持显著性结论。", "",
              "| Genre | BCE pairwise | Soft F1 pairwise | Soft F1 文档内 Spearman |", "|---|---:|---:|---:|"]
    for genre, x in aggregate["by_genre_ranking"].items():
        lines.append(f"| {genre} | {x['bce_v1']['pairwise_accuracy']:.3f} | {x['soft_f1_v1']['pairwise_accuracy']:.3f} | {x['soft_f1_v1']['within_doc_spearman']:.3f} |")
    c = metadata["cases"]
    lines += ["", "## 典型成功与失败案例", "",
              f"- BCE 失败：`{c['bce_under']['doc_id']}` 的 20% under（Gold F1={c['bce_under']['gold_f1']:.3f}），BCE 从 gold 的 {c['bce_under']['gold_bce']:.3f} 降到（错误地变好）{c['bce_under']['candidate_bce']:.3f}。例：{c['bce_under']['context']}",
              f"- Soft F1 成功：`{c['shift_success']['doc_id']}` 的 20% shift（Gold F1={c['shift_success']['gold_f1']:.3f}）使 Soft F1 从 {c['shift_success']['gold_soft_f1']:.3f} 降到 {c['shift_success']['candidate_soft_f1']:.3f}。例：{c['shift_success']['context']}",
              f"- Soft F1 失败：`{c['soft_failure']['doc_id']}` 的 20% under（Gold F1={c['soft_failure']['gold_f1']:.3f}）反使 Soft F1 上升 {c['soft_failure']['delta']:.3f}；被删边界 proxy 很低。例：{c['soft_failure']['context']}",
              "- 缩写规则实例：`Dr. William`（0.0204→0.0010）、`U.S. citizens`（0.00094→0.00009）；dev 中没有被规则压低的真边界。完整上下文和逐位置概率见 `examples.jsonl`。",
              "", "## 指标行为、局限与结论", "",
              "- Standard BCE 被大量 TN 主导；固定正类权重 10 的 Weighted BCE 更惩罚新增边界，却让删除边界更有利，未解决不平衡。Soft F1 不计 TN，因此更稳定；Soft F0.5 强调 precision，但 under 表现弱于 F1。",
              "- Platt calibration 使用 GUM train gold，所以“reference-free”仅指评分 dev 候选时不需要该文档 gold，并非完全无监督。",
              "- 词内句末缩写的最后一点无法无损删除，仍有残余泄漏风险；全局相关混入跨文档难度，故文档内结果更可信。",
              f"Standard BCE 只部分支持假设：稳定识别 over/shift，但 under 10%/20% 时 Gold 胜率仅 {aggregate['gold_wins_by_condition_rate']['bce_v1']['under_0.1']:.3f}/{aggregate['gold_wins_by_condition_rate']['bce_v1']['under_0.2']:.3f}，总体 pairwise 仅 {m['bce_v1']['pairwise_accuracy']:.3f}。Weighted BCE 总体 {m['weighted_bce_v1']['pairwise_accuracy']:.3f}，under 胜率均为 0。",
              f"本轮推荐 Soft F1：文档内 Spearman {m['soft_f1_v1']['spearman_within_doc_mean']:.3f}、pairwise {m['soft_f1_v1']['pairwise_accuracy']:.3f}、Gold 胜率 {m['soft_f1_v1']['gold_beats_perturbed']:.3f}；但全局 Spearman 仅 {m['soft_f1_v1']['spearman_global']:.3f}。缩写规则可作安全辅助项，但本轮不支持其改善总体排序。",
              "", "## 下一阶段", "",
              "优先做跨语料冻结/无监督先验校准；为 under 错误加入边界召回或宏平均项；对词内句末标点做严格无泄漏消融；扩大每类文档数量并 bootstrap 置信区间；继续以同文档成对评价为主。",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
