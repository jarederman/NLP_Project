# Preliminary Results：无参考句子分割评价

## 实际运行设置

- 模型：`Qwen/Qwen3.5-9B-Base`；`Qwen3_5ForCausalLM`，BF16 raw next-token logits；未调用 Chat API、未生成断句答案。
- 数据：GUM 官方 train 用于两参数 Platt calibration，dev 用于正式比较；未读取 test。dev 为 30 个文档、1575 个句子、24388 个候选空白位置。
- 受控文本：用 gold 仅移除句末独立 `.?!` token（共 1414 个），保留内部标点、缩写和小数；gold 只用于 benchmark 构造与事后验证。
- GPU：NVIDIA H200；Transformers 5.7.0；PyTorch 2.10.0+cu128；随机种子 20261025。
- raw 值只是 LM 句末标点 token 总概率，不是边界概率。calibrated probability 是在 train 上冻结的 Platt 映射（slope=0.5554, intercept=1.9808），仍非跨域真实概率。

## 主要排序结果

| Metric（方向统一为越高质量越好） | 全局 Spearman | 文档内 Spearman 均值 | Pairwise accuracy | Gold 胜过扰动 |
|---|---:|---:|---:|---:|
| Raw punctuation BCE | 0.035 | 0.098 | 0.539 | 0.733 |
| Calibrated BCE V1 | 0.078 | 0.200 | 0.572 | 0.754 |
| Calibrated BCE V2 (abbr.) | 0.078 | 0.200 | 0.572 | 0.754 |
| Weighted BCE V1 | 0.097 | 0.185 | 0.566 | 0.738 |
| Weighted BCE V2 (abbr.) | 0.097 | 0.185 | 0.566 | 0.738 |
| Soft F1 V1 | 0.211 | 0.800 | 0.844 | 0.950 |
| Soft F1 V2 (abbr.) | 0.211 | 0.800 | 0.844 | 0.950 |
| Soft F0.5 V1 | 0.183 | 0.628 | 0.748 | 0.925 |
| Soft F0.5 V2 (abbr.) | 0.183 | 0.628 | 0.748 | 0.925 |
| Sentence-length baseline | -0.017 | -0.041 | 0.490 | 0.713 |

## 按扰动类型

BCE 越低越好，Soft F1 越高越好。

| 条件 | n | Gold F1 | BCE V1 | BCE V2 | Weighted BCE V2 | Soft F1 V2 | 长度 baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| gold | 30 | 1.000 | 0.152 | 0.152 | 0.623 | 0.312 | -0.719 |
| mixed | 60 | 0.926 | 0.165 | 0.165 | 0.692 | 0.293 | -0.764 |
| over | 60 | 0.931 | 0.190 | 0.190 | 0.817 | 0.297 | -0.830 |
| shift | 60 | 0.852 | 0.177 | 0.177 | 0.759 | 0.273 | -0.726 |
| under | 60 | 0.918 | 0.142 | 0.142 | 0.569 | 0.285 | -0.729 |

### Gold 对各扰动的胜率

| 指标 | Over 10/20% | Under 10/20% | Shift 10/20% | Mixed 10/20% |
|---|---:|---:|---:|---:|
| Calibrated BCE V1 | 1.000 / 1.000 | 0.133 / 0.000 | 1.000 / 1.000 | 0.933 / 0.967 |
| Weighted BCE V1 | 1.000 / 1.000 | 0.000 / 0.000 | 1.000 / 1.000 | 0.933 / 0.967 |
| Soft F1 V1 | 0.933 / 0.933 | 0.867 / 0.967 | 1.000 / 1.000 | 0.933 / 0.967 |
| Soft F0.5 V1 | 1.000 / 1.000 | 0.700 / 0.800 | 1.000 / 1.000 | 0.933 / 0.967 |
| Sentence-length baseline | 0.967 / 0.967 | 0.567 / 0.567 | 0.633 / 0.633 | 0.567 / 0.800 |

## Abbreviation / Context Rules

规则触发 21 个 dev 位置：事后审计全部 21 个位于非边界，真边界误伤 0 个。以 0.5 为阈值，修复 0 个 false positive、引入 0 个 false negative。规则不删除候选位置，只按固定因子降权。
V1→V2 pairwise accuracy 变化：BCE +0.000、Weighted BCE +0.000、Soft F1 +0.000。方向安全，但本轮无可测排序改善；这些位置的 V1 概率本来已很低。

## 概率尺度、类别不平衡与长度

dev boundary prevalence 仅 6.335%。raw proxy 在 boundary/non-boundary 上均值为 0.0392/0.0020；17406 / 24388 个 raw 值低于 1e-4。校准后均值为 0.326/0.050，范围 [0.0009, 0.9659]。raw BCE 受极端小概率和 clipping 强烈影响；校准改善排序，但未消除 under 偏好。
Gold 文档质量分与文档候选位置数的 Spearman：BCE 0.059、Weighted BCE 0.136、Soft F1 0.156，本样本未见强长度相关。

## 文本类型

每个 genre 只有 2 个 dev 文档，下表仅作描述，不支持显著性结论。

| Genre | BCE pairwise | Soft F1 pairwise | Soft F1 文档内 Spearman |
|---|---:|---:|---:|
| academic | 0.600 | 0.743 | 0.674 |
| bio | 0.600 | 0.714 | 0.536 |
| conversation | 0.529 | 0.843 | 0.803 |
| court | 0.614 | 0.829 | 0.766 |
| essay | 0.557 | 0.914 | 0.895 |
| fiction | 0.600 | 0.900 | 0.895 |
| interview | 0.486 | 0.814 | 0.753 |
| letter | 0.557 | 0.857 | 0.791 |
| news | 0.571 | 0.871 | 0.837 |
| podcast | 0.529 | 0.871 | 0.874 |
| speech | 0.600 | 0.814 | 0.782 |
| textbook | 0.657 | 0.814 | 0.745 |
| vlog | 0.557 | 0.957 | 0.967 |
| voyage | 0.543 | 0.843 | 0.812 |
| whow | 0.586 | 0.871 | 0.870 |

## 典型成功与失败案例

- BCE 失败：`GUM_conversation_grounded` 的 20% under（Gold F1=0.888），BCE 从 gold 的 0.318 降到（错误地变好）0.266。例：“in it Did you just wash — Yes Really bothers me I kn- — | Oh oh I know I mean that 's the thing I thought of too”（p=0.0121）
- Soft F1 成功：`GUM_bio_emperor` 的 20% shift（Gold F1=0.811）使 Soft F1 从 0.324 降到 0.229。例：真边界 “retense by claiming to be "Protector of Mexico" as well | Born in England, Norton spent most of his early life in”（p=0.8763） 被移到非边界 “e inadequacies of the legal and political structures of | the United States On September 17, 1859, he took matter”（p=0.0017）
- Soft F1 失败：`GUM_bio_byron` 的 20% under（Gold F1=0.884）反使 Soft F1 上升 0.010；被删边界 proxy 很低。例：“ng) against convicted or even suspected offenders. [26] | The liaison, on the other hand, may well have been "pur”（p=0.0054）
- 缩写规则实例：`Dr. William`（0.0204→0.0010）、`U.S. citizens`（0.00094→0.00009）；dev 中没有被规则压低的真边界。完整上下文和逐位置概率见 `examples.jsonl`。

## 指标行为、局限与结论

- Standard BCE 被大量 TN 主导；固定正类权重 10 的 Weighted BCE 更惩罚新增边界，却让删除边界更有利，未解决不平衡。Soft F1 不计 TN，因此更稳定；Soft F0.5 强调 precision，但 under 表现弱于 F1。
- Platt calibration 使用 GUM train gold，所以“reference-free”仅指评分 dev 候选时不需要该文档 gold，并非完全无监督。
- 词内句末缩写的最后一点无法无损删除，仍有残余泄漏风险；全局相关混入跨文档难度，故文档内结果更可信。
Standard BCE 只部分支持假设：稳定识别 over/shift，但 under 10%/20% 时 Gold 胜率仅 0.133/0.000，总体 pairwise 仅 0.572。Weighted BCE 总体 0.566，under 胜率均为 0。
本轮推荐 Soft F1：文档内 Spearman 0.800、pairwise 0.844、Gold 胜率 0.950；但全局 Spearman 仅 0.211。缩写规则可作安全辅助项，但本轮不支持其改善总体排序。

## 下一阶段

优先做跨语料冻结/无监督先验校准；为 under 错误加入边界召回或宏平均项；对词内句末标点做严格无泄漏消融；扩大每类文档数量并 bootstrap 置信区间；继续以同文档成对评价为主。
