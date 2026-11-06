# GUM Dev 真实 Sentence Splitter 排名实验

本报告仅在四个系统的预测与 BERT Soft F1 分数冻结后读取 GUM Dev Gold。Soft F1 公式未修改。

## 系统结果

非空白间隙边界（splitter 切在 surface token 内部，如邮件地址；或切在无空格的两个 token 之间，如 `al.|:`）不参与无参考 Soft F1，但在 Gold 元评测中计为该系统的 false positive。

| System | Gold Macro F1 | Gold Micro F1 | Mean Soft F1 | Non-gap FP |
|---|---:|---:|---:|---:|
| punkt | 0.9131 | 0.9310 | 0.7576 | 2 |
| sat | 0.9720 | 0.9742 | 0.7553 | 0 |
| stanza | 0.9561 | 0.9655 | 0.7656 | 0 |
| wtp | 0.9094 | 0.8992 | 0.7200 | 4 |

## 文档内排序

- 可计算 Spearman 的文档：29/30
- Mean/Median within-document Spearman：0.3110 / 0.4000
- Pairwise accuracy（Soft F1 tie 计 0.5）：0.6164，可判定 pairs=159
- Strict pairwise accuracy：0.6164
- 四系统输出完全相同的文档：1/30

### 各系统对

| Pair | Accuracy | Strict | Pairs | Ties |
|---|---:|---:|---:|---:|
| punkt--sat | 0.4138 | 0.4138 | 29 | 0 |
| punkt--stanza | 0.4444 | 0.4444 | 27 | 0 |
| punkt--wtp | 0.7586 | 0.7586 | 29 | 0 |
| sat--stanza | 0.4783 | 0.4783 | 23 | 0 |
| sat--wtp | 0.8800 | 0.8800 | 25 | 0 |
| stanza--wtp | 0.7308 | 0.7308 | 26 | 0 |

## Corpus-level 选择诊断

Soft F1 选择：**stanza**；Gold oracle：**sat**；selection regret=0.0158。

## 部署基线

| Policy | Gold Macro F1 | Gold Micro F1 |
|---|---:|---:|
| Soft F1 选择（stanza） | 0.9561 | 0.9655 |
| Always-sat | 0.9720 | 0.9742 |
| Direct Internal（calibrated p ≥ 0.5） | 0.8477 | 0.8317 |

Soft F1 选择相对 Always-sat：Macro -0.0158，Micro -0.0087（负值表示不如固定策略）。

## 解释限制

这是 GUM 内域真实系统实验。BERT 的 Platt calibration 使用 GUM Train Gold；Dev Gold 仅用于最终 meta-evaluation。协议版本：gum-real-splitters-v1.1（mask：surface_v1.1_terminal_unit_incl_reference_markers_else_insert_gap）。Direct Internal 阈值事先固定、未在任何 Gold 上调节。本轮未运行 EWT，也没有据此调整 Soft F1。
