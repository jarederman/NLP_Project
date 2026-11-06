# GUM Dev 真实 Sentence Splitter 排名实验

本报告仅在四个系统的预测与 BERT Soft F1 分数冻结后读取 GUM Dev Gold。Soft F1 公式未修改。

## 系统结果

词内边界（splitter 在单个 surface token 内部切分，如邮件地址）不参与无参考 Soft F1，但在 Gold 元评测中计为该系统的 false positive。

| System | Gold Macro F1 | Gold Micro F1 | Mean Soft F1 | Intra-token FP |
|---|---:|---:|---:|---:|
| punkt | 0.9131 | 0.9310 | 0.7638 | 2 |
| sat | 0.9720 | 0.9742 | 0.7274 | 0 |
| stanza | 0.9561 | 0.9655 | 0.7355 | 0 |
| wtp | 0.9094 | 0.8992 | 0.7083 | 4 |

## 文档内排序

- 可计算 Spearman 的文档：29/30
- Mean/Median within-document Spearman：0.1455 / 0.3162
- Pairwise accuracy（Soft F1 tie 计 0.5）：0.5409，可判定 pairs=159
- Strict pairwise accuracy：0.5409
- 四系统输出完全相同的文档：1/30

### 各系统对

| Pair | Accuracy | Strict | Pairs | Ties |
|---|---:|---:|---:|---:|
| punkt--sat | 0.3448 | 0.3448 | 29 | 0 |
| punkt--stanza | 0.2963 | 0.2963 | 27 | 0 |
| punkt--wtp | 0.6552 | 0.6552 | 29 | 0 |
| sat--stanza | 0.4348 | 0.4348 | 23 | 0 |
| sat--wtp | 0.8000 | 0.8000 | 25 | 0 |
| stanza--wtp | 0.7308 | 0.7308 | 26 | 0 |

## Corpus-level 选择诊断

Soft F1 选择：**punkt**；Gold oracle：**sat**；selection regret=0.0589。

## 解释限制

这是 GUM 内域真实系统实验。BERT 的 Platt calibration 使用 GUM Train Gold；Dev Gold 仅用于最终 meta-evaluation。本轮没有运行 Always-SaT、Direct Internal Splitter 或 EWT，也没有据此调整 Soft F1。
