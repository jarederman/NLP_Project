# GUM Train 5-fold cross-fitting（P1 开发集）

协议：gum-real-splitters-v1.1（mask：surface_v1.1_terminal_unit_incl_reference_markers_else_insert_gap）；197 篇文档，15 个领域（genre）。每折的 Platt 只在其余各折的 Gold 上拟合；GUM dev/test 未读取。

置信区间：按领域分层、按文档重采样的 bootstrap，95% 百分位区间。

## 关键指标（Soft F1）

| 层次 | 指标 | 估计 [95% CI] |
|---|---|---|
| 前提 | Gold-win 率 | 0.1624 [0.1117, 0.2132] |
| L1 | Pairwise accuracy | 0.6292 [0.5961, 0.6618] |
| L2 | 领域平均 Kendall τ | 0.2889 [0.1556, 0.3778] |
| L3 | 领域平均 Selection Regret | 0.0277 [0.0226, 0.0381] |
| L3 | 领域平均（所选 − Always-sat） | -0.0263 [-0.0367, -0.0210] |
| 参考 | 语料级 Macro Regret | 0.0227 [0.0187, 0.0273] |

领域内选中 oracle 的比例：0.13；Always-sat 的领域平均 regret：0.0014。

## 系统（语料级）

| System | Gold Macro F1 | Gold Micro F1 | Mean Soft F1 |
|---|---:|---:|---:|
| punkt | 0.9142 | 0.9264 | 0.7437 |
| sat | 0.9714 | 0.9733 | 0.7403 |
| stanza | 0.9487 | 0.9566 | 0.7517 |
| wtp | 0.9021 | 0.8967 | 0.7048 |
| Direct Internal（p ≥ 0.5） | 0.8310 | 0.8181 | — |

## 按领域（Soft F1，Macro）

| Domain | Docs | Selected | Oracle | Regret | 所选 − fixed | Kendall τ | Direct Internal |
|---|---:|---|---|---:|---:|---:|---:|
| academic | 14 | punkt | sat | 0.0593 | -0.0593 | 0.00 | 0.8387 |
| bio | 16 | stanza | sat | 0.0442 | -0.0442 | 0.67 | 0.8603 |
| conversation | 11 | stanza | sat | 0.0051 | -0.0051 | 0.67 | 0.7036 |
| court | 10 | punkt | stanza | 0.0121 | -0.0087 | 0.00 | 0.7795 |
| essay | 10 | punkt | sat | 0.0580 | -0.0580 | 0.00 | 0.8580 |
| fiction | 15 | punkt | sat | 0.0315 | -0.0315 | 0.33 | 0.8670 |
| interview | 15 | punkt | sat | 0.0387 | -0.0387 | 0.00 | 0.8399 |
| letter | 14 | punkt | sat | 0.0508 | -0.0508 | -0.33 | 0.7700 |
| news | 20 | sat | sat | 0.0000 | +0.0000 | 0.67 | 0.8777 |
| podcast | 10 | punkt | stanza | 0.0104 | +0.0063 | 0.67 | 0.8022 |
| speech | 11 | punkt | sat | 0.0156 | -0.0156 | 0.00 | 0.8984 |
| textbook | 11 | sat | sat | 0.0000 | +0.0000 | 0.67 | 0.8711 |
| vlog | 11 | sat | stanza | 0.0009 | +0.0000 | 0.67 | 0.7794 |
| voyage | 14 | punkt | sat | 0.0574 | -0.0574 | 0.00 | 0.8244 |
| whow | 15 | stanza | sat | 0.0315 | -0.0315 | 0.33 | 0.8360 |

## 评分函数对照（辅助）

| Scorer | Pairwise acc | 领域平均 Regret | 领域平均（所选 − fixed） | 领域平均 τ |
|---|---|---|---|---|
| soft_f1 | 0.6292 [0.5961, 0.6618] | 0.0277 [0.0226, 0.0381] | -0.0263 [-0.0367, -0.0210] | 0.2889 [0.1556, 0.3778] |
| neg_bce | 0.5596 [0.5259, 0.5908] | 0.0391 [0.0323, 0.0443] | -0.0377 [-0.0428, -0.0307] | 0.0444 [-0.0222, 0.1556] |
| normalized_soft_f1 | 0.6292 [0.5961, 0.6618] | 0.0257 [0.0225, 0.0379] | -0.0243 [-0.0367, -0.0209] | 0.3111 [0.1556, 0.3778] |
