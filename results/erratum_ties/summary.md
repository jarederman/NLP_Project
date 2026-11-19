# 勘误（2026-11-19）：平局精确化后修订 6–9 的变化

旧值 = `results/erratum_ties/untied/`（浮点噪声决定并列分数的先后）；新值 = 现在的 `results/a6`…`a9`（分数四舍五入到 1e-10 后计算）。只重跑 CPU 分析阶段；冻结的组件、证据和测试推理都没有重跑。

## a6/dev_report.json

排序类数值变化 188 个，最大 |Δ| = 0.0009；其他数值变化 267 个；非数值字段（选择、判定、标记）变化 0 个；路径不一致 8 个。

其他数值变化（前 10 个）：

- `/M/ci/M0/global_pearson/0`：0.798256 → 0.798256
- `/M/ci/M0/global_pearson/1`：0.863797 → 0.863797
- `/M/ci/M1/global_pearson/0`：-0.124139 → -0.124139
- `/M/ci/M1/global_pearson/1`：-0.0175645 → -0.0175645
- `/M/ci/M1s/global_pearson/0`：0.852196 → 0.852196
- `/M/ci/M1s/global_pearson/1`：0.914552 → 0.914552
- `/M/ci/M2/global_pearson/0`：0.872682 → 0.872682
- `/M/ci/M2/global_pearson/1`：0.931294 → 0.931294
- `/M/ci/M3/global_pearson/0`：0.911379 → 0.911379
- `/M/ci/M3/global_pearson/1`：0.948414 → 0.948414

全局 Spearman 点估计变化最大的 15 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/evidence/ci/ev:E1sm/global_spearman/0` | 0.8786 | 0.8795 | +0.0009 |
| `/evidence/ci/ev:stack+hsmm/global_spearman/0` | 0.7669 | 0.7678 | +0.0009 |
| `/evidence/point/ev:stack+hsmm/global_spearman` | 0.8065 | 0.8072 | +0.0007 |
| `/S/paired_vs_baseline/S1/global_spearman/0` | 0.1859 | 0.1865 | +0.0006 |
| `/evidence/paired_vs_baseline/ev:stack/global_spearman/0` | 0.1859 | 0.1865 | +0.0006 |
| `/evidence/ci/ev:E2/global_spearman/0` | 0.7385 | 0.7379 | -0.0006 |
| `/evidence/point/ev:E1sm/global_spearman` | 0.9022 | 0.9026 | +0.0004 |
| `/evidence/point/ev:E2/global_spearman` | 0.7814 | 0.7812 | -0.0002 |
| `/evidence/paired_vs_baseline/ev:E2+hsmm/global_spearman/0` | 0.1378 | 0.1376 | -0.0002 |
| `/evidence/ci/ev:E2b/global_spearman/0` | 0.5925 | 0.5927 | +0.0002 |
| `/evidence/paired_vs_baseline/ev:stack+hsmm/global_spearman/0` | 0.1735 | 0.1736 | +0.0002 |
| `/evidence/paired_vs_baseline/ev:E1sm/global_spearman/0` | 0.2703 | 0.2705 | +0.0002 |
| `/evidence/paired_vs_baseline/ev:E2/global_spearman/0` | 0.1495 | 0.1494 | -0.0001 |
| `/evidence/paired_vs_baseline/ev:E2b/global_spearman/0` | 0.0150 | 0.0151 | +0.0001 |
| `/S/point/S1/global_spearman` | 0.8180 | 0.8181 | +0.0001 |

## a6/test_report.json

排序类数值变化 139 个，最大 |Δ| = 0.0239；其他数值变化 236 个；非数值字段（选择、判定、标记）变化 0 个；路径不一致 0 个。

其他数值变化（前 10 个）：

- `/corpora/gentle/M/ci/M0/global_pearson/0`：0.678173 → 0.678173
- `/corpora/gentle/M/ci/M0/global_pearson/1`：0.811693 → 0.811693
- `/corpora/gentle/M/ci/M1/global_pearson/0`：-0.292955 → -0.292955
- `/corpora/gentle/M/ci/M1/global_pearson/1`：-0.0117073 → -0.0117073
- `/corpora/gentle/M/ci/M1s/global_pearson/0`：0.739245 → 0.739245
- `/corpora/gentle/M/ci/M1s/global_pearson/1`：0.849979 → 0.849979
- `/corpora/gentle/M/ci/M2/global_pearson/0`：0.804862 → 0.804862
- `/corpora/gentle/M/ci/M2/global_pearson/1`：0.941826 → 0.941826
- `/corpora/gentle/M/ci/M3/global_pearson/0`：0.84589 → 0.84589
- `/corpora/gentle/M/paired_vs_baseline/M1/global_pearson/0`：-1.0946 → -1.0946

全局 Spearman 点估计变化最大的 15 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/corpora/lines/evidence/ci/ev:E1sm/global_spearman/0` | 0.5638 | 0.5399 | -0.0239 |
| `/corpora/lines/evidence/ci/ev:stack+hsmm/global_spearman/0` | 0.5192 | 0.5027 | -0.0165 |
| `/corpora/lines/evidence/paired_vs_baseline/ev:E1sm/global_spearman/0` | -0.1827 | -0.1912 | -0.0085 |
| `/corpora/lines/S/ci/S1/global_spearman/0` | 0.5328 | 0.5403 | +0.0075 |
| `/corpora/lines/evidence/ci/ev:stack/global_spearman/0` | 0.5328 | 0.5403 | +0.0075 |
| `/corpora/lines/evidence/point/ev:E1sm/global_spearman` | 0.8378 | 0.8309 | -0.0069 |
| `/corpora/gentle/evidence/paired_vs_baseline/ev:E2/global_spearman/0` | -0.0278 | -0.0321 | -0.0043 |
| `/corpora/lines/evidence/point/ev:stack+hsmm/global_spearman` | 0.8023 | 0.7989 | -0.0035 |
| `/corpora/gentle/evidence/paired_vs_baseline/ev:E2+hsmm/global_spearman/0` | -0.0280 | -0.0308 | -0.0028 |
| `/corpora/lines/evidence/point/ev:E2b/global_spearman` | 0.8154 | 0.8128 | -0.0025 |
| `/corpora/lines/S/point/S1/global_spearman` | 0.8154 | 0.8177 | +0.0023 |
| `/corpora/lines/evidence/point/ev:stack/global_spearman` | 0.8154 | 0.8177 | +0.0023 |
| `/corpora/gentle/evidence/ci/ev:E2b/global_spearman/0` | 0.6004 | 0.5991 | -0.0013 |
| `/corpora/gentle/evidence/paired_vs_baseline/ev:E2b/global_spearman/0` | -0.0424 | -0.0435 | -0.0011 |
| `/corpora/gentle/evidence/point/ev:E2+hsmm/global_spearman` | 0.7405 | 0.7396 | -0.0009 |

## a7/posthoc_report.json

排序类数值变化 34 个，最大 |Δ| = 0.0062；其他数值变化 44 个；非数值字段（选择、判定、标记）变化 0 个；路径不一致 0 个。

其他数值变化（前 10 个）：

- `/gum/hard_reference/S1_minus_H(stack)/ci/0`：-0.00819805 → -0.00826374
- `/gum/hard_reference/S1_minus_H(stack)/ci/1`：0.0259488 → 0.0259422
- `/gum/hard_reference/S1_minus_H(stack)/diff`：0.00830724 → 0.00843531
- `/gum/same_family/sat_bias_E1_minus_E0`：-0.0568554 → -0.0570003
- `/gum/same_family/sat_bias_E1sm_minus_E0`：0.0637578 → 0.0645952
- `/gum/same_family/stanza_bias_E2_minus_E0`：0.0378907 → 0.0380163
- `/gum/sensitivity/stratified/S1_minus_M0/ci/0`：0.0424776 → 0.0425961
- `/gum/sensitivity/stratified/S1_minus_M0/ci/1`：0.127966 → 0.127827
- `/gum/sensitivity/stratified/S1_minus_M0/diff`：0.0846961 → 0.0848242
- `/gum/sensitivity/stratified/S1_minus_S0/ci/0`：0.185931 → 0.186543

## a7/seeds_report.json

排序类数值变化 28 个，最大 |Δ| = 0.0007；其他数值变化 112 个；非数值字段（选择、判定、标记）变化 0 个；路径不一致 0 个。

其他数值变化（前 10 个）：

- `/runs/0/S1`：0.818009 → 0.818137
- `/runs/0/S1_minus_S0/ci/0`：0.185931 → 0.186543
- `/runs/0/S1_minus_S0/ci/1`：0.309307 → 0.308856
- `/runs/0/S1_minus_S0/diff`：0.244176 → 0.244304
- `/runs/0/evidence/E0+hsmm/diff_vs_E0/ci/0`：-0.0145556 → -0.0145565
- `/runs/0/evidence/E0+hsmm/diff_vs_E0/ci/1`：0.00233003 → 0.00227965
- `/runs/0/evidence/E0+hsmm/diff_vs_E0/diff`：-0.00595786 → -0.00600712
- `/runs/0/evidence/E1+hsmm/diff_vs_E0/ci/0`：-0.0217008 → -0.0217676
- `/runs/0/evidence/E1+hsmm/diff_vs_E0/ci/1`：0.14824 → 0.148143
- `/runs/0/evidence/E1+hsmm/diff_vs_E0/diff`：0.0641702 → 0.0641655

全局 Spearman 点估计变化最大的 15 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/runs/0/evidence/stack+hsmm/global_spearman` | 0.8065 | 0.8072 | +0.0007 |
| `/runs/1/evidence/stack+hsmm/global_spearman` | 0.7933 | 0.7938 | +0.0005 |
| `/runs/2/evidence/stack+hsmm/global_spearman` | 0.7811 | 0.7816 | +0.0005 |
| `/runs/3/evidence/stack/global_spearman` | 0.7980 | 0.7976 | -0.0004 |
| `/runs/3/evidence/stack+hsmm/global_spearman` | 0.7886 | 0.7883 | -0.0003 |
| `/runs/2/evidence/E2/global_spearman` | 0.6358 | 0.6361 | +0.0003 |
| `/runs/2/evidence/stack/global_spearman` | 0.7941 | 0.7939 | -0.0002 |
| `/runs/0/evidence/E2/global_spearman` | 0.7814 | 0.7812 | -0.0002 |
| `/runs/1/evidence/stack/global_spearman` | 0.8045 | 0.8043 | -0.0002 |
| `/runs/1/evidence/E2/global_spearman` | 0.5938 | 0.5940 | +0.0002 |
| `/runs/3/evidence/E2+hsmm/global_spearman` | 0.6170 | 0.6168 | -0.0002 |
| `/runs/2/evidence/E2+hsmm/global_spearman` | 0.6287 | 0.6285 | -0.0002 |
| `/runs/1/evidence/E2+hsmm/global_spearman` | 0.5832 | 0.5830 | -0.0001 |
| `/runs/0/evidence/stack/global_spearman` | 0.8180 | 0.8181 | +0.0001 |
| `/runs/0/evidence/E0+hsmm/global_spearman` | 0.5679 | 0.5678 | -0.0000 |

## a8/report.json

排序类数值变化 83 个，最大 |Δ| = 0.0460；其他数值变化 49 个；非数值字段（选择、判定、标记）变化 0 个；路径不一致 0 个。

其他数值变化（前 10 个）：

- `/corpora/gentle/ablation/variants/stack-E1/bias_change/punkt`：0.049926 → 0.0497411
- `/corpora/gentle/ablation/variants/stack-E1/bias_change/stanza`：-0.00406805 → -0.00388314
- `/corpora/gentle/ablation/variants/stack-E1/minus_stack/0`：-0.0704071 → -0.0696695
- `/corpora/gentle/ablation/variants/stack-E1/minus_stack/1`：-0.15613 → -0.152178
- `/corpora/gentle/ablation/variants/stack-E2/minus_stack/0`：-0.0441578 → -0.0434782
- `/corpora/gentle/ablation/variants/stack-E2/minus_stack/1`：-0.077889 → -0.0776326
- `/corpora/gentle/ablation/variants/stack-E2/minus_stack/2`：-0.0162604 → -0.0156876
- `/corpora/gum/ablation/variants/stack-E1/bias_change/punkt`：0.0450959 → 0.0458141
- `/corpora/gum/ablation/variants/stack-E1/bias_change/sat`：-0.0563304 → -0.0568264
- `/corpora/gum/ablation/variants/stack-E1/bias_change/stanza`：0.0316615 → 0.0315809

全局 Spearman 点估计变化最大的 15 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/corpora/lines/ablation/variants/stack-E2/global_spearman/0` | 0.7748 | 0.7850 | +0.0102 |
| `/corpora/lines/ablation/variants/stack/global_spearman/0` | 0.8212 | 0.8177 | -0.0034 |
| `/corpora/lines/ablation/variants/stack-E1/global_spearman/0` | 0.8290 | 0.8319 | +0.0029 |
| `/corpora/lines/baselines/S1/global_spearman/0` | 0.8154 | 0.8177 | +0.0023 |
| `/corpora/gum/length/medium/S1_global_spearman` | 0.7801 | 0.7810 | +0.0009 |
| `/corpora/gentle/ablation/variants/stack-E1/global_spearman/0` | 0.8127 | 0.8134 | +0.0007 |
| `/corpora/gentle/ablation/variants/stack-E2/global_spearman/0` | 0.8389 | 0.8396 | +0.0007 |
| `/corpora/gum/ablation/variants/stack-E1/global_spearman/0` | 0.7272 | 0.7278 | +0.0006 |
| `/corpora/gum/ablation/variants/stack-E2/global_spearman/0` | 0.7756 | 0.7753 | -0.0003 |
| `/corpora/gum/length/long/S1_global_spearman` | 0.8701 | 0.8703 | +0.0002 |
| `/corpora/gum/length/short/S1_global_spearman` | 0.8092 | 0.8094 | +0.0002 |
| `/corpora/gum/ablation/variants/stack/global_spearman/0` | 0.8180 | 0.8181 | +0.0001 |
| `/corpora/gum/baselines/S1/global_spearman/0` | 0.8180 | 0.8181 | +0.0001 |
| `/corpora/gentle/perturbation/cross_document/E2/global_spearman/0` | 0.4724 | 0.4724 | -0.0000 |
| `/corpora/gum/perturbation/cross_document/E1/global_spearman/0` | 0.7882 | 0.7882 | +0.0000 |

## a9/casing_report.json

排序类数值变化 160 个，最大 |Δ| = 0.0405；其他数值变化 0 个；非数值字段（选择、判定、标记）变化 0 个；路径不一致 0 个。

全局 Spearman 点估计变化最大的 15 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/corpora/ewt/evaluator/sat12lsm-cased/global_spearman/0` | 0.4962 | 0.5174 | +0.0212 |
| `/corpora/ewt/evaluator/sat12lsm-lower/global_spearman/0` | 0.4356 | 0.4493 | +0.0137 |
| `/corpora/ewt/evaluator/sat12l-cased/global_spearman/0` | 0.4326 | 0.4413 | +0.0087 |
| `/corpora/lines/evaluator/sat12lsm-cased/global_spearman/0` | 0.8810 | 0.8728 | -0.0083 |
| `/corpora/lines/evaluator/stack-cased/global_spearman/0` | 0.8324 | 0.8400 | +0.0076 |
| `/corpora/ewt/paired/sat12lsm-cased - sat12lsm-lower/global_spearman/0` | 0.0606 | 0.0681 | +0.0075 |
| `/corpora/ewt/paired/sat12l-cased - sat12l-lower/global_spearman/0` | 0.0527 | 0.0590 | +0.0063 |
| `/corpora/gumdev/paired/sat12lsm-cased - sat12lsm-lower/global_spearman/0` | 0.0573 | 0.0511 | -0.0062 |
| `/corpora/lines/evaluator/sat12lsm-lower/global_spearman/0` | 0.8365 | 0.8309 | -0.0056 |
| `/corpora/lines/paired/stack-cased - stack/global_spearman/0` | 0.0170 | 0.0223 | +0.0053 |
| `/corpora/gumdev/evaluator/sat12lsm-lower/global_spearman/0` | 0.8400 | 0.8432 | +0.0031 |
| `/corpora/gumdev/evaluator/sat12lsm-cased/global_spearman/0` | 0.8973 | 0.8942 | -0.0031 |
| `/corpora/lines/paired/sat12lsm-cased - sat12lsm-lower/global_spearman/0` | 0.0446 | 0.0419 | -0.0027 |
| `/corpora/ewt/evaluator/sat12l-lower/global_spearman/0` | 0.3799 | 0.3823 | +0.0024 |
| `/corpora/lines/evaluator/stack/global_spearman/0` | 0.8154 | 0.8177 | +0.0023 |

## a9/layout_report.json

排序类数值变化 818 个，最大 |Δ| = 0.0421；其他数值变化 173 个；非数值字段（选择、判定、标记）变化 0 个；路径不一致 0 个。

其他数值变化（前 10 个）：

- `/corpora/ewt/scenarios/flat/stack_minus_M0/cased-flat/0`：-0.148866 → -0.134846
- `/corpora/ewt/scenarios/flat/stack_minus_M0/cased-flat/1`：-0.26774 → -0.256902
- `/corpora/ewt/scenarios/flat/stack_minus_M0/cased-flat/2`：-0.0380659 → -0.0204986
- `/corpora/ewt/scenarios/flat/stack_minus_M0/cased-nl/0`：-0.127494 → -0.0934624
- `/corpora/ewt/scenarios/flat/stack_minus_M0/cased-nl/1`：-0.251446 → -0.219262
- `/corpora/ewt/scenarios/flat/stack_minus_M0/cased-nl/2`：-0.013521 → 0.023257
- `/corpora/ewt/scenarios/flat/stack_minus_M0/lower-flat/0`：-0.167548 → -0.141376
- `/corpora/ewt/scenarios/flat/stack_minus_M0/lower-flat/1`：-0.284004 → -0.262557
- `/corpora/ewt/scenarios/flat/stack_minus_M0/lower-flat/2`：-0.0555526 → -0.0248434
- `/corpora/ewt/scenarios/flat/stack_minus_M0/lower-nl/0`：-0.121544 → -0.0967252

全局 Spearman 点估计变化最大的 15 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/corpora/ewt/scenarios/flat/evaluator/stack:cased-nl/global_spearman/0` | 0.5299 | 0.5639 | +0.0340 |
| `/corpora/ewt/scenarios/nl/evaluator/stack:cased-nl/global_spearman/0` | 0.5461 | 0.5728 | +0.0268 |
| `/corpora/ewt/scenarios/flat/evaluator/stack:lower-flat/global_spearman/0` | 0.4898 | 0.5160 | +0.0262 |
| `/corpora/ewt/scenarios/flat/evaluator/stack:lower-nl/global_spearman/0` | 0.5358 | 0.5606 | +0.0248 |
| `/corpora/ewt/scenarios/flat/evaluator/stack:ref91/global_spearman/0` | 0.5065 | 0.5305 | +0.0240 |
| `/corpora/ewt/scenarios/nl/evaluator/stack:lower-flat/global_spearman/0` | 0.5045 | 0.5256 | +0.0211 |
| `/corpora/ewt/scenarios/nl/evaluator/stack:lower-nl/global_spearman/0` | 0.5503 | 0.5702 | +0.0198 |
| `/corpora/ewt/scenarios/nl/evaluator/stack:ref91/global_spearman/0` | 0.5270 | 0.5449 | +0.0179 |
| `/corpora/ewt/scenarios/flat/evaluator/E2:cased-nl/global_spearman/0` | 0.5229 | 0.5389 | +0.0160 |
| `/corpora/ewt/scenarios/flat/evaluator/stack:cased-flat/global_spearman/0` | 0.5085 | 0.5225 | +0.0140 |
| `/corpora/ewt/scenarios/flat/evaluator/E2:lower-nl/global_spearman/0` | 0.4572 | 0.4707 | +0.0135 |
| `/corpora/ewt/scenarios/nl/evaluator/E2:cased-nl/global_spearman/0` | 0.5381 | 0.5514 | +0.0133 |
| `/corpora/ewt/scenarios/nl/evaluator/E2:lower-nl/global_spearman/0` | 0.4728 | 0.4860 | +0.0132 |
| `/corpora/ewt/scenarios/nl/evaluator/stack:cased-flat/global_spearman/0` | 0.5269 | 0.5380 | +0.0111 |
| `/corpora/ewt/scenarios/flat/evaluator/E1:cased-flat/global_spearman/0` | 0.4326 | 0.4413 | +0.0087 |

