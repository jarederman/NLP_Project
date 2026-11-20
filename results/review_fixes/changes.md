# 代码审查（2026-11-20）修复后重跑：与修复前报告的差异

旧值 = `results/review_fixes/before/`；新值 = 重跑后的 `results/`。只重跑 CPU 分析阶段；冻结的模型、证据和一次性测试的冻结分数都没有改动。

## a8/report.json

数值变化 461 个（最大 |Δ| = 0.6011）；非数值字段变化 0 个；路径不一致 0 个。

变化最大的 25 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/corpora/lines/error_pricing/reward_ratio/DS/miss_other_over_miss_terminal/1` | -22.6369 | -23.2379 | -0.6011 |
| `/corpora/lines/perturbation/per_scorer/DS/detect/split_terminal/0.1/2` | 0.6667 | 0.4444 | -0.2222 |
| `/corpora/lines/perturbation/per_scorer/DS/detect/split_terminal/0.2/2` | 0.6667 | 0.4444 | -0.2222 |
| `/corpora/lines/perturbation/per_scorer/DS/detect/split_terminal/0.4/2` | 0.6667 | 0.4444 | -0.2222 |
| `/corpora/lines/perturbation/per_scorer/DS/detect/split_terminal/0.1/0` | 0.2857 | 0.1429 | -0.1429 |
| `/corpora/lines/perturbation/per_scorer/DS/detect/split_terminal/0.2/0` | 0.2857 | 0.1429 | -0.1429 |
| `/corpora/lines/perturbation/per_scorer/DS/detect/split_terminal/0.4/0` | 0.2857 | 0.1429 | -0.1429 |
| `/corpora/lines/error_pricing/reward_ratio/DS/miss_other_over_miss_terminal/0` | -4.4040 | -4.2664 | +0.1376 |
| `/corpora/lines/error_pricing/reward_ratio/DS/per_system/wtp/miss_other` | 0.4540 | 0.3252 | -0.1288 |
| `/corpora/gentle/error_pricing/reward_ratio/DS/per_system/wtp/fp_punct` | 0.4233 | 0.5512 | +0.1279 |
| `/corpora/gentle/error_pricing/reward_ratio/DS/per_system/wtp/fp_terminal` | -0.2252 | -0.3488 | -0.1236 |
| `/corpora/lines/error_pricing/reward_ratio/DS/pooled/fp_punct/2` | 0.7394 | 0.8382 | +0.0988 |
| `/corpora/gentle/error_pricing/reward_ratio/DS/pooled/fp_punct/0` | 0.1744 | 0.2619 | +0.0875 |
| `/corpora/gentle/error_pricing/reward_ratio/DS/pooled/fp_punct/1` | -0.0935 | -0.0106 | +0.0829 |
| `/corpora/gentle/error_pricing/reward_ratio/DS/pooled/fp_punct/2` | 0.5165 | 0.5955 | +0.0789 |
| `/corpora/lines/error_pricing/reward_ratio/DS/per_system/sat/fp_punct` | -0.4534 | -0.5318 | -0.0785 |
| `/corpora/gentle/error_pricing/reward_ratio/DS/per_system/stanza/fp_terminal` | -1.4313 | -1.5066 | -0.0753 |
| `/corpora/gum/error_pricing/reward_ratio/DS/per_system/wtp/miss_other` | 0.4997 | 0.4308 | -0.0688 |
| `/corpora/lines/error_pricing/reward_ratio/DS/per_system/wtp/fp_punct` | 0.6642 | 0.7302 | +0.0659 |
| `/corpora/gum/error_pricing/reward_ratio/DS/per_system/wtp/fp_nopunct` | 0.7488 | 0.8125 | +0.0638 |
| `/corpora/gum/error_pricing/reward_ratio/DS/per_system/wtp/fp_punct` | 0.5081 | 0.5670 | +0.0589 |
| `/corpora/gum/error_pricing/reward_ratio/DS/pooled/fp_nopunct/2` | 0.7381 | 0.7917 | +0.0536 |
| `/corpora/gentle/error_pricing/reward_ratio/DS/per_system/wtp/miss_other` | 0.2707 | 0.2171 | -0.0536 |
| `/corpora/gum/error_pricing/reward_ratio/DS/pooled/fp_nopunct/1` | 0.5175 | 0.5698 | +0.0523 |
| `/corpora/lines/perturbation/cross_document/DS/between_doc` | 0.7667 | 0.7167 | -0.0500 |

## a11/report.json

数值变化 1713 个（最大 |Δ| = 0.4431）；非数值字段变化 0 个；路径不一致 4 个。

路径不一致（前 10 个）：

- `/corpora/ewt/perturbation/soft_scorer`
- `/corpora/gentle/perturbation/soft_scorer`
- `/corpora/gum/perturbation/soft_scorer`
- `/corpora/lines/perturbation/soft_scorer`

变化最大的 25 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/corpora/ewt/perturbation/slope/hi/0/0` | 1.0076 | 0.5644 | -0.4431 |
| `/corpora/ewt/perturbation/slope/point/0/0` | 0.9605 | 0.5218 | -0.4387 |
| `/corpora/ewt/perturbation/slope/lo/0/0` | 0.9182 | 0.4822 | -0.4360 |
| `/corpora/ewt/perturbation/slope/lo/2/0` | 1.1029 | 0.7167 | -0.3862 |
| `/corpora/ewt/perturbation/slope/point/2/0` | 1.1302 | 0.7471 | -0.3831 |
| `/corpora/ewt/perturbation/slope/hi/2/0` | 1.1589 | 0.7761 | -0.3828 |
| `/corpora/ewt/perturbation/slope/lo/0/6` | 0.7447 | 0.3750 | -0.3698 |
| `/corpora/ewt/perturbation/slope/point/0/6` | 0.7783 | 0.4089 | -0.3694 |
| `/corpora/ewt/perturbation/slope/hi/0/6` | 0.8123 | 0.4439 | -0.3684 |
| `/corpora/ewt/perturbation/slope/lo/3/0` | 1.0640 | 0.6984 | -0.3656 |
| `/corpora/ewt/perturbation/slope/point/3/0` | 1.0866 | 0.7242 | -0.3624 |
| `/corpora/ewt/perturbation/slope/hi/3/0` | 1.1128 | 0.7509 | -0.3619 |
| `/corpora/ewt/perturbation/slope/hi/4/0` | 1.1036 | 0.7447 | -0.3589 |
| `/corpora/ewt/perturbation/slope/hi/6/0` | 1.0863 | 0.7279 | -0.3584 |
| `/corpora/ewt/perturbation/slope/point/4/0` | 1.0732 | 0.7157 | -0.3575 |
| `/corpora/ewt/perturbation/slope/lo/4/0` | 1.0449 | 0.6884 | -0.3565 |
| `/corpora/ewt/perturbation/slope/point/6/0` | 1.0544 | 0.6990 | -0.3555 |
| `/corpora/ewt/perturbation/slope/lo/6/0` | 1.0234 | 0.6709 | -0.3525 |
| `/corpora/ewt/perturbation/slope/lo/2/6` | 0.9519 | 0.6054 | -0.3464 |
| `/corpora/ewt/perturbation/slope/lo/5/0` | 1.0145 | 0.6698 | -0.3447 |
| `/corpora/ewt/perturbation/slope/point/5/0` | 1.0360 | 0.6936 | -0.3424 |
| `/corpora/ewt/perturbation/slope/hi/5/0` | 1.0593 | 0.7178 | -0.3414 |
| `/corpora/ewt/perturbation/slope/lo/3/6` | 0.9608 | 0.6234 | -0.3374 |
| `/corpora/ewt/perturbation/slope/point/2/6` | 0.9650 | 0.6308 | -0.3342 |
| `/corpora/ewt/perturbation/slope/hi/1/0` | 0.9911 | 0.6576 | -0.3335 |

## a12/report.json

数值变化 790 个（最大 |Δ| = 0.3558）；非数值字段变化 0 个；路径不一致 2 个。

路径不一致（前 10 个）：

- `/corpora/ewtdev/perturbation/soft_scorer`
- `/corpora/linesx/perturbation/soft_scorer`

变化最大的 25 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/corpora/ewtdev/perturbation/slope/lo/1/0` | 1.0782 | 0.7224 | -0.3558 |
| `/corpora/ewtdev/perturbation/slope/hi/1/0` | 1.1232 | 0.7713 | -0.3518 |
| `/corpora/ewtdev/perturbation/slope/point/1/0` | 1.0984 | 0.7469 | -0.3516 |
| `/corpora/ewtdev/perturbation/slope/hi/3/0` | 1.1094 | 0.7646 | -0.3448 |
| `/corpora/ewtdev/perturbation/slope/hi/5/0` | 1.0924 | 0.7488 | -0.3436 |
| `/corpora/ewtdev/perturbation/slope/point/3/0` | 1.0784 | 0.7349 | -0.3434 |
| `/corpora/ewtdev/perturbation/slope/lo/3/0` | 1.0526 | 0.7099 | -0.3427 |
| `/corpora/ewtdev/perturbation/slope/point/5/0` | 1.0623 | 0.7201 | -0.3422 |
| `/corpora/ewtdev/perturbation/slope/lo/5/0` | 1.0370 | 0.6963 | -0.3407 |
| `/corpora/ewtdev/perturbation/slope/lo/2/0` | 1.0457 | 0.7064 | -0.3392 |
| `/corpora/ewtdev/perturbation/slope/lo/1/6` | 0.9679 | 0.6303 | -0.3377 |
| `/corpora/ewtdev/perturbation/slope/point/2/0` | 1.0618 | 0.7286 | -0.3332 |
| `/corpora/ewtdev/perturbation/slope/hi/2/0` | 1.0809 | 0.7511 | -0.3298 |
| `/corpora/ewtdev/perturbation/slope/lo/2/6` | 0.9746 | 0.6491 | -0.3254 |
| `/corpora/ewtdev/perturbation/slope/point/1/6` | 0.9802 | 0.6560 | -0.3242 |
| `/corpora/ewtdev/perturbation/slope/lo/4/0` | 1.0084 | 0.6851 | -0.3234 |
| `/corpora/ewtdev/perturbation/slope/lo/5/6` | 0.9537 | 0.6350 | -0.3187 |
| `/corpora/ewtdev/perturbation/slope/lo/3/6` | 0.9553 | 0.6374 | -0.3179 |
| `/corpora/ewtdev/perturbation/slope/point/4/0` | 1.0215 | 0.7059 | -0.3155 |
| `/corpora/ewtdev/perturbation/slope/point/2/6` | 0.9859 | 0.6706 | -0.3153 |
| `/corpora/ewtdev/perturbation/slope/lo/0/0` | 0.8960 | 0.5828 | -0.3132 |
| `/corpora/ewtdev/perturbation/slope/point/5/6` | 0.9692 | 0.6571 | -0.3121 |
| `/corpora/ewtdev/perturbation/slope/point/0/0` | 0.9257 | 0.6141 | -0.3116 |
| `/corpora/ewtdev/perturbation/slope/point/3/6` | 0.9711 | 0.6599 | -0.3111 |
| `/corpora/ewtdev/perturbation/slope/lo/4/6` | 0.9623 | 0.6515 | -0.3109 |

## a13/report.json

数值变化 98 个（最大 |Δ| = 0.0307）；非数值字段变化 0 个；路径不一致 0 个。

变化最大的 25 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/shift/selectivity_ci/1` | 30.7888 | 30.8195 | +0.0307 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/shift/selectivity_ci/0` | 21.9797 | 21.9498 | -0.0300 |
| `/corpora/gum/scenarios/flat/ranking/Masked LM (PuncEv)/genre_tau/1` | 0.1556 | 0.1333 | -0.0222 |
| `/corpora/gum/scenarios/flat/ranking/Masked LM (PuncEv)/genre_tau/2` | 0.3778 | 0.3556 | -0.0222 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/split_punct/selectivity_ci/1` | 14.5656 | 14.5850 | +0.0194 |
| `/corpora/gum/sensitivity/0.10/detect/Masked LM (PuncEv)/merge_other_inpar/2` | 0.5334 | 0.5513 | +0.0179 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/merge_other_par/selectivity_ci/0` | -3.6012 | -3.6178 | -0.0165 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/split_nopunct/selectivity_ci/1` | 14.4150 | 14.4291 | +0.0141 |
| `/corpora/gum/sensitivity/0.10/detect/Masked LM (PuncEv)/merge_other_inpar/0` | 0.4267 | 0.4400 | +0.0133 |
| `/corpora/gum/sensitivity/k=1/detect/Masked LM (PuncEv)/merge_other_inpar/2` | 0.4900 | 0.5033 | +0.0132 |
| `/corpora/gum/sensitivity/k=1/detect/Masked LM (PuncEv)/merge_other_inpar/1` | 0.3058 | 0.3187 | +0.0128 |
| `/corpora/gum/sensitivity/k=1/detect/Masked LM (PuncEv)/merge_other_inpar/0` | 0.3953 | 0.4060 | +0.0107 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/split_terminal/z` | -6.4212 | -6.4107 | +0.0106 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/merge_terminal/z` | 16.3188 | 16.3086 | -0.0102 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/merge_other_inpar/selectivity_ci/0` | -1.7819 | -1.7717 | +0.0102 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/split_nopunct/selectivity_ci/0` | 10.3397 | 10.3297 | -0.0100 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/merge_other_par/selectivity_ci/1` | -2.0020 | -2.0114 | -0.0095 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/split_terminal/selectivity_ci/1` | -5.1784 | -5.1692 | +0.0092 |
| `/corpora/gum/sensitivity/0.10/detect/Masked LM (PuncEv)/merge_other_inpar/1` | 0.3291 | 0.3378 | +0.0087 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/split_punct/selectivity_ci/0` | 10.4273 | 10.4188 | -0.0085 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/merge_other_par/z` | -2.7079 | -2.7161 | -0.0083 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/split_terminal/selectivity_ci/0` | -8.1190 | -8.1113 | +0.0078 |
| `/corpora/gum/sensitivity/k=1/z/Masked LM (PuncEv)/merge_terminal/selectivity_ci/1` | 4.3746 | 4.3815 | +0.0069 |
| `/corpora/gum/sensitivity/0.10/detect/Masked LM (PuncEv)/merge_other_par/1` | 0.2557 | 0.2491 | -0.0066 |
| `/corpora/gum/sensitivity/0.10/z/Masked LM (PuncEv)/merge_terminal/selectivity_ci/1` | 19.5218 | 19.5281 | +0.0063 |

## a13/extra.json

数值变化 0 个（最大 |Δ| = 0.0000）；非数值字段变化 0 个；路径不一致 84 个。

路径不一致（前 10 个）：

- `/ewtdev/nl/sensitivity_k1/Ensemble (ours + SaT-12l-sm, average)/_detected/merge_other_inpar`
- `/ewtdev/nl/sensitivity_k1/Ensemble (ours + SaT-12l-sm, average)/_detected/merge_other_par`
- `/ewtdev/nl/sensitivity_k1/Ensemble (ours + SaT-12l-sm, average)/_detected/merge_terminal`
- `/ewtdev/nl/sensitivity_k1/Ensemble (ours + SaT-12l-sm, average)/_detected/shift`
- `/ewtdev/nl/sensitivity_k1/Ensemble (ours + SaT-12l-sm, average)/_detected/split_nopunct`
- `/ewtdev/nl/sensitivity_k1/Ensemble (ours + SaT-12l-sm, average)/_detected/split_punct`
- `/ewtdev/nl/sensitivity_k1/Ensemble (ours + SaT-12l-sm, average)/_detected/split_terminal`
- `/ewtdev/nl/sensitivity_k1/Ours/_detected/merge_other_inpar`
- `/ewtdev/nl/sensitivity_k1/Ours/_detected/merge_other_par`
- `/ewtdev/nl/sensitivity_k1/Ours/_detected/merge_terminal`

## a13/classic.json

数值变化 145 个（最大 |Δ| = 0.6667）；非数值字段变化 0 个；路径不一致 0 个。

变化最大的 25 个：

| 路径 | 旧 | 新 | Δ |
|---|---:|---:|---:|
| `/corpora/gentle/scenarios/nl/ranking/Dawid-Skene/corpus_tau` | 1.0000 | 0.3333 | -0.6667 |
| `/corpora/gentle/scenarios/nl/ranking/Dawid-Skene/genre_tau/0` | 0.6141 | 0.2391 | -0.3750 |
| `/corpora/gentle/scenarios/flat/ranking/Dawid-Skene/genre_tau/0` | 0.5308 | 0.1935 | -0.3373 |
| `/corpora/ewt/scenarios/flat/ranking/Dawid-Skene/corpus_tau` | 0.6667 | 0.3333 | -0.3333 |
| `/corpora/ewtdev/scenarios/flat/ranking/Dawid-Skene/corpus_tau` | 0.6667 | 0.3333 | -0.3333 |
| `/corpora/gentle/scenarios/flat/ranking/Dawid-Skene/corpus_tau` | 0.6667 | 0.3333 | -0.3333 |
| `/corpora/gum/scenarios/nl/ranking/Dawid-Skene/corpus_tau` | 0.6667 | 0.3333 | -0.3333 |
| `/corpora/linesx/scenarios/flat/ranking/Dawid-Skene/genre_tau/2` | 1.0000 | 0.6667 | -0.3333 |
| `/corpora/gentle/scenarios/nl/ranking/Dawid-Skene/genre_tau/1` | 0.4167 | 0.1141 | -0.3026 |
| `/corpora/gentle/scenarios/flat/ranking/Dawid-Skene/genre_tau/2` | 0.6250 | 0.3333 | -0.2917 |
| `/corpora/gentle/scenarios/flat/ranking/Dawid-Skene/genre_tau/1` | 0.2808 | 0.0268 | -0.2540 |
| `/corpora/gentle/scenarios/nl/ranking/Dawid-Skene/genre_tau/2` | 0.7917 | 0.5417 | -0.2500 |
| `/corpora/gentle/scenarios/flat/ranking/Dawid-Skene/doc_tau/1` | 0.2969 | 0.1168 | -0.1801 |
| `/corpora/gentle/scenarios/nl/ranking/Dawid-Skene/doc_tau/1` | 0.4490 | 0.2815 | -0.1675 |
| `/corpora/gentle/scenarios/nl/ranking/Dawid-Skene/doc_tau/0` | 0.5855 | 0.4200 | -0.1654 |
| `/corpora/ewt/scenarios/flat/ranking/Dawid-Skene/doc_tau/1` | 0.4356 | 0.2721 | -0.1635 |
| `/corpora/gentle/scenarios/nl/ranking/Dawid-Skene/doc_tau/2` | 0.7159 | 0.5577 | -0.1581 |
| `/corpora/gentle/scenarios/flat/ranking/Dawid-Skene/doc_tau/0` | 0.3945 | 0.2424 | -0.1521 |
| `/corpora/ewt/scenarios/flat/ranking/Dawid-Skene/doc_tau/0` | 0.5180 | 0.3665 | -0.1515 |
| `/corpora/ewt/scenarios/flat/ranking/Dawid-Skene/doc_tau/2` | 0.6033 | 0.4629 | -0.1404 |
| `/corpora/gentle/scenarios/flat/ranking/Dawid-Skene/doc_tau/2` | 0.4974 | 0.3655 | -0.1319 |
| `/corpora/gum/scenarios/flat/ranking/Dawid-Skene/genre_tau/0` | 0.7556 | 0.6444 | -0.1111 |
| `/corpora/ewt/scenarios/flat/table/Dawid-Skene/within_doc_pairwise` | 0.7541 | 0.6634 | -0.0906 |
| `/corpora/gum/scenarios/flat/ranking/Dawid-Skene/genre_tau/2` | 0.7778 | 0.6889 | -0.0889 |
| `/corpora/gum/scenarios/flat/ranking/Dawid-Skene/genre_tau/1` | 0.6000 | 0.5111 | -0.0889 |

## a13/softhard.json

数值变化 0 个（最大 |Δ| = 0.0000）；非数值字段变化 0 个；路径不一致 0 个。

