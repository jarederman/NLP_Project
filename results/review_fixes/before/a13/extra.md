# 修订 13 补充（事后分析 P）：同族偏差与不拟合的平均集成

## GUM (D)（无换行）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.931 [0.908, 0.951] | 0.931 | -0.013 | +0.022 | -0.014 | +0.005 |
| SaT-12l-sm probabilities | 0.936 [0.918, 0.953] | 0.940 | -0.010 | +0.031 | -0.005 | -0.015 |
| Ours | 0.903 [0.880, 0.923] | 0.897 | -0.003 | +0.008 | -0.008 | +0.004 |
| Ensemble (ours + SaT-12l-sm, average) | 0.924 [0.903, 0.942] | 0.925 | -0.013 | +0.039 | -0.007 | -0.018 |

集成 − 其他：SaT-12l-sm output as reference -0.008 [-0.028, +0.013]；SaT-12l-sm probabilities -0.012 [-0.024, -0.003]；Ours +0.021 [+0.009, +0.034]

## GUM (D)（带版式）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.953 [0.937, 0.968] | 0.974 | -0.011 | +0.021 | -0.015 | +0.004 |
| SaT-12l-sm probabilities | 0.965 [0.952, 0.977] | 0.979 | -0.014 | +0.032 | -0.013 | -0.005 |
| Ours | 0.948 [0.933, 0.961] | 0.946 | +0.020 | -0.004 | -0.006 | -0.010 |
| Ensemble (ours + SaT-12l-sm, average) | 0.962 [0.949, 0.974] | 0.972 | -0.003 | +0.025 | -0.011 | -0.011 |

集成 − 其他：SaT-12l-sm output as reference +0.009 [-0.008, +0.024]；SaT-12l-sm probabilities -0.003 [-0.010, +0.004]；Ours +0.014 [+0.007, +0.021]

检出率（1 个错误）：SaT-12l-sm probabilities: merge_terminal 1.00, merge_other_par 1.00, merge_other_inpar 0.69, split_terminal 0.89, split_punct 0.99, split_nopunct 1.00, shift 1.00, control 0.50；Ours: merge_terminal 0.99, merge_other_par 0.98, merge_other_inpar 0.07, split_terminal 0.81, split_punct 1.00, split_nopunct 1.00, shift 1.00, control 0.49；Ensemble (ours + SaT-12l-sm, average): merge_terminal 1.00, merge_other_par 1.00, merge_other_inpar 0.57, split_terminal 0.88, split_punct 1.00, split_nopunct 1.00, shift 1.00, control 0.49

## GENTLE（无换行）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.880 [0.785, 0.951] | 0.833 | +0.038 | +0.011 | -0.043 | -0.006 |
| SaT-12l-sm probabilities | 0.891 [0.817, 0.952] | 0.833 | +0.067 | -0.008 | -0.034 | -0.026 |
| Ours | 0.907 [0.860, 0.938] | 0.785 | -0.012 | -0.002 | -0.013 | +0.027 |
| Ensemble (ours + SaT-12l-sm, average) | 0.902 [0.853, 0.937] | 0.840 | +0.018 | +0.021 | -0.023 | -0.016 |

集成 − 其他：SaT-12l-sm output as reference +0.022 [-0.040, +0.096]；SaT-12l-sm probabilities +0.011 [-0.026, +0.048]；Ours -0.004 [-0.044, +0.030]

## GENTLE（带版式）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.909 [0.825, 0.968] | 0.883 | +0.022 | +0.013 | -0.041 | +0.006 |
| SaT-12l-sm probabilities | 0.929 [0.860, 0.971] | 0.879 | +0.039 | +0.006 | -0.028 | -0.017 |
| Ours | 0.903 [0.836, 0.961] | 0.879 | +0.080 | -0.046 | -0.009 | -0.025 |
| Ensemble (ours + SaT-12l-sm, average) | 0.922 [0.861, 0.968] | 0.886 | +0.053 | -0.008 | -0.018 | -0.027 |

集成 − 其他：SaT-12l-sm output as reference +0.013 [-0.038, +0.069]；SaT-12l-sm probabilities -0.006 [-0.033, +0.014]；Ours +0.020 [-0.007, +0.048]

检出率（1 个错误）：SaT-12l-sm probabilities: merge_terminal 0.99, merge_other_par 1.00, merge_other_inpar 0.55, split_terminal 0.79, split_punct 0.98, split_nopunct 1.00, shift 1.00, control 0.51；Ours: merge_terminal 0.99, merge_other_par 1.00, merge_other_inpar 0.17, split_terminal 0.76, split_punct 1.00, split_nopunct 1.00, shift 1.00, control 0.52；Ensemble (ours + SaT-12l-sm, average): merge_terminal 0.99, merge_other_par 1.00, merge_other_inpar 0.43, split_terminal 0.81, split_punct 0.99, split_nopunct 1.00, shift 1.00, control 0.51

## LinES test（无换行）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.888 [0.632, 0.971] | 0.917 | -0.025 | +0.023 | +0.034 | -0.032 |
| SaT-12l-sm probabilities | 0.873 [0.622, 0.962] | 0.917 | -0.020 | +0.023 | +0.059 | -0.062 |
| Ours | 0.873 [0.623, 0.949] | 0.833 | -0.014 | +0.023 | +0.034 | -0.043 |
| Ensemble (ours + SaT-12l-sm, average) | 0.876 [0.624, 0.958] | 0.896 | -0.023 | +0.023 | +0.059 | -0.059 |

集成 − 其他：SaT-12l-sm output as reference -0.012 [-0.059, +0.059]；SaT-12l-sm probabilities +0.003 [-0.030, +0.042]；Ours +0.003 [-0.028, +0.093]

## EWT test（无换行）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.754 [0.687, 0.812] | 0.835 | -0.004 | +0.025 | -0.022 | +0.002 |
| SaT-12l-sm probabilities | 0.778 [0.718, 0.834] | 0.829 | +0.008 | +0.019 | -0.002 | -0.025 |
| Ours | 0.787 [0.734, 0.840] | 0.832 | -0.013 | +0.021 | -0.023 | +0.015 |
| Ensemble (ours + SaT-12l-sm, average) | 0.818 [0.767, 0.863] | 0.846 | -0.012 | +0.033 | -0.013 | -0.007 |

集成 − 其他：SaT-12l-sm output as reference +0.063 [+0.012, +0.112]；SaT-12l-sm probabilities +0.040 [-0.011, +0.086]；Ours +0.030 [-0.003, +0.060]

## EWT test（带版式）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.828 [0.774, 0.873] | 0.890 | -0.010 | +0.017 | -0.025 | +0.018 |
| SaT-12l-sm probabilities | 0.879 [0.841, 0.912] | 0.908 | -0.002 | +0.020 | -0.012 | -0.007 |
| Ours | 0.894 [0.859, 0.924] | 0.874 | +0.021 | +0.002 | -0.001 | -0.022 |
| Ensemble (ours + SaT-12l-sm, average) | 0.910 [0.880, 0.937] | 0.902 | +0.004 | +0.023 | -0.007 | -0.020 |

集成 − 其他：SaT-12l-sm output as reference +0.082 [+0.039, +0.133]；SaT-12l-sm probabilities +0.032 [+0.000, +0.063]；Ours +0.017 [+0.002, +0.033]

## EWT dev (fresh)（无换行）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.738 [0.677, 0.791] | 0.812 | +0.008 | +0.019 | -0.026 | -0.001 |
| SaT-12l-sm probabilities | 0.762 [0.709, 0.812] | 0.807 | +0.012 | +0.024 | -0.014 | -0.022 |
| Ours | 0.797 [0.746, 0.843] | 0.823 | -0.000 | +0.019 | -0.027 | +0.008 |
| Ensemble (ours + SaT-12l-sm, average) | 0.808 [0.757, 0.852] | 0.829 | +0.000 | +0.031 | -0.021 | -0.011 |

集成 − 其他：SaT-12l-sm output as reference +0.070 [+0.022, +0.116]；SaT-12l-sm probabilities +0.046 [+0.003, +0.092]；Ours +0.010 [-0.012, +0.033]

## EWT dev (fresh)（带版式）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.797 [0.747, 0.843] | 0.878 | +0.004 | +0.019 | -0.028 | +0.006 |
| SaT-12l-sm probabilities | 0.835 [0.794, 0.871] | 0.881 | +0.006 | +0.030 | -0.020 | -0.017 |
| Ours | 0.851 [0.806, 0.892] | 0.869 | +0.033 | +0.003 | -0.008 | -0.027 |
| Ensemble (ours + SaT-12l-sm, average) | 0.862 [0.824, 0.896] | 0.893 | +0.015 | +0.027 | -0.014 | -0.028 |

集成 − 其他：SaT-12l-sm output as reference +0.065 [+0.024, +0.108]；SaT-12l-sm probabilities +0.027 [+0.007, +0.048]；Ours +0.011 [-0.010, +0.035]

检出率（1 个错误）：SaT-12l-sm probabilities: merge_terminal 0.97, merge_other_par 1.00, merge_other_inpar 0.63, split_terminal 0.75, split_punct 0.99, split_nopunct 1.00, shift 1.00, control 0.49；Ours: merge_terminal 0.99, merge_other_par 1.00, merge_other_inpar 0.31, split_terminal 0.69, split_punct 1.00, split_nopunct 1.00, shift 1.00, control 0.49；Ensemble (ours + SaT-12l-sm, average): merge_terminal 0.99, merge_other_par 1.00, merge_other_inpar 0.58, split_terminal 0.73, split_punct 0.99, split_nopunct 1.00, shift 1.00, control 0.49

## LinES train+dev (fresh)（无换行）

| 方法 | 全局 Spearman [CI] | 文档内配对 | 秩残差 punkt | 秩残差 sat | 秩残差 stanza | 秩残差 wtp |
|---|---:|---:|---:|---:|---:|---:|
| SaT-12l-sm output as reference | 0.630 [0.446, 0.809] | 0.898 | +0.006 | +0.042 | +0.027 | -0.074 |
| SaT-12l-sm probabilities | 0.630 [0.484, 0.784] | 0.932 | +0.016 | +0.044 | +0.054 | -0.113 |
| Ours | 0.563 [0.321, 0.788] | 0.922 | +0.022 | +0.041 | +0.038 | -0.101 |
| Ensemble (ours + SaT-12l-sm, average) | 0.532 [0.333, 0.746] | 0.932 | +0.024 | +0.054 | +0.052 | -0.130 |

集成 − 其他：SaT-12l-sm output as reference -0.098 [-0.223, +0.038]；SaT-12l-sm probabilities -0.098 [-0.210, +0.006]；Ours -0.031 [-0.118, +0.061]

检出率（1 个错误）：SaT-12l-sm probabilities: merge_terminal 1.00, merge_other_par nan, merge_other_inpar 0.48, split_terminal 0.39, split_punct 1.00, split_nopunct 1.00, shift 1.00, control 0.51；Ours: merge_terminal 1.00, merge_other_par nan, merge_other_inpar 0.51, split_terminal 0.23, split_punct 1.00, split_nopunct 1.00, shift 1.00, control 0.56；Ensemble (ours + SaT-12l-sm, average): merge_terminal 1.00, merge_other_par nan, merge_other_inpar 0.47, split_terminal 0.32, split_punct 1.00, split_nopunct 1.00, shift 1.00, control 0.57

