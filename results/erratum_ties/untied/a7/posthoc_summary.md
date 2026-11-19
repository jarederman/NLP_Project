# 修订 7：测试后稳健性分析（只作分析）

GENTLE / LinES 上的数字都是测试后分析；修订 6 的选型、判据与结论不变。CI 为 1000 次 bootstrap。

GUM 重算与开发报告一致：True

## 7.1 软证据 vs 硬伪参考（证据 p ≥ 0.5 当参考答案）

| 全局 Spearman | GUM 开发（LOGO） | GENTLE（测试后） | LinES（测试后） |
|---|---:|---:|---:|
| S0 软（E0） | 0.574 | 0.672 | 0.868 |
| H(E0) 硬 | 0.508 | 0.675 | 0.820 |
| S1 软（stack） | 0.818 | 0.883 | 0.815 |
| H(stack) 硬 | 0.810 | 0.896 | 0.824 |
| S1 − H(stack) [CI] | +0.008 [-0.008, +0.026] | -0.013 [-0.037, +0.010] | -0.008 [-0.155, +0.171] |
| S0 − H(E0) [CI] | +0.066 [+0.040, +0.092] | -0.004 [-0.050, +0.054] | +0.048 [-0.020, +0.201] |
| 文档内配对：S1 / H(stack) | 0.817 / 0.822 | 0.743 / 0.698 | 0.771 / 0.740 |

## 7.2 同源偏差：各单一证据下每个系统的平均秩残差（正 = 被高估）

### GUM 开发（LOGO）

| 证据 | punkt | sat | stanza | wtp | 语料级排名 | 与各系统错误重叠率 |
|---|---:|---:|---:|---:|---|---|
| E0 | +0.042 | -0.058 | +0.006 | +0.009 | sat > stanza > punkt > wtp | punkt 0.35, sat 0.40, stanza 0.44, wtp 0.20 |
| E1 | +0.046 | -0.114 | -0.039 | +0.108 | sat > stanza > wtp > punkt | punkt 0.37, sat 0.39, stanza 0.41, wtp 0.29 |
| E1sm | +0.018 | +0.006 | -0.008 | -0.016 | sat > stanza > punkt > wtp | punkt 0.19, sat 0.39, stanza 0.25, wtp 0.10 |
| E2 | +0.093 | -0.088 | +0.044 | -0.049 | sat > stanza > punkt > wtp | punkt 0.40, sat 0.35, stanza 0.51, wtp 0.17 |
| E2b | +0.102 | -0.114 | +0.008 | +0.004 | sat > punkt > stanza > wtp | punkt 0.49, sat 0.39, stanza 0.48, wtp 0.26 |
| stack | +0.067 | -0.048 | +0.008 | -0.028 | sat > stanza > punkt > wtp | punkt 0.38, sat 0.40, stanza 0.45, wtp 0.19 |

SaT 残差 E1 − E0：-0.057；E1sm − E0：+0.064；Stanza 残差 E2 − E0：+0.038（GUM 被污染，只作参考）

### GENTLE（测试后）

| 证据 | punkt | sat | stanza | wtp | 语料级排名 | 与各系统错误重叠率 |
|---|---:|---:|---:|---:|---|---|
| E0 | +0.080 | -0.043 | +0.001 | -0.037 | sat > punkt > stanza > wtp | punkt 0.69, sat 0.62, stanza 0.69, wtp 0.48 |
| E1 | -0.010 | -0.038 | -0.048 | +0.096 | wtp > sat > stanza > punkt | punkt 0.51, sat 0.57, stanza 0.57, wtp 0.56 |
| E1sm | +0.125 | -0.068 | -0.031 | -0.026 | punkt > sat > stanza > wtp | punkt 0.47, sat 0.39, stanza 0.49, wtp 0.33 |
| E2 | +0.125 | -0.093 | +0.014 | -0.046 | punkt > stanza > sat > wtp | punkt 0.54, sat 0.43, stanza 0.59, wtp 0.39 |
| E2b | +0.131 | -0.097 | +0.005 | -0.038 | punkt > stanza > sat > wtp | punkt 0.53, sat 0.41, stanza 0.58, wtp 0.38 |
| stack | +0.046 | -0.034 | -0.004 | -0.009 | sat > stanza > punkt > wtp | punkt 0.49, sat 0.45, stanza 0.54, wtp 0.41 |

SaT 残差 E1 − E0：+0.005；E1sm − E0：-0.024；Stanza 残差 E2 − E0：+0.014；判定（> 0.05）：{'sat_bias_E1_minus_E0': False, 'sat_bias_E1sm_minus_E0': False, 'stanza_bias_E2_minus_E0': False}

### LinES（测试后）

| 证据 | punkt | sat | stanza | wtp | 语料级排名 | 与各系统错误重叠率 |
|---|---:|---:|---:|---:|---|---|
| E0 | +0.002 | -0.011 | +0.006 | +0.003 | sat > stanza > punkt > wtp | punkt 0.47, sat 0.53, stanza 0.54, wtp 0.32 |
| E1 | -0.051 | -0.039 | +0.012 | +0.077 | sat > stanza > wtp > punkt | punkt 0.40, sat 0.49, stanza 0.50, wtp 0.41 |
| E1sm | +0.014 | +0.002 | +0.006 | -0.022 | sat > stanza > punkt > wtp | punkt 0.41, sat 0.50, stanza 0.48, wtp 0.33 |
| E2 | +0.026 | -0.020 | +0.043 | -0.049 | stanza > sat > punkt > wtp | punkt 0.50, sat 0.54, stanza 0.57, wtp 0.33 |
| E2b | +0.039 | -0.032 | +0.046 | -0.052 | stanza > sat > punkt > wtp | punkt 0.57, sat 0.62, stanza 0.67, wtp 0.38 |
| stack | +0.005 | -0.014 | +0.049 | -0.040 | stanza > sat > punkt > wtp | punkt 0.47, sat 0.51, stanza 0.57, wtp 0.37 |

SaT 残差 E1 − E0：-0.028；E1sm − E0：+0.012；Stanza 残差 E2 − E0：+0.037；判定（> 0.05）：{'sat_bias_E1_minus_E0': False, 'sat_bias_E1sm_minus_E0': False, 'stanza_bias_E2_minus_E0': False}

## 7.3 小样本敏感性

| 配对差 [CI] | 语料 | 分层 bootstrap | 不分层 bootstrap |
|---|---|---:|---:|
| S1 − S0 | GUM 开发 | +0.244 [+0.186, +0.309] | +0.244 [+0.173, +0.321] |
| M3 − M0 | GUM 开发 | +0.127 [+0.090, +0.165] | +0.127 [+0.086, +0.171] |
| S1 − M0 | GUM 开发 | +0.085 [+0.042, +0.128] | +0.085 [+0.038, +0.129] |
| S1 − S0 | GENTLE | +0.212 [+0.101, +0.326] | +0.212 [+0.069, +0.411] |
| M3 − M0 | GENTLE | +0.070 [+0.010, +0.137] | +0.070 [-0.020, +0.188] |
| S1 − M0 | GENTLE | +0.053 [-0.016, +0.124] | +0.053 [-0.068, +0.176] |
| S1 − S0 | LinES | -0.053 [-0.177, +0.065] | -0.053 [-0.177, +0.065] |
| M3 − M0 | LinES | -0.069 [-0.342, +0.142] | -0.069 [-0.342, +0.142] |
| S1 − M0 | LinES | -0.046 [-0.268, +0.103] | -0.046 [-0.268, +0.103] |
| S1 − S0 | GENTLE + LinES | +0.155 [+0.078, +0.243] | +0.155 [+0.049, +0.285] |
| M3 − M0 | GENTLE + LinES | +0.011 [-0.036, +0.059] | +0.011 [-0.068, +0.094] |
| S1 − M0 | GENTLE + LinES | +0.001 [-0.047, +0.049] | +0.001 [-0.086, +0.088] |

GENTLE 逐一去掉一个领域（点估计）：

| 去掉的领域 | S1 − S0 | M3 − M0 | S1 | M0 |
|---|---:|---:|---:|---:|
| dictionary | +0.217 | +0.063 | 0.883 | 0.862 |
| esports | +0.143 | +0.101 | 0.880 | 0.789 |
| legal | +0.218 | +0.044 | 0.876 | 0.854 |
| medical | +0.237 | +0.100 | 0.890 | 0.793 |
| poetry | +0.273 | +0.041 | 0.892 | 0.863 |
| proof | +0.235 | +0.077 | 0.871 | 0.823 |
| syllabus | +0.155 | +0.044 | 0.880 | 0.853 |
| threat | +0.253 | +0.123 | 0.860 | 0.748 |

全部为正：{'S1_minus_S0': True, 'M3_minus_M0': True}；GENTLE 结论按 7.3 规则“稳健”：**False**
