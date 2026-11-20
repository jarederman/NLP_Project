# 修订 13 补充基线（事后分析 P）：B8 GPT-2 困惑度、B9 Dawid–Skene

## 全局 Spearman（无换行；括号内为文档内配对准确率）

| 方法 | GUM (D) | GENTLE | LinES test | EWT test | EWT dev (fresh) | LinES train+dev (fresh) |
|---|---:|---:|---:|---:|---:|---:|
| Punctuation rule | 0.340 (0.45) | 0.383 (0.41) | 0.437 (0.54) | 0.369 (0.49) | 0.378 (0.50) | 0.296 (0.55) |
| Length prior | 0.048 (0.53) | 0.037 (0.55) | 0.079 (0.42) | 0.161 (0.51) | 0.157 (0.50) | 0.267 (0.42) |
| Masked LM (PuncEv) | 0.349 (0.63) | 0.542 (0.48) | 0.758 (0.81) | 0.508 (0.60) | 0.464 (0.62) | 0.440 (0.70) |
| SaT-12l-sm output as reference | 0.931 (0.93) | 0.880 (0.83) | 0.888 (0.92) | 0.754 (0.84) | 0.738 (0.81) | 0.630 (0.90) |
| SaT-12l-sm probabilities | 0.936 (0.94) | 0.891 (0.83) | 0.873 (0.92) | 0.778 (0.83) | 0.762 (0.81) | 0.630 (0.93) |
| Consensus M0 | 0.733 (0.73) | 0.830 (0.62) | 0.861 (0.74) | 0.657 (0.72) | 0.657 (0.72) | 0.468 (0.80) |
| DocAgree | 0.637 (0.50) | 0.773 (0.50) | 0.741 (0.50) | 0.529 (0.50) | 0.508 (0.50) | 0.410 (0.50) |
| GPT-2 perplexity | 0.104 (0.77) | 0.064 (0.65) | -0.322 (0.50) | 0.043 (0.74) | 0.062 (0.71) | 0.049 (0.78) |
| Dawid-Skene | 0.746 (0.77) | 0.826 (0.69) | 0.859 (0.77) | 0.667 (0.75) | 0.618 (0.70) | 0.505 (0.81) |
| Ours | 0.903 (0.90) | 0.907 (0.78) | 0.873 (0.83) | 0.787 (0.83) | 0.797 (0.82) | 0.563 (0.92) |

Ours − 经典基线：GUM (D): GPT-2 perplexity +0.798 [+0.687, +0.898], Dawid-Skene +0.157 [+0.115, +0.205]；GENTLE: GPT-2 perplexity +0.842 [+0.600, +1.071], Dawid-Skene +0.080 [+0.005, +0.161]；LinES test: GPT-2 perplexity +1.194 [+0.419, +1.639], Dawid-Skene +0.013 [-0.110, +0.153]；EWT test: GPT-2 perplexity +0.745 [+0.644, +0.844], Dawid-Skene +0.121 [+0.051, +0.186]；EWT dev (fresh): GPT-2 perplexity +0.735 [+0.640, +0.838], Dawid-Skene +0.179 [+0.113, +0.249]；LinES train+dev (fresh): GPT-2 perplexity +0.514 [+0.049, +0.985], Dawid-Skene +0.058 [-0.030, +0.166]

系统排名（领域平均 τ / 文档平均 τ）：GUM (D): GPT-2 perplexity 0.56/0.55, Dawid-Skene 0.76/0.53, Ours 0.87/0.80；GENTLE: GPT-2 perplexity 0.36/0.31, Dawid-Skene 0.53/0.39, Ours 0.61/0.55；LinES test: GPT-2 perplexity 0.33/0.05, Dawid-Skene 0.67/0.57, Ours 1.00/0.68；EWT test: GPT-2 perplexity 0.87/0.50, Dawid-Skene 0.60/0.52, Ours 0.67/0.70；EWT dev (fresh): GPT-2 perplexity 0.47/0.47, Dawid-Skene 0.60/0.45, Ours 0.53/0.70；LinES train+dev (fresh): GPT-2 perplexity 0.33/0.55, Dawid-Skene 0.67/0.60, Ours 1.00/0.84

## 全局 Spearman（带版式；括号内为文档内配对准确率）

| 方法 | GUM (D) | GENTLE | EWT test | EWT dev (fresh) |
|---|---:|---:|---:|---:|
| Punctuation rule | 0.827 (0.89) | 0.734 (0.73) | 0.842 (0.83) | 0.796 (0.81) |
| Length prior | 0.223 (0.72) | 0.076 (0.59) | 0.142 (0.57) | 0.149 (0.54) |
| Masked LM (PuncEv) | 0.764 (0.92) | 0.708 (0.79) | 0.736 (0.83) | 0.698 (0.84) |
| SaT-12l-sm output as reference | 0.953 (0.97) | 0.909 (0.88) | 0.828 (0.89) | 0.797 (0.88) |
| SaT-12l-sm probabilities | 0.965 (0.98) | 0.929 (0.88) | 0.879 (0.91) | 0.835 (0.88) |
| Consensus M0 | 0.675 (0.72) | 0.795 (0.67) | 0.590 (0.71) | 0.625 (0.73) |
| DocAgree | 0.475 (0.50) | 0.747 (0.50) | 0.482 (0.50) | 0.509 (0.50) |
| GPT-2 perplexity | 0.071 (0.70) | 0.037 (0.56) | 0.008 (0.62) | 0.039 (0.64) |
| Dawid-Skene | 0.698 (0.77) | 0.868 (0.79) | 0.604 (0.71) | 0.619 (0.73) |
| Ours | 0.948 (0.95) | 0.903 (0.88) | 0.894 (0.87) | 0.851 (0.87) |

Ours − 经典基线：GUM (D): GPT-2 perplexity +0.877 [+0.789, +0.954], Dawid-Skene +0.250 [+0.214, +0.287]；GENTLE: GPT-2 perplexity +0.866 [+0.624, +1.081], Dawid-Skene +0.035 [-0.031, +0.097]；EWT test: GPT-2 perplexity +0.885 [+0.802, +0.965], Dawid-Skene +0.290 [+0.221, +0.358]；EWT dev (fresh): GPT-2 perplexity +0.812 [+0.719, +0.906], Dawid-Skene +0.232 [+0.173, +0.293]

系统排名（领域平均 τ / 文档平均 τ）：GUM (D): GPT-2 perplexity 0.62/0.43, Dawid-Skene 0.56/0.53, Ours 0.93/0.89；GENTLE: GPT-2 perplexity 0.07/0.13, Dawid-Skene 0.61/0.59, Ours 0.73/0.76；EWT test: GPT-2 perplexity 0.73/0.34, Dawid-Skene 0.40/0.41, Ours 0.87/0.79；EWT dev (fresh): GPT-2 perplexity 0.60/0.38, Dawid-Skene 0.47/0.48, Ours 1.00/0.78

## 敏感度：注入 1 个错误的检出率（GUM (D)；* = 95% CI 下界 > 0.5）

| 方法 | merge_terminal | merge_other_par | merge_other_inpar | split_terminal | split_punct | split_nopunct | shift | control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Punctuation rule | 1.00* | 0.00 | 0.00 | 0.00 | 1.00* | 1.00* | 0.94* | 0.50 |
| Length prior | 0.99* | 1.00* | 1.00* | 0.00 | 0.01 | 0.01 | 0.49 | 0.50 |
| Masked LM (PuncEv) | 0.92* | 0.29 | 0.40 | 0.22 | 0.99* | 0.99* | 0.97* | 0.49 |
| SaT-12l-sm output as reference | 1.00* | 0.92* | 0.80* | 0.87* | 0.99* | 1.00* | 0.99* | 0.50 |
| SaT-12l-sm probabilities | 1.00* | 0.89* | 0.69* | 0.89* | 0.99* | 1.00* | 1.00* | 0.50 |
| Consensus | 1.00* | 0.66* | 0.57 | 0.60* | 0.99* | 1.00* | 0.98* | 0.50 |
| DocAgree | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| Ours (flat) | 1.00* | 0.87* | 0.62* | 0.84* | 0.99* | 1.00* | 1.00* | 0.50 |
| Dawid-Skene | 1.00* | 0.68* | 0.60* | 0.60* | 0.99* | 1.00* | 0.99* | 0.50 |
| Ours (layout) | 0.99* | 0.98* | 0.07 | 0.81* | 1.00* | 1.00* | 1.00* | 0.49 |

## 敏感度：注入 1 个错误的检出率（GENTLE；* = 95% CI 下界 > 0.5）

| 方法 | merge_terminal | merge_other_par | merge_other_inpar | split_terminal | split_punct | split_nopunct | shift | control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Punctuation rule | 1.00* | 0.00 | 0.00 | 0.00 | 1.00* | 1.00* | 0.82* | 0.50 |
| Length prior | 1.00* | 1.00* | 0.99* | 0.00 | 0.01 | 0.01 | 0.53 | 0.50 |
| Masked LM (PuncEv) | 0.89* | 0.17 | 0.22 | 0.36 | 0.98* | 0.97* | 0.90* | 0.50 |
| SaT-12l-sm output as reference | 0.99* | 0.72* | 0.59 | 0.78* | 0.97* | 1.00* | 0.93* | 0.50 |
| SaT-12l-sm probabilities | 0.99* | 0.67* | 0.57 | 0.79* | 0.98* | 1.00* | 0.99* | 0.51 |
| Consensus | 1.00* | 0.54 | 0.64* | 0.42 | 0.95* | 0.99* | 0.90* | 0.50 |
| DocAgree | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| Ours (flat) | 0.99* | 0.73* | 0.64* | 0.79* | 0.92* | 0.98* | 0.99* | 0.49 |
| Dawid-Skene | 1.00* | 0.59 | 0.64* | 0.42 | 0.95* | 0.99* | 0.95* | 0.51 |
| Ours (layout) | 0.99* | 1.00* | 0.17 | 0.76* | 1.00* | 1.00* | 1.00* | 0.52 |

## 敏感度：注入 1 个错误的检出率（EWT dev (fresh)；* = 95% CI 下界 > 0.5）

| 方法 | merge_terminal | merge_other_par | merge_other_inpar | split_terminal | split_punct | split_nopunct | shift | control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Punctuation rule | 1.00* | 0.01 | 0.01 | 0.00 | 0.97* | 0.97* | 0.89* | 0.50 |
| Length prior | 1.00* | 1.00* | 1.00* | 0.00 | 0.01 | 0.00 | 0.52 | 0.50 |
| Masked LM (PuncEv) | 0.81* | 0.54 | 0.28 | 0.23 | 0.99* | 0.97* | 0.95* | 0.51 |
| SaT-12l-sm output as reference | 0.98* | 0.91* | 0.71* | 0.68* | 0.97* | 0.99* | 0.97* | 0.50 |
| SaT-12l-sm probabilities | 0.96* | 0.85* | 0.64* | 0.75* | 0.99* | 1.00* | 1.00* | 0.49 |
| Consensus | 1.00* | 0.52 | 0.51 | 0.55 | 0.97* | 0.98* | 0.96* | 0.50 |
| DocAgree | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| Ours (flat) | 0.99* | 0.93* | 0.84* | 0.66* | 0.97* | 0.99* | 1.00* | 0.50 |
| Dawid-Skene | 1.00* | 0.63* | 0.57 | 0.50 | 0.98* | 1.00* | 0.98* | 0.50 |
| GPT-2 perplexity | 0.82* | 0.86* | 0.89* | 0.53 | 0.87* | 0.95* | 0.98* | 0.49 |
| Ours (layout) | 0.99* | 1.00* | 0.31 | 0.69* | 1.00* | 1.00* | 1.00* | 0.49 |

## 敏感度：注入 1 个错误的检出率（LinES train+dev (fresh)；* = 95% CI 下界 > 0.5）

| 方法 | merge_terminal | merge_other_par | merge_other_inpar | split_terminal | split_punct | split_nopunct | shift | control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Punctuation rule | 1.00* | nan | 0.00 | 0.00 | 1.00* | 1.00* | 0.96* | 0.50 |
| Length prior | 1.00* | nan | 1.00* | 0.00 | 0.00 | 0.00 | 0.51 | 0.51 |
| Masked LM (PuncEv) | 0.93* | nan | 0.32 | 0.24 | 1.00* | 1.00* | 0.98* | 0.51 |
| SaT-12l-sm output as reference | 1.00* | nan | 0.56 | 0.39 | 0.99* | 1.00* | 0.98* | 0.50 |
| SaT-12l-sm probabilities | 1.00* | nan | 0.48 | 0.39 | 1.00* | 1.00* | 1.00* | 0.51 |
| Consensus | 1.00* | nan | 0.47 | 0.20 | 0.99* | 1.00* | 0.98* | 0.50 |
| DocAgree | 0.50 | nan | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| Ours (flat) | 1.00* | nan | 0.51 | 0.23 | 1.00* | 1.00* | 1.00* | 0.56 |
| Dawid-Skene | 1.00* | nan | 0.47 | 0.20 | 0.99* | 1.00* | 0.98* | 0.50 |

