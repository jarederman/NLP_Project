# 系统分歧位置分析（GUM train，logo）

只看 sat 与另一系统恰有一方切分的空隙。“被奖励”= 证据 p 超过放置该边界的系统自身 Soft F1 的一半（此时 Soft F1 因这个边界而上升）。

| 配置 | 对比 | 只有谁切 | 左侧上下文 | 空隙数 | 其中 Gold | 平均 p | 被奖励 |
|---|---|---|---|---:|---:|---:|---:|
| platt | sat–punkt | punkt | terminal | 224 | 3.6% | 0.388 | 42.4% |
| platt | sat–punkt | sat | non_terminal | 1170 | 76.5% | 0.241 | 27.2% |
| platt | sat–punkt | sat | terminal | 166 | 92.8% | 0.865 | 92.2% |
| platt | sat–stanza | sat | non_terminal | 842 | 69.8% | 0.203 | 22.2% |
| platt | sat–stanza | sat | terminal | 73 | 35.6% | 0.584 | 68.5% |
| platt | sat–stanza | stanza | non_terminal | 26 | 23.1% | 0.140 | 15.4% |
| platt | sat–stanza | stanza | terminal | 25 | 32.0% | 0.705 | 80.0% |
| platt | sat–wtp | sat | non_terminal | 450 | 71.3% | 0.247 | 29.1% |
| platt | sat–wtp | sat | terminal | 574 | 94.3% | 0.656 | 83.4% |
| platt | sat–wtp | wtp | non_terminal | 921 | 7.7% | 0.051 | 4.2% |
| platt | sat–wtp | wtp | terminal | 256 | 4.3% | 0.362 | 36.7% |
| B+C+K+A+L+O/hgb | sat–punkt | punkt | terminal | 224 | 3.6% | 0.183 | 17.0% |
| B+C+K+A+L+O/hgb | sat–punkt | sat | non_terminal | 1170 | 76.5% | 0.472 | 51.0% |
| B+C+K+A+L+O/hgb | sat–punkt | sat | terminal | 166 | 92.8% | 0.887 | 95.2% |
| B+C+K+A+L+O/hgb | sat–stanza | sat | non_terminal | 842 | 69.8% | 0.366 | 37.9% |
| B+C+K+A+L+O/hgb | sat–stanza | sat | terminal | 73 | 35.6% | 0.584 | 65.8% |
| B+C+K+A+L+O/hgb | sat–stanza | stanza | non_terminal | 26 | 23.1% | 0.314 | 30.8% |
| B+C+K+A+L+O/hgb | sat–stanza | stanza | terminal | 25 | 32.0% | 0.723 | 80.0% |
| B+C+K+A+L+O/hgb | sat–wtp | sat | non_terminal | 450 | 71.3% | 0.427 | 46.9% |
| B+C+K+A+L+O/hgb | sat–wtp | sat | terminal | 574 | 94.3% | 0.936 | 95.8% |
| B+C+K+A+L+O/hgb | sat–wtp | wtp | non_terminal | 921 | 7.7% | 0.108 | 6.8% |
| B+C+K+A+L+O/hgb | sat–wtp | wtp | terminal | 256 | 4.3% | 0.170 | 14.8% |
