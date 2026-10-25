# BERT 双向上下文 vs Qwen 左侧上下文

除概率来源外，GUM 划分、train Platt calibration、候选扰动、Soft F1 公式与评价代码完全相同。

| Probability source | 全局 Spearman | 文档内 Spearman | Pairwise accuracy | Gold 胜率 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B-Base left-context | 0.211 | 0.800 | 0.844 | 0.950 |
| BERT-base-uncased bidirectional | 0.239 | 0.877 | 0.894 | 0.996 |

## Gold 对各扰动的胜率

| Source | Over 10/20% | Under 10/20% | Shift 10/20% | Mixed 10/20% |
|---|---:|---:|---:|---:|
| Qwen left-context | 0.933/0.933 | 0.867/0.967 | 1.000/1.000 | 0.933/0.967 |
| BERT bidirectional | 1.000/1.000 | 0.967/1.000 | 1.000/1.000 | 1.000/1.000 |

## 结论

BERT 双向概率在本轮 优于 Qwen 左侧概率；pairwise accuracy 差值为 +0.050。
BERT 在候选位置使用右侧词汇和句法信息；由于本模型为 uncased，它不能利用大小写。双向信息属于本实验要测的优势，但与在线 causal 评价可用的信息不同。
