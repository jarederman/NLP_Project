# E5：候选的全新测试集（结构核查，2026-11-19）

作业 23694637（`scripts/fetch_test_candidates.slurm`）。从 UD r2.18 下载，数据在 `data/UD_candidates/`（gitignored，含 SHA256SUMS）；
统计结果在 `results/e5_candidates.json`。**只统计结构**（文档、段落、句子、词数、领域前缀），没有运行任何切分器或证据，也没有计算任何与切分质量有关的量，
所以对测试集的选择不会受“难不难”影响。

| 树库 | 文档（`# newdoc`） | 段落（`# newpar`） | 句子 | 每篇句数中位数 | 领域 | 训练数据重叠 | 能否作文档级测试 |
|---|---:|---:|---:|---:|---|---|---|
| **EWT dev** | 318（278 篇 ≥ 2 句） | 750 | 2001 | 4 | reviews 192、answers 61、newsgroup 36、email 15、weblog 14 | Stanza 训练用 EWT train（dev 可能用于选模型）；SaT-sm 只用 GUM | **能**；与 EWT test 同领域（网页文本） |
| PUD | 397（298 篇 ≥ 2 句） | 0 | 1000 | 2 | 新闻 215、维基 182 | **Stanza 的 combined 分词模型用整个 PUD test 训练** | 勉强：文档很短，对 Stanza 被污染 |
| LinES train + dev | 9 + 9 | 0 | 3457 + 1118 | 363 / 120 | 文学、技术文档（与 LinES test 同源） | 无 | 能，但不是新领域；可把 LinES 的结论从 9 篇扩到 27 篇 |
| ParTUT | 0 | 0 | 2090 | — | 法律、维基、新闻 | 无 | **不能**：没有文档标记 |
| ESLSpok | 0 | 0 | 2320 | — | 学习者口语 | 无 | **不能**：没有文档标记 |
| CTeTex test | 196 | 0 | 276 | 1 | ? | — | **不能**：几乎都是单句 |

Stanza 的组成来自其源码 `stanza/utils/datasets/prepare_tokenizer_treebank.py`（`build_combined_english_dataset`：
EWT/GUM/GUMReddit train + PUD/Pronouns test；ParTUT 和 LinES 只在 POS 的附加列选项中出现）。SaT-sm 的英语 UD 数据是 UD_English-GUM
（`wtpsplit/data/language_info.csv`）。

**建议**：用 **EWT dev** 做最终评估器的一次性测试（无换行和带版式两种条件都能测；318 篇，5 个领域）。说明两点：它与已用过的 EWT test 同领域，
所以检验的是“新文档”而不是“新领域”；EWT 恰好是共识胜过证据的领域，所以这次测试也能直接检验“证据 vs 共识”的结论。
可选的第二个测试：LinES train + dev（检验 LinES 的失败是否在更多文档上成立）。需要先写预注册（判定规则、指标、对照组），再跑。
