# Reference-Free Sentence Segmentation Evaluation

本项目在无待评文档 gold boundary 输入的条件下，用语言模型在候选间隙处的句末证据（初步研究为 Qwen3.5 next-token 句末标点概率，真实系统研究为 BERT `[MASK]` 概率）加 Platt calibration 和 Soft F1，为真实 sentence splitter 打分、排序与选择。GUM test 从不读取；EWT test 只在全部配置冻结后运行一次。实验结果汇总见下文“实验记录”。

## 项目结构

- `src/data.py`：CoNLL-U 文档级读取、`SpaceAfter=No` 处理、受控去句末标点文本与 boundary alignment。
- `src/lm.py`：`Qwen3_5ForCausalLM` BF16 forward、token/character alignment、候选位置 logits 和文档级缓存。
- `src/candidates.py`：固定种子的 Gold/Over/Under/Shift/Mixed 候选。
- `src/scoring.py`：raw/calibrated BCE、weighted BCE、Soft F1/Fβ、句长 baseline、缩写规则。
- `src/evaluate.py`：Spearman、同文档 pairwise ranking、gold-win、类型/genre 汇总和实例选择。
- `src/contracts.py`：将原始文档、Gold boundary offsets 和 splitter 输出分为独立 JSONL artifacts，并用文本 SHA256 防止错位或 Gold 泄漏。
- `src/selection.py`：真实系统的 Macro/Micro 对齐聚合、Direct Internal threshold、pairwise ranking、Selection Regret 和 Always-SaT headroom。
- `src/meta_eval.py`：Gold-win、pairwise、按领域选择/Regret/Kendall τ、Direct Internal 与分层 bootstrap（dev、train CV、EWT 共用）。
- `src/train_crossfit.py`、`src/analyze_evidence_gaps.py`：P1 train 交叉拟合开发集及其证据误差分析。
- `src/quality_agreement.py`：跨文档质量一致性（主指标：全局 Spearman；按 |ΔGold F1| 分档的配对准确率）。
- `src/boundary_probe.py`、`src/boundary_probe_eval.py`、`src/layout_data.py`：`[MASK]` 隐状态探针、插入似然比与版面控制变量（修订 3）。
- `src/ewt_test.py`：冻结方法的 EWT 一次性测试（修订 4）。
- `paper/draft.md`：论文草稿（中文）。
- `tests/`：不依赖 GPU 的数据契约与 selection-metric 单元测试。
- `results/`：正式输出；`logs/`：Slurm stdout/stderr。

## 存储与环境

所有命令先 source：

```bash
source /scratch/wj2301/NLP_Project/scripts/env.sh
```

该脚本把 `HOME`、HF、datasets、XDG、pip、Torch、Triton、扩展编译、CUDA、Numba、Matplotlib、Python bytecode 及临时目录全部重定向到本项目的 scratch 路径。复用环境为 `/scratch/wj2301/conda_envs/jit`（PyTorch 2.10 / Transformers 5.7），不修改参考仓库及其环境。

## 复现

按顺序提交且仅在 smoke 成功后运行正式实验：

```bash
cd /scratch/wj2301/NLP_Project
mkdir -p logs
smoke_job=$(sbatch --parsable scripts/run_smoke.slurm)
# 检查 logs/smoke_${smoke_job}.out 中 status=PASS 后：
sbatch scripts/run_formal.slurm
```

作业内部会下载官方 `UD_English-GUM` 到 `data/UD_English-GUM`，并优先加载 `Qwen/Qwen3.5-9B-Base`；只有该仓库发生实质性加载失败时才回退 `Qwen/Qwen3.5-9B`。模型下载到 `.cache/huggingface/hub`。LM 结果按文档缓存到 `data/lm_cache/`，候选扰动和指标重算不会重复执行 Qwen。

## 输出语义

`raw_punctuation_probability` 是精确单 token `.?!。？！` 的 next-token 总概率，并非真实 boundary probability。`calibrated_boundary_probability` 是用 GUM train gold 拟合、对 dev 冻结的 Platt 映射，也不能视为跨域完美概率。受控文本使用 gold 删除独立句末 `.?!` 只为构造 benchmark；评分候选时不读取 dev gold。词内句末缩写无法无损删除其最后一个点，这是当前实验明确记录的残余泄漏风险。

正式产物为 `results/metrics.csv`、`results/examples.jsonl`、`results/summary.md`，另含 `aggregate.json`、`run_metadata.json`、`calibration.json` 和 `smoke_test.json`。

已有 LM cache 时，可以只在 CPU 上重新生成指标和报告，不会再次加载 Qwen：

```bash
source scripts/env.sh
/scratch/wj2301/conda_envs/jit/bin/python -m src.analyze_cached
```

本次实际运行的 Job ID、日志和状态记录在 `results/job_ids.json`。

## Final-study infrastructure

Proposal 后的核心实验将 GUM 固定到 UD commit `1fe635509c649e376dfb449d528424ab78f4eaee`（197 篇 Train、30 篇 Dev 文档）。GUM train/dev 和 EWT test 分别由下列脚本下载；每个目录同时保存 `UD_RELEASE` 和 `SHA256SUMS`，不再跟随浮动的 `master`：

```bash
source scripts/env.sh
scripts/fetch_gum.sh
scripts/fetch_ewt.sh
```

最终研究将原始输入、Gold 和系统输出分开保存。Splitter 与 evaluator 只能读取 `RawDocument` artifacts；Gold offsets 由最终 meta-evaluation 阶段读取，并通过 `doc_id` 与文本 SHA256 对齐。

无需模型或 GPU 即可验证新数据契约和聚合指标：

```bash
python3 -m unittest discover -s tests -v
```

`src.selection.selection_report` 以 Macro Selection Regret 为主要决策指标，并单独返回 pooled Micro sensitivity；Macro score 不会与 Micro Gold F1 混合比较。Minimum study 的候选系统为 Punkt、Stanza、WtP 和 SaT，关键部署基线为 Direct Internal Splitter 与 Always-SaT。

## GUM 真实 Splitter 实验（Step 1 + Step 2）

本轮固定使用上述 UD commit 的 GUM Train/Dev，并且不读取 Test。`src.surface_data` 从
CoNLL-U surface token、MWT 与 `SpaceAfter=No` 重建完整文档；该路径忽略 `# text`、
`# sent_id` 和句间空行。`src.materialize_dev_gold` 只会在无参考分数冻结后生成
Dev Gold artifact，随后仅 `src.analyze_real_splitters` 会加载它。

评分协议保持 `google-bert/bert-base-uncased` + Platt calibration + 原 Soft F1 不变。
若候选 gap 前已经存在 `.?!`（包括引号内的终止标点），BERT query 替换已有标点；
否则在 gap 插入 `[MASK]`，避免询问“是否需要第二个标点”。真实候选系统为 Punkt、
Stanza、`wtp-bert-mini` 和无 UD LoRA 的 `sat-3l-sm`。

v1.1 起（`configs/real_splitters.json` 中 `protocol_version=gum-real-splitters-v1.1`），终止标点后紧跟的
引号/括号/引用标记 `[n]` 视为同一个终止单元，证据放在整个单元之后的 gap；单元内部的 gap 按插入查询。
Splitter 切在非空白 gap 上的边界（token 内部，如邮件地址；或无空格相邻 token 之间，如 `al.|:`）
不参与 Soft F1，但在 Gold 元评测中计为该系统的 false positive。

代码已稳定，正式作业直接提交（脚本第一步运行单元测试，失败则不覆盖任何产物）：

```bash
sbatch scripts/setup_real_environment.slurm      # 仅首次
formal=$(sbatch --parsable scripts/run_real_formal.slurm)    # GUM dev；同时生成 train BERT cache
sbatch --dependency=afterok:$formal --kill-on-invalid-dep=yes scripts/run_train_cv.slurm   # P1
```

新增 Python 包和 NLTK/Stanza/Hugging Face 模型全部位于项目 `.venv/` 与 `.cache/`，不会写入 `$HOME`。

- GUM dev 产物：`results/real_splitter_per_doc.csv`、`real_splitter_summary.csv`、`real_splitter_ranking.{md,json}`、
  `real_splitter_examples.jsonl`、`alignment_failures.jsonl`、`real_calibration.json`、`real_run_metadata.json`。
- P1 产物：`results/train_cv/{summary.md,report.json,per_domain.csv,evidence_by_context.{md,json}}`；
  中间产物（gold-free 冻结分数、fold 划分、每折校准参数）在 `data/real_system/train_cv/`。
- 换协议重跑前先归档：v1 结果在 `results/real_v1/`，v1 数据产物在 `data/real_system_v1/`（未纳入 git）。

### P1：GUM train 5 折交叉拟合（开发集）

GUM dev 每个 genre 仅 2 篇且已在诊断中看过，因此开发与选型改在 GUM train（197 篇、15 个 genre）上做
按 genre 分层的 5 折交叉拟合：第 k 折的 Platt 只用其余各折 Gold 拟合，再给第 k 折打分；全部 gold-free
分数写盘冻结后才做元评测（`src/train_crossfit.py`，元评测逻辑在 `src/meta_eval.py`，dev/EWT 共用）。
置信区间为按领域分层、按文档重采样的 bootstrap（2000 次，95% 百分位）。

成功标准（事先确定，可按结果弹性调整）：Gold-win 率过半；L1 pairwise accuracy CI 下界 > 0.5；
L2 领域平均 Kendall τ > 0；L3 领域平均 Selection Regret 低于 Always-SaT。

## 实验记录

> **勘误（2026-11-19）**：修订 6–9 的排序统计（全局 Spearman 等）已用平局精确化重算，`results/a6`…`a9` 中为新值；
> 下文较早段落里的数字是勘误前的值，除下面标注“勘误后”的几处外未逐一改写。变化与显著性翻转的完整列表见本文“勘误：平局精确化”一节
> 和 `results/erratum_ties/summary.md`。所有预先写定的判定都不变。

### GUM dev（已在 v1 诊断中看过，仅作参考，不再用于调参）

| 指标 | v1 | v1.1 |
|---|---:|---:|
| Pairwise accuracy | 0.541 | 0.616 |
| 语料级 Soft F1 选择（Gold oracle 均为 SaT） | Punkt | Stanza |
| Macro Selection Regret | 0.0589 | 0.0158 |
| Micro Selection Regret | 0.0432 | 0.0087 |
| Direct Internal（p ≥ 0.5）Macro Gold F1 | 0.826 | 0.848 |
| Gold-win | 5/30 | — |

四系统 Gold Macro F1（v1/v1.1 相同）：SaT 0.9720、Stanza 0.9561、Punkt 0.9131、WtP 0.9094。
v1 → v1.1 的唯一变化是终止单元包含引用标记 `[n]`（v1 诊断发现的真实 bug）。
v1 诊断：约 10% 的 Gold 边界（标题、列表项、破折号、引用）左侧没有可见终止标点，证据很弱。

### P1 train 交叉拟合（v1.1，job 21776943，2026-11-04）

| 层次 | 指标 | 估计 [95% CI] | 标准 | 是否达标 |
|---|---|---|---|---|
| 前提 | Gold-win 率 | 0.162 [0.112, 0.213] | > 0.5 | 否 |
| L1 | Pairwise accuracy | 0.629 [0.596, 0.662] | 下界 > 0.5 | 是 |
| L2 | 领域平均 Kendall τ | 0.289 [0.156, 0.378] | > 0 | 是 |
| L3 | 领域平均 Selection Regret | 0.0277 [0.0226, 0.0381]（Always-SaT 0.0014） | 低于 Always-SaT | 否 |
| L3 | 领域平均（所选 − Always-SaT） | −0.0263 [−0.0367, −0.0210] | > 0 | 否 |

15 个领域中仅 news、textbook 选中 SaT，9 个领域选 Punkt。Direct Internal 0.831，低于四个系统中的任何一个。
评分函数对照：neg BCE 更差（pairwise 0.560，τ 0.04）；归一化 Soft F1 与原版几乎相同（regret 0.0257），只作辅助指标。

误差分析（`results/train_cv/evidence_by_context.md`）：左侧有终止标点的 gap 中 94.3% 是 Gold，
但两参数 Platt 只给 Gold 平均 p=0.770，16.2% 低于 0.5；无标点 Gold（标题、列表项，占 6.1%）平均 p=0.293。
由于 Soft F1 下一个边界只有在 p > F/2 时才划算，切得更全的 SaT 被系统性低估。

证据强度上限诊断（`src/evidence_ceiling.py` → `results/train_cv/evidence_ceiling.md`；假想证据由 train Gold 构造，
**仅作分析，不能用于选型**）。候选与 Soft F1 不变，只替换证据：

| 情景 | 证据自身 Direct Internal F1 | Gold-win | Pairwise | 领域平均 τ | 所选 − Always-SaT | 选 SaT 的领域 |
|---|---:|---:|---:|---:|---:|---:|
| 实际证据 | 0.831 | 0.162 | 0.629 | 0.289 | −0.0263 | 3/15 |
| 只把带终止标点的 gap 换成 Gold | 0.919 | 0.305 | 0.674 | 0.422 | −0.0206 | 2/15 |
| 只把无终止标点的 Gold 边界换成 1 | 0.879 | 0.777 | 0.845 | 0.711 | **+0.0011** | 14/15 |
| 无终止标点的 gap 上 logit shift 3 | 0.889 | 0.614 | 0.823 | 0.844 | −0.0022 | 11/15 |
| 证据 = Gold | 1.000 | 1.000 | 0.996 | 0.978 | +0.0014 | 12/15 |

结论：决定选择结果的是**无终止标点的真实边界**（标题、列表项、破折号等，约占 Gold 的 10%），
而不是带标点 gap 上的校准偏差。证据本身不需要比 SaT 更会切分：只修这一类时证据的 Direct Internal F1
仅 0.879（低于 SaT 的 0.971），选择却已优于 Always-SaT。

### P2：BERT 证据 + 不依赖 Gold 的 NLP 特征（GUM train，2026-11-06）

`src/gap_features.py` 为每个候选空隙计算只依赖原文的特征组：B（BERT logit、mask 模式）、C（左侧上下文类别及其与
logit 的交互）、K（左右词形）、A（缩写线索，Kiss & Strunk 式文档内类型统计）、L（距相邻终止单元的距离、左右片段
标题式大写比例）、O（Punkt 式文档内正字法统计）、W（左右词身份，训练折内目标编码）；V（候选系统投票）只作对照。
`src/calibration_search.py` 对 22 个“特征组 × 模型（逻辑回归 / 梯度提升树）”配置，分别在 P1 的 5 折和**留一领域
（LOGO）**下交叉拟合，Soft F1 与候选不变。选型规则在实验前写入 `results/calibration_search/PREREGISTRATION.md`，
以 LOGO 为准。CPU 作业：`sbatch scripts/run_calibration_search.slurm`（约 4 分钟）。完整表：
`results/calibration_search/summary.md`（第一轮另存为 `wave1_summary.md`）；分歧位置分析：`disagreements.md`。

| 配置（LOGO） | 所选 − Always-SaT [95% CI] | Gold-win | Pairwise | 领域平均 τ | 选 SaT 的领域 | 证据 DI F1 |
|---|---:|---:|---:|---:|---:|---:|
| platt（P1 基线） | −0.0263 [−0.0370, −0.0210] | 0.168 | 0.628 | 0.289 | 3/15 | 0.831 |
| B（加 mask 模式） | −0.0392 | 0.071 | 0.502 | 0.044 | 0/15 | 0.918 |
| B+C+K+A+L（逻辑回归） | −0.0179 | 0.173 | 0.668 | 0.467 | 3/15 | 0.936 |
| C+K+A+L/hgb（无 LM） | −0.0173 | 0.274 | 0.720 | 0.556 | 3/15 | 0.943 |
| **B+C+K+A+L/hgb（按规则选中）** | −0.0151 [−0.0230, −0.0112] | 0.310 | 0.734 | 0.578 | 4/15 | 0.946 |
| B+C+K+A+L+O/hgb（主指标最高） | −0.0136 [−0.0186, −0.0087] | 0.320 | 0.762 | 0.689 | 4/15 | 0.951 |
| B+C+K+A+L+V（投票，不参与选型） | −0.0022 [−0.0064, +0.0015] | 0.259 | 0.845 | 0.800 | 12/15 | 0.966 |

要点：
- 特征组合相对 Platt 的提升显著（配对 CI 全在 0 以上），L1/L2 大幅改善；LOGO 与 5 折结果接近，说明跨领域可迁移。
  但 L3 仍未超过 Always-SaT。
- **更好的校准不等于更好的选择**：只加 mask 模式（B）让证据整体更准（DI F1 0.918），却把无标点 Gold 的平均 p 压到 0.04，选择变得更差。
- 加上特征后，uncased BERT 的边际贡献很小（无 LM 配置与之不可区分）。
- 瓶颈在 SaT 比 Punkt 多切的 1170 个无标点空隙：其中 76.5% 是 Gold，但最佳配置只奖励了 51%（Platt 为 27%）。
- **注（修正案 2）**：这些提升主要来自大小写特征（K、O、L 的 Title-Case 比例），与“不让证据看到大小写”的设计意图冲突，
  因此只作分析，不作为方法；不含大小写的 B+C+A+L 不优于 Platt。

### 主目标修正：跨文档质量一致性（修正案 2，2026-11-06/07）

主目标改为“指标能跨领域、跨文档识别并比较分句质量，尤其是质量差距大时”；选出更好的 splitter 只是副任务。
主指标：全部（文档, 系统）条目上分数与 Gold F1 的**全局 Spearman**；辅助：全部配对（含跨文档）准确率按 |ΔGold F1|
分箱，重点看 ≥0.05（大差距配对）；（领域 × 系统）单元 Spearman；文档内配对 / τ / regret 作为次要诊断。CI 为领域内
分层的文档级 bootstrap（1000 次），评分函数或证据之间做配对差。实现：`src/quality_agreement.py`，输出
`results/quality_agreement/<name>/`。

**归一化对照**（证据、候选、Gold 不变，只换打分方式；配对差 = 归一化 − 原版 Soft F1）：

| 数据（证据） | Soft F1 全局 Spearman | 归一化 | 配对差 [95% CI] | 大差距配对 |
|---|---:|---:|---:|---:|
| 真实系统，GUM train 5 折（BERT，788 条目） | 0.118 | **0.354** | +0.235 [+0.181, +0.286] | 0.594 → 0.713 |
| 合成扰动，GUM dev（BERT，已看过，仅参考） | 0.239 | **0.442** | +0.203 [+0.114, +0.299] | 0.632 → 0.752 |
| 合成扰动，GUM dev（Qwen 左侧上下文） | 0.211 | 0.217 | +0.006 [−0.040, +0.043] | — |

归一化不改变文档内排序（文档内配对 0.629 不变），但显著提升 BERT 证据的跨文档可比性；Qwen 证据上无效（原因未查）。
合成部分：`src/normalization_synthetic.py`。

**BERT 上下文对照**（job 22107999，2026-11-07；`src/conditioned_bert.py` + `src/conditioned_eval.py`，
`scripts/run_conditioned_context.slurm`）。每个空隙的查询完全相同，只改变其他空隙的渲染：`raw` 原文；
`raw_plus_candidate` 原文 + 在候选切分但无终止标点处加 “.”；`candidate_only` 终止标点完全跟随候选（切处保留/补
“.”，不切处删除）。非终止标点始终保留。每种条件各自 5 折交叉拟合 Platt。raw 条件与冻结缓存逐值一致
（max diff 0.0）。

| 上下文 | 全局 Spearman（归一化） | 配对差 vs raw | 大差距配对 | 配对差 vs raw | 文档内配对 |
|---|---:|---:|---:|---:|---:|
| raw | **0.354** | — | **0.713** | — | **0.629** |
| raw_plus_candidate | 0.318 | [−0.060, −0.011] | 0.685 | [−0.043, −0.012] | 0.582 |
| candidate_only | 0.275 | [−0.117, −0.040] | 0.654 | [−0.084, −0.034] | 0.566 |

原版 Soft F1 下结论相同（0.118 / 0.075 / 0.049，配对差 CI 均在 0 以下）。自我印证检查（校准后平均 p，候选切了但非
Gold）：raw 0.240 → raw_plus_candidate 0.452 → candidate_only 0.458；其余三格基本不变。即候选条件化主要让证据
附和候选自身的错误边界，降低区分度。成本：GPU 抽取 268 s（197 篇 × 9 次渲染，约为 raw 单次的 8–9 倍），整作业
11 分钟。结论：保留 raw 原文上下文。

**标点 vs 边界：探针 / 似然比 / 版面控制变量**（修订 3；job 22259665，2026-11-08；`src/boundary_probe.py`、
`src/boundary_probe_eval.py`、`src/layout_data.py`，`scripts/run_boundary_probe.slurm`；完整表
`results/boundary_probe/summary.md`）。冻结 BERT 与 v1.1 查询不变；方法 1 = `[MASK]` 处第 8/12 层隐状态上的 L2 逻辑回归
探针（λ=1e-3）；方法 2 = 插入 “.” 前后左 2 / 右 3 个 token 的掩码对数概率差；`para` = 段落分隔（`# newpar`）以 `[SEP]`
写入输入（控制变量，只作分析；GUM 的段落分隔全是 Gold，占 Gold 39.2%、占无终止标点 Gold 86.1%）。

| 配置（LOGO，归一化） | 全局 Spearman | 配对差 vs raw/platt | 大差距配对 | 文档内配对 | 无标点 Gold 平均 p | 带标点非 Gold 平均 p | 所选 − 永远 SaT | Gold-win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw/platt | 0.355 | — | 0.714 | 0.628 | 0.249 | 0.641 | −0.0263 | 0.162 |
| raw/lr | 0.416 | [+0.026, +0.097] | 0.746 | 0.633 | 0.295 | 0.628 | −0.0288 | 0.183 |
| raw/probe | 0.559 | [+0.142, +0.272] | 0.832 | 0.814 | 0.548 | 0.204 | −0.0020 | 0.523 |
| **raw/probe+lr（按规则选中）** | **0.574** | [+0.156, +0.280] | 0.841 | 0.826 | 0.561 | 0.203 | +0.0003 | 0.553 |
| para/platt | 0.605 | [+0.199, +0.304] | 0.851 | 0.832 | 0.796 | 0.627 | −0.0002 | 0.695 |
| para/probe+lr | 0.735 | [+0.309, +0.445] | 0.923 | 0.925 | 0.906 | 0.161 | +0.0011 | 0.807 |

要点：探针同时修复无标点真边界与缩写类假边界；5 折结果方向一致（raw/probe+lr 0.621）；λ 敏感性不改变结论；选择类
指标为点估计（不在预注册判据内）。para 下非段落分隔的无标点 Gold 平均 p 反而降低（0.348 vs raw 0.514）。raw 抽取与冻结
缓存：中位差 0，632 / 168917 个空隙差值 > 0.01（bf16 批次差异）。成本：整作业 13 分 49 秒；GPU 非空闲时平均利用率 98.1%。

**方法冻结与 EWT test**（修订 4，2026-11-08）。用户决定主方法为**不带版面信息的 `raw/probe+lr`**，`para` 只作补充实验。
最终模型在全部 197 篇 GUM train 上拟合一次（λ=1e-3），EWT test（UD English EWT r2.18，316 篇、5 个领域：answers、
email、newsgroup、reviews、weblog）只运行一次：原文重建 → 四个切分器 → 证据 → 写盘冻结分数 → 之后才读 EWT Gold →
分析；版面产物（由句首 `# newpar` 推出）在主方法分数冻结后才生成。主判据：归一化 Soft F1 全局 Spearman，
`raw/probe+lr` 相对 `raw/platt` 的配对差 CI 全在 0 以上。作业：`sbatch scripts/run_ewt_test.slurm`（`src/ewt_test.py`；
job 22337008），输出 `results/ewt_test/{summary.md,report.json,data_manifest.json}`，中间产物在 `data/real_system_ewt/`。

**EWT test 结果**（一次性；GPU 作业 22337008 冻结全部分数后在物化 Gold 时停止，按修订 4a 处理 12 个无空白位置的
Gold 边界后，CPU 作业 22340933 只补跑 Gold 物化与分析；`scripts/run_ewt_analysis.slurm`；完整表 `results/ewt_test/summary.md`）。
EWT：316 篇、1761 个 Gold 边界（其中 524 个左侧无终止标点，占 29.8%，GUM 为 9.8%）。系统 Gold Macro F1：SaT 0.897、
Stanza 0.878、Punkt 0.827、WtP 0.796。

| 配置（归一化 Soft F1） | 全局 Spearman [95% CI] | 配对差 vs raw/platt | 大差距配对 | 文档内配对 | GUM LOGO 参照 |
|---|---:|---:|---:|---:|---:|
| raw/platt | 0.338 [0.249, 0.416] | — | 0.674 | 0.606 | 0.355 |
| raw/probe | 0.394 | +0.056 [+0.011, +0.107] | 0.705 | 0.709 | 0.559 |
| **raw/probe+lr（主方法）** | **0.411 [0.312, 0.497]** | **+0.074 [+0.023, +0.126]** | 0.713 | 0.711 | 0.574 |
| para/probe+lr（补充） | 0.468 [0.374, 0.561] | +0.131 [+0.083, +0.184] | 0.746 | 0.751 | 0.735 |

- **主判据成立**：配对差 CI 全在 0 以上。大差距配对（+0.014 到 +0.065）与文档内配对（+0.060 到 +0.148）的配对差也显著；
  领域×系统 Spearman（只有 20 个单元）无显著差异（0.627 → 0.605）。五个领域的点估计全部提升（reviews 最弱，0.250 → 0.313）。
- 跨语料增益约为 GUM 上的三分之一（+0.074 vs +0.219），基线几乎不变（0.338 vs 0.355）：探针学到的一部分是 GUM 的标注习惯。
- 探针在 EWT 上仍同时修复两类错误：无终止标点 Gold 平均 p 0.221 → 0.587，带终止标点非 Gold 0.715 → 0.379。
- 选择（次要）：raw/probe+lr 的领域平均 所选 − 永远 SaT 为 −0.0016 [−0.0110, +0.0074]（platt −0.0440），Gold-win 0.725；
  语料级三种配置都选 SaT（= oracle）。
- 成本：GPU 抽取每种输入约 27 秒；GPU 作业全程平均利用率 27.7%（EWT 小，作业时间主要花在切分器与模型加载），非空闲 89.6%。

**无参考基线对照**（修订 5；CPU 作业 22344595 / 22344845；`src/baselines.py`、`scripts/run_baselines.slurm`；完整表
`results/baselines/summary.md`）。B1 单系统伪参考、B2 留一共识（其余三系统多数票）、B3 启发式（标点规则伪参考、句长）；
EWT 上为测试后补加的比较，主方法不变。

| 全局 Spearman（归一化） | GUM train（LOGO） | EWT test |
|---|---:|---:|
| 主方法 probe+lr | 0.574 | 0.411 |
| B2 共识 | **0.733**（+0.159 [+0.085, +0.239]） | **0.657**（+0.246 [+0.129, +0.368]） |
| B1 伪参考 = SaT（去掉 SaT 自身条目，与主方法同子集 0.604 / 0.423） | 0.882 | 0.718 |
| B1 伪参考 = Punkt / WtP（同上） | 0.553 / 0.583 | 0.487 / 0.464 |
| B3a 标点规则伪参考 | 0.342 | 0.392 |
| 主方法 + 共识（秩平均，事后） | 0.751 | 0.661 |

要点：共识类基线在全局 Spearman 上明显强于主方法；但主方法 (1) 文档内配对显著更好（GUM 0.826 vs 0.727）或持平（EWT）；
(2) **能认出最好的系统**：语料级排名 SaT 第一，领域级选 SaT 12/15（GUM）、4/5（EWT），领域平均 regret 0.0011 / 0.0018，
而共识把 Stanza 排第一，只在 1/15、1/5 个领域选 SaT，regret 0.020 / 0.015（共识惩罚“唯一正确”的系统）；
(3) 所有系统一致的错误占系统错误的 12.7% / 28.3%，共识对此失明，主方法对其中 37% / 40% 的共同漏切 Gold 边界给出 p > 0.5；
(4) 秩平均相对共识：全局 Spearman 无显著变化，文档内配对 +0.098 [+0.075, +0.120]（GUM）/ +0.068 [+0.026, +0.105]（EWT）。

**分解：文档之间 vs 文档之内**（事后分析，修订 5 补充 (7)；`src/baseline_decomposition.py`；`results/baselines/decomposition.md`）。
Gold F1 方差约一半来自文档之间（GUM 50.5%，EWT 48.0%）。文档之间 Spearman：主方法 0.544 / 0.582，共识 0.893 / 0.884，
标点规则 0.814 / 0.885；文档之内：主方法 0.817 / 0.404，共识 0.579 / 0.555，秩平均 0.769 / 0.623（GUM / EWT）。
即一致性类信号赢在“认出困难文档”，证据赢在 GUM 的文档内排序；论文叙述据此调整为“证据 vs 一致性”的互补比较（草稿 v0.3）。

**强证据 + 其他信号的组合**（修订 6 / 6a，2026-11-12 写定；`src/a6_{data,signals,gpu,combine,test}.py`，
`scripts/run_a6_{dev,test}.slurm`；完整表 `results/a6/{dev,test}_summary.md`）。开发：GUM train 197 篇 LOGO；测试（冻结后
一次）：UD_English-GENTLE r2.18 test（26 篇、8 个体裁，主）与 UD_English-LinES r2.18 test（9 篇，次），EWT 不再复用。
所有证据输入小写。组件：E0 冻结主方法；E1 `sat-12l` 零样本；E2 微调 ModernBERT-large 标注器；规则特征 + KN 三元组特征；
stack = 以上的 L2 逻辑回归；可选半马尔可夫长度先验（+hsmm）。单系统线 S0/S1/S2，多系统线 M0 共识 / M1–M3（Dawid–Skene、
QE 岭回归）。污染核查：Stanza `combined` 训练含 EWT/GUM train，`sat-*-sm` 训练含 GUM train（`sat-12l-sm` 因此只作分析）；
GENTLE / LinES test 对所有候选与证据都干净。

- 作业：开发 22938779 在微调阶段因 Transformers 5.7 不再接受 `reference_compile` 失败（代码修复，设置不变），重跑
  23036839 用时 21 分钟（微调 ModernBERT 15 折 + 最终模型共 7 分钟）；测试 23042729 用时 10 分钟，`gold_file_opened: false`
  下冻结分数后才物化 Gold（GENTLE 0 个、LinES 5 个无空白 Gold 边界，按 4a 处理）。
- 选型（开发集冻结）：证据 **stack**，S* = **S1**，M* = **M3**；开发判据与公平性全部成立。

| 全局 Spearman（归一化） | GUM 开发（LOGO） | **GENTLE test（主）** | LinES test（9 篇） | GENTLE + LinES |
|---|---:|---:|---:|---:|
| S0 冻结主方法（E0） | 0.574 | 0.672 | 0.868 | 0.718 |
| **S1 stack 证据** | 0.818 | **0.883** | 0.815 | 0.874 |
| 　S1 − S0 [95% CI] | +0.244 [+0.186, +0.309] | **+0.212 [+0.101, +0.326]** | −0.053 [−0.177, +0.065] | +0.155 [+0.078, +0.243] |
| M0 留一共识 | 0.733 | 0.830 | 0.861 | 0.872 |
| **M3 QE 岭回归（多系统）** | 0.861 | **0.900** | 0.793 | 0.883 |
| 　M3 − M0 [95% CI] | +0.127 [+0.090, +0.165] | **+0.070 [+0.010, +0.137]** | −0.069 [−0.342, +0.142] | +0.011 [−0.036, +0.059] |

- **主测试（GENTLE）成功**：两条配对差 CI 都在 0 以上；公平性：S1/M3 最大平均秩残差 0.046/0.027，语料级第一都是 SaT（= Gold）。
  期望目标 S* ≥ M0 在开发集和 GENTLE 上都成立（点估计 0.883 vs 0.830）。文档之内 Spearman：S0 0.441 → S1 0.692，M0 0.370 → M3 0.865。
- **组合本身是关键**：GENTLE 上单独的 E2（0.738）与 E0 无显著差异，E1 零样本 0.809，只有 stack 显著优于 E0；证据切分 F1
  E0 0.681 → stack 0.798，系统出错处证据也出错的比例从 0.48–0.69 降到 0.41–0.54。+hsmm 在三个语料上都不带来增益。
- **LinES（次）不成立**：点估计下降但 CI 都跨 0。LinES 只有 9 篇，四个系统 Gold Macro F1 都在 0.93–0.95，可分辨的质量差异小；
  基线 S0 本来就高（0.868）。stack 在“带终止标点但非 Gold”空隙上的平均 p 为 0.765（GUM 0.264，GENTLE 0.188），提示
  LinES 的切分习惯与 GUM 不同（待查，未在预注册内）。
- M1（单一基率 Dawid–Skene）在三个语料上都是标签翻转解（GENTLE −0.085），按 6a 原样报告；M1s 只作分析。

**测试后稳健性分析**（修订 7，2026-11-14，计算前写定，只作分析；`src/a7_posthoc.py`，`scripts/run_a7_{seeds,analysis}.slurm`；
CPU 作业 23134083（7.1–7.3）、GPU 作业 23134081 + CPU 作业 23134084（7.4）；完整表 `results/a7/{posthoc,seeds}_summary.md`）。
GUM 上重算与开发报告逐值一致。

- **7.1 软证据 vs 硬伪参考**（stack 在 p ≥ 0.5 处当参考答案，候选的边界 F1）：S1 − H(stack) 在 GUM / GENTLE / LinES 上为
  +0.008 [−0.008, +0.026] / −0.013 [−0.037, +0.010] / −0.008 [−0.155, +0.171]。证据变强后软打分不再有额外增益，S1 在功能上
  等同于“独立的强参考切分器”；只有弱证据（E0，GUM）时软打分显著更好（+0.066 [+0.040, +0.092]）。
- **7.2 同源偏差**：预登记的三项判定（SaT 在 E1/E1sm 下、Stanza 在 E2 下的秩残差比 E0 高 > 0.05）在 GENTLE、LinES 上都不成立。
  **未登记的观察**：零样本 `sat-12l`（E1）在三个语料上都高估 WtP（SaT 的前身）：+0.108 / +0.096 / +0.077；stack 中降到 −0.009 到 −0.040。
  GENTLE 上单一证据都认错第一名（E1 → WtP；E1sm、E2、E2b → Punkt），只有 stack 与 Gold 一致（SaT）。
- **7.3 小样本敏感性**：GENTLE 上 S1 − S0 不分层 bootstrap +0.212 [+0.069, +0.411]，逐一去掉 8 个领域全为正（+0.143 到 +0.273），
  **稳健**；M3 − M0 不分层 +0.070 [−0.020, +0.188]（去掉单个领域全为正），**按规则不稳健**；S1 − M0 在 GENTLE 为
  +0.053 [−0.016, +0.124]（不分层 [−0.068, +0.176]），合并测试 +0.001，“单系统超过共识”只在 GUM 开发上显著（+0.085 [+0.042, +0.128]）。
- **7.4 种子方差**（E2 的 LOGO 以 3 个额外种子基数重训）：4 个种子都选中 stack，S1 − S0、M3 − M0 的 CI 全在 0 以上，**选型对种子稳定**。
  但 E2 本身很不稳定：0.781 / 0.594 / 0.636 / 0.697（均值 0.677，sd 0.081），原始种子恰为最高；stack 0.804 ± 0.010，M3 0.850 ± 0.008。
  修订 6 开发报告中的 E2 数字偏乐观，stack 偏乐观约 0.014。

**补充有效性实验**（修订 8，2026-11-17，计算前写定，只作分析，纯 CPU；`src/a8_supplement.py`，`scripts/run_a8_supplement.slurm`；
CPU 作业 23521208；完整表 `results/a8/summary.md`）。GUM 上 S1 与开发报告逐值一致。第一次运行（23520243）主步骤限制为 4 线程，
BLAS 浮点差异使 GUM S1 差 1e-4，判定都相同，存档于 `results/a8_omp4_run1/`；正式结果用与修订 6/7 相同的默认线程数重跑。
所有候选只在空白空隙上表示；GENTLE / LinES 为测试后分析。

- **8.1 Gold 作为第 5 个候选**：没有任何评分方法“能认出 Gold”（排第一比例 ≥ 0.8）。GUM 上 Stack 0.399、Probe 0.429、PuncEv 0.101；
  GENTLE 上 Stack 0.426；LinES 上 0.148。Gold 主要输给 SaT（GUM 上 Stack 下 Gold 胜 SaT 仅 0.579）。注意：系统输出离 Gold 很近
  （GUM 上 SaT 与 Gold 只差中位数 2 个空隙，197 篇中 50 篇有系统与 Gold 完全相同，平局记 0.5），Gold 的胜负取决于少数几个空隙，
  而这些空隙正是证据的盲区（见 8.2）。
- **8.2 人工扰动**（GUM 约 11 万个候选，7 类 × 4 个比例 × 20 个重复）：所有证据类评分对“合并-终止”“多切-标点/无标点”“错位”的识别率
  ≈ 1、斜率 ≈ 1；但对**“合并-其他”（删掉无终止标点的 Gold 边界）有盲区**：GUM 上斜率 Probe 0.32、Stack 0.24、E2 0.12、共识 −0.07
  （不对称指数全部 < 0.5），对**“多切-终止”（缩写后误切）也不敏感**（识别率 0.74–0.85）。PuncEv 在这两类上斜率为负（−0.23 / −0.84），
  即奖励这两类错误。只有 Dawid–Skene（借用了系统投票）没有盲区（不对称指数 0.72）。LinES 上 Stack 对这两类的斜率都为负（−0.14 / −0.48），
  直接解释了 LinES 的失败。跨文档（混合类型）全局 Spearman：Stack 0.919 最高（Probe 0.840，PuncEv 0.635，共识 0.807，DS 0.876）；
  Stack 相对 Probe 的优势来自文档之间（0.731 vs 0.173），文档内两者相当，Stack 在两类盲区上反而略差（识别率 −0.06）。
- **8.3 真实错误计价**：Stack 修正“漏切-其他”得到的分数只有真实 F1 增量的 16%（漏切-终止 89%），比值 0.18 [0.11, 0.25]。
  按预登记判定，“Punkt 被高估由共同盲区解释”在 GENTLE、LinES 上成立，在 GUM 上不成立（漏切-其他占比最高的是 Stanza 0.84，
  不是 Punkt 0.69；但 Punkt 这类错误的绝对代价最大，+0.059 F1）。
- **8.4 原始问题（选择遗憾）**：没有方法稳定优于 Always-SaT。GUM 上所有方法的逐文档路由遗憾都显著高于 Always-SaT；GENTLE 上只有
  M3 显著更低（−0.043 [−0.079, −0.015]），LinES 上无显著差异。S1/S0/M3 在语料级都选中 SaT（Macro 遗憾 0），PuncEv 与 M0 选中 Stanza。
- **8.5 简单基线**：只看文档难度的 DocAgree（四个系统两两 F1 的平均，文档内常数）全局 Spearman 在 GUM / GENTLE / LinES 为
  0.637 / 0.773 / 0.741，其文档之间分量（0.913 / 0.883 / 0.867）与 S1 相当甚至更高；S1 − DocAgree 为 +0.181 [+0.125, +0.246] /
  +0.110 [+0.036, +0.193] / +0.075 [−0.180, +0.424]（LinES 跨 0）。S1 的增益主要来自文档之内。句长先验 ≈ 0，标点规则 0.34–0.44。
- **8.6 Stack 去掉单一证据**：去掉 E1 使全局 Spearman 下降（GUM −0.091，GENTLE −0.070；LinES +0.008 n.s.），去掉 E2 下降
  −0.042 / −0.044 / −0.046。GUM 上触发“同源偏差”判定（去掉 E1 后 SaT 残差变化绝对值 0.056 > 0.05），但方向是 E1 **减轻**了 SaT
  本来的被低估（−0.104 → −0.048），加入 E1 后 SaT 仍被低估；GENTLE、LinES 上不触发。GENTLE 上去掉 E1 后语料级第一名变成 Punkt（错）。
- **8.7 文档长度**：S1 全局 Spearman 短/中/长 0.809 / 0.780 / 0.870；Stack 识别率（混合，r = 0.10）0.977 / 0.990 / 0.998，短文档略差。

**SaT 分析与大小写试验**（修订 9，2026-11-18；用户决定：证据可以看大小写，原文段落换行可作为输入；目标仍是评估器）。
SaT 源码放在 `third_party/wtpsplit`（gitignored，commit 记录在 `third_party/wtpsplit.COMMIT`，`scripts/fetch_wtpsplit.slurm`），
设计分析见 `reports/sat_analysis_2026-11-18.md`。测试集复用规则写在修订 9（复用的数字标 R，取舍只看 GUM 开发集）。
9.1：`src/a9_casing.py`，GPU 作业 23615880（L40S，2.6 分钟）+ CPU 作业 23615881；完整表 `results/a9/casing_summary.md`。
sat-3l-sm 在原文大小写、阈值 0.25 下与被评 SaT 系统逐空隙一致（GUM 上 168,917 个空隙中只差 1 个），stack 逐值复现修订 6。

- **按预先写定的判定采用 cased 证据**（GUM LOGO）：Stack-cased（stack 中 E1 换成看大小写的 sat-12l）− Stack =
  全局 Spearman +0.059 [+0.039, +0.080]（0.877 vs 0.818），文档内配对 +0.056；“合并-其他”盲区斜率 0.238 → 0.463（+0.225 [+0.191, +0.259]）。
  **共同盲区造成的偏差几乎消失**：Punkt 秩残差 +0.067 → +0.017，SaT −0.048 → −0.006。
- 复用的测试集（R）：GENTLE Stack-cased 0.903 vs 0.883（+0.020 [−0.026, +0.072]），文档之内 0.692 → 0.806，盲区斜率 +0.171 显著，
  Punkt 残差 +0.046 → −0.007；LinES +0.017（勘误后 +0.022 [−0.055, +0.176]，n.s.，盲区来自标注规范，大小写没有帮助）；
  EWT 上单一 SaT 证据看大小写后 +0.05 到 +0.06（勘误后 +0.059 到 +0.068，显著），
  但 EWT 的绝对值仍低（sat-12l-sm cased 0.496，勘误后 0.517；共识 0.657）。
- sat-12l-sm（cased）单独作证据最强（GUM dev 0.897，GENTLE 0.884），但它与被评的 sat-3l-sm 同族、同训练数据（GUM train）：
  GUM dev 上 SaT 秩残差 +0.078（被高估），GUM train 上被污染，因此不放进主证据。
- 诊断（证据当切分器用，R）：Stack-cased 在 GENTLE / LinES 上显著优于被评的 SaT 系统（+0.043 [+0.003, +0.088] / +0.013 [+0.003, +0.024]）；
  sat-12l-sm cased 在 GENTLE 上 0.888 vs 0.818（+0.070 [+0.025, +0.118]）。

**9.2 + 9.3：大小写 × 版式的 2×2 析因**（控制变量；`src/a9_layout.py`）。GPU 作业 23656450（H200，30 分钟；同一脚本在 general 账户下的
孪生作业 23656448 被自动取消），CPU 分析作业 23656451（5 分钟）；完整表 `results/a9/layout_summary.md`。
换行版文本 = UD `# newpar` 处的空格换成 `\n`（GUM 4353 处，GENTLE 652，EWT 538，LinES 0）。四个证据格 = 证据输入的（大小写, 版式）；
场景 S = 被评系统是否也在换行文本上运行。核对：(lower, flat) 格的 stack 在 GUM / GENTLE / LinES 上逐值复现修订 6（最大差 0）；
LinES（无段落，阴性对照）上系统输出完全相同，sat-12l 证据最大差 0.023（logit，半精度误差）。
实施说明（出结果前）：SaT 的 `split_on_input_newlines` 会删掉换行符，片段拼不回原文，改为在偏移上补换行边界（与 SaT 默认行为逐个一致）。

- **按预先写定的判定采用 lower-nl**（GUM LOGO，主场景 S = nl）：四格 stack 全局 Spearman lower-flat 0.844，cased-flat 0.908，
  **lower-nl 0.937**，cased-nl 0.944；cased-nl 只高 0.007（< 0.01 容差），取更简单的 lower-nl；相对 9.1 的 Stack-cased（ref91，0.897）
  +0.040 [+0.024, +0.056]。适用范围：**输入带段落版式时**；没有版式的文本仍用 ref91（见下面的 LinES）。
- **主效应与交互**（GUM，S = nl，stack 全局 Spearman）：版式 +0.093 [+0.073, +0.116]（小写时）/ +0.036 [+0.025, +0.048]（cased 时）；
  大小写 +0.063 [+0.045, +0.084]（flat 时）/ +0.007 [−0.002, +0.014]（nl 时，n.s.；文档内配对 +0.019 [+0.010, +0.029] 显著）；
  交互 −0.056 [−0.076, −0.040]。**两者可以互相替代**：它们修的是同一个盲区（无标点边界大多在段落处，段落后首字母大写近似于换行）。
- **sat-12l（E1）的版式效应恰好为 0**：SaT 训练时删掉了所有换行（它的任务就是预测换行），推理时分词器也把 `\n` 当空格。
  版式的增益全部来自 E2（ModernBERT 把 `\n` 当作单独的 token，版式效应 +0.07）和 stack 里的“段落换行”特征。
- **盲区修复**：“合并-其他”斜率（理想 1）GUM 0.238（lower-flat）→ 0.463（ref91）→ 0.723（lower-nl）→ 0.777（cased-nl）；
  GENTLE 0.146 → 0.317 → 0.461 → 0.709。共同盲区偏差：GUM（S = nl）Punkt 秩残差 +0.084 → +0.025（lower-nl）。
- **被评系统看到换行后**：Punkt、Stanza 忽略单个换行，输出不变；SaT 提升（GUM 0.972 → 0.980，GENTLE 0.818 → 0.875，EWT 0.900 → 0.935）；
  WtP 反而下降（0.903 → 0.827，0.765 → 0.735，0.799 → 0.640）。系统之间分歧变大，共识 M0 变差（GUM 0.733 → 0.675，EWT 0.657 → 0.590），
  stack lower-nl − M0 在 GUM 上 +0.262 [+0.224, +0.299]。
- 复用的测试集（R，S = nl）：GENTLE lower-nl 0.893 vs ref91 0.876（+0.018 [−0.059, +0.098]），cased-nl 0.926（+0.051 [−0.001, +0.101]）；
  EWT lower-nl 0.550 vs ref91 0.527（+0.023 [−0.012, +0.055]），但仍不显著低于 M0（0.590，−0.040 [−0.173, +0.081]；S = flat 时 M0 显著更高）。
  **勘误后**：EWT（S = nl）lower-nl 0.570 vs ref91 0.545（+0.025 [−0.005, +0.055]），与 M0（0.591）只差 −0.020 [−0.159, +0.102]；
  S = flat 时 M0（0.657）只比无换行的格显著更高（lower-flat −0.141 [−0.263, −0.025]），比换行格不再显著（lower-nl −0.097 [−0.220, +0.017]）。
  GENTLE（S = nl）lower-nl 0.890 vs ref91 0.876（+0.015 [−0.063, +0.095]）。
- **LinES（没有版式）暴露了版式依赖**：系统输出与证据输入都相同，但 lower-nl 的 stack 从 0.815 降到 0.743（−0.072 [−0.304, +0.091]，
  9 篇，n.s.），“合并-其他”斜率 −0.142 → −0.632。原因：E2 和 stack 在 GUM 上学到“无标点边界 ≈ 段落换行”，没有换行时就压低这类边界。
  cased-nl 受影响较小（0.837）。所以版式评估器只用于带版式的输入。
- 观察（不改变判定）：在 nl 条件下加大小写，文档内配对在 GUM / GENTLE / EWT 上都显著为正（+0.019 / +0.067 / +0.019），
  cased-nl 在 GENTLE、LinES 上也优于 lower-nl。按预先写定的规则仍采用 lower-nl，cased-nl 作为敏感性变体一并报告。
- **诊断：证据当切分器**（p ≥ 0.5，与同一场景下被评的 sat-3l-sm 比 macro F1，配对 CI；事后分析，P/R；`src/a9_segmenter.py`，
  `results/a9/segmenter_ci.md`）：cased-nl 在四个语料上都显著优于 SaT：GENTLE 0.953 vs 0.875（+0.078 [+0.040, +0.115]，19 篇更好 / 6 篇更差），
  EWT 0.963 vs 0.935（+0.028 [+0.012, +0.045]），LinES +0.008 [+0.003, +0.015]，GUM +0.006 [+0.003, +0.009]（被污染）。
  lower-nl：EWT +0.020 显著，GENTLE +0.021 与 LinES −0.006 不显著。stack 的权重在 GUM Gold 上有监督地拟合，GENTLE 沿用 GUM 的标注规范。
- **与 12 层 SaT 比**（同一文件；原文大小写、wtpsplit 默认阈值；看到换行时在换行处强制切开。用 sat-12l 在换行文本上直接算的概率核对过，
  四个语料中决定不同的空隙为 0–1 个）。SaT 本身：sat-12l-sm 是最强的 SaT（GENTLE 0.888 → 看到换行 0.925，EWT 0.920 → 0.939，
  LinES 0.974，GUM 0.988 被污染），sat-12l（只有换行预训练）较弱（GENTLE 0.848，EWT 0.888，看到换行时）。
  stack 当切分器 vs **sat-12l-sm**：cased-nl 在 GENTLE +0.028 [+0.004, +0.056]、EWT +0.024 [+0.009, +0.042] 上显著更好，
  LinES −0.010 [−0.024, +0.004]、GUM −0.003 不显著；lower-nl 在 LinES 显著更差（−0.024），GENTLE −0.029 不显著，EWT +0.017 接近显著；
  无换行的格在 GENTLE / GUM 上显著更差，EWT 上显著更好（ref91 +0.029）。vs **sat-12l**：除 LinES 外各格都显著更好（stack 本身以 sat-12l 为组件）。
  结论：看大小写和换行的 stack 已经与最强的 SaT 持平或更好（GENTLE、EWT 显著），但两者都用了 GUM 的监督，GENTLE 又沿用 GUM 的标注规范。
- 注意：UD 的段落标记是清洗过的理想版式，版式条件下的数字是乐观上限。

**修订 10：面向质量的证据融合 vs 事后质量回归（控制变量）**（`src/a10_quality.py`；CPU 作业 23678606/07/08，各约 12 分钟；
完整表 `results/a10/summary_T{1,2,3}.md`，逐条目分数 `results/a10/scores_*.npz`）。同一设置内所有方法共用同一个设计矩阵、LOGO 折、条目和 Q_norm，
只改训练目标或第二阶段。T1 = 修订 6 的 lower-flat，T2 = ref91（无换行输入的现行评估器），T3 = lower-nl（带版式输入的现行评估器）。
核对：T1 的 S1 证据与修订 6 的 stack 逐值相同；T1 的 S2 − S1 = +0.001（复现修订 6 的 0.819 vs 0.818）。

| GUM LOGO 全局 Spearman | T1 | T2 | T3 |
|---|---:|---:|---:|
| S1（BCE stack → Q_norm，现行） | 0.818 | 0.877 | 0.939 |
| S2 岭回归 QE（T3 为 S2-ns） | 0.819（+0.001） | 0.860（−0.017） | 0.927（−0.012*） |
| A F1 回归器 / R 残差 QE | 0.820 / 0.820 | 0.873 / 0.873 | 0.930 / 0.930 |
| B-q 只用质量损失 | 0.795 | 0.849 | 0.933 |
| B-qr 质量 + 排序 | 0.805 | 0.849 | 0.937 |
| **B-qb 质量 + BCE** | **0.849（+0.031 [+0.014, +0.049]）** | **0.903（+0.026 [+0.014, +0.038]）** | **0.948（+0.009 [+0.004, +0.015]）** |
| B-qbr 质量 + BCE + 排序 | 0.848 | 0.903 | 0.948 |
| B-qb 留一系统（LOSO）− S1 | +0.021 [+0.004, +0.038] | +0.022 [+0.011, +0.034] | +0.010 [+0.005, +0.015] |

- **按预先写定的判定，T2 和 T3 都采用 B-qb**（B-qbr 也满足全部条件，但与 B-qb 相差 < 0.01，取更简单者）：相对 S1 和 S2 的 CI 都在 0 以上，
  LOSO 下仍为正，扰动斜率变好（“合并-其他”T2 0.463 → 0.608，T3 0.723 → 0.754；四类平均 0.727 → 0.794 / 0.780 → 0.817），
  边界概率基本不变（留出 log loss T2 0.0103 → 0.0107，AUC 0.9996 → 0.9993）。共同盲区偏差也减小（T1 Punkt 秩残差 +0.067 → +0.026）。
- **事后回归（S2 / A / R）都不比 S1 好**，而且**换一个没见过的系统就显著变差**（LOSO − S1：T1 −0.026 到 −0.040，T2 −0.024 到 −0.032，T3 −0.016 到 −0.023）：
  它们学到的是这四个系统的特点。A 与 R 几乎相同：R 只把 Q_norm 系数的收缩目标从 0 改成 1，岭回归 λ = 1、736 个条目时影响很小。
- **只用质量损失（B-q、B-qr）不行**：GUM 上比 S1 低（T1 −0.023，T2 −0.027），LOSO 下 −0.07，留出 log loss 变成 3–4 倍，
  GENTLE 上 −0.08 到 −0.10。正如预想：只约束四个系统的质量分数，对边界概率的约束不够。**BCE 项是关键**；排序损失没有额外收益。
- 复用的测试集（R）上收益变小、多数不显著：B-qb − S1 在 GENTLE T1 −0.002 / T2 +0.007 / **T3 +0.020 [+0.005, +0.039]**，
  LinES +0.019 / +0.032 / +0.014（9 篇，CI 很宽），EWT −0.001 / −0.008 / +0.002。T3 的 GENTLE 上 S2-ns / A / R 为 +0.035 到 +0.039（不显著）。
  所以 GUM 上的收益有一部分是领域内的；跨领域时 B-qb 至少不比 S1 差。
- **测量问题（平局，影响以前的数字）**：同一文档里两个系统的切分完全相同时，数学上相等的分数因浮点求和顺序不同而相差约 1e-16，
  排序时被当成一高一低。这会放大分数秩的方差，系统性地**压低全局 Spearman**（文档内配对不受影响，Gold 相同的条目对本来就被排除）。
  修订 10 在评估前把分数四舍五入到 1e-10 后：GUM 上 S1 只变 +0.0002 到 +0.002，但 **EWT 上 S1 从 0.494 / 0.499 / 0.549 升到 0.516 / 0.531 / 0.570**，
  LinES T2 从 0.832 升到 0.840。修订 6–9 的数字（特别是 EWT 和各方法与 M0 的比较）都含这个偏差，需要统一用平局精确化重算一遍；
  GUM 上的判定差距都远大于这个偏差，不受影响。未修正的第一轮修订 10 结果保存在 `results/a10_run1_untied/`（判定同样是 B-qb）。

**勘误：平局精确化（2026-11-19，用户同意）**。修正写进共用的统计函数（`src/quality_agreement.py` `tie_exact`：分数与 Gold F1
四舍五入到 1e-10 后再排序；`statistics`、`evaluate`、`fairness`、`decomposition`、修订 8 的跨文档统计、`a9_layout.fast_stats` 都使用它），
只重跑 CPU 分析阶段（`scripts/run_erratum_ties.slurm`；作业 23686424 修订 6、23686425 修订 7、23686426 修订 8、23686427 修订 9），
不重跑 GPU、证据或测试推理；修订 6 的冻结组件不重写（`--erratum` 模式核对预注册选择）。旧报告在 `results/erratum_ties/untied/`，
逐项对照 `results/erratum_ties/summary.md`、`changes.json`，显著性翻转 `significance_flips.json`。

- **所有预先写定的判定都不变**：修订 6 的选择（证据 stack、S* = S1、M* = M3）与冻结组件一致；9.1 采用 cased 证据；9.2+9.3 采用 lower-nl
  （0.939 vs cased-nl 0.944，差 0.0046 < 0.01；相对 ref91 +0.041 [+0.027, +0.058]）；修订 10 采用 B-qb。非数值字段（选择、判定、标记）零变化。
- 变化大小：GUM ≤ 0.002，GENTLE ≤ 0.003，LinES ≤ 0.008（9 篇，CI 边界变化更大），**EWT 上 stack 类评分 +0.018 到 +0.034**
  （例：S = nl 时 lower-nl 0.550 → 0.570，ref91 0.527 → 0.545）。**M0、DocAgree 和修订 4 的 EWT 一次性测试（Probe 0.411 vs PuncEv 0.338，
  +0.074 [+0.023, +0.126]）完全不变**：它们的计算本来就让相同的切分得到完全相同的分数；所以以前 EWT 上“stack 不如 M0”的差距被夸大了。
  修订 4 的数字是在 `results/erratum_ties/ewt_test/` 里重算核对的（原结果未覆盖）；修订 2–5 的其他分析走同一条逐条计算的路径，没有重跑。
- 6 处显著性翻转，全在复用的测试集上（R，只报告）：EWT 无换行场景下 lower-nl / cased-nl − M0 不再显著（−0.097 [−0.220, +0.017] /
  −0.093 [−0.219, +0.023]）；EWT 上“cased 时版式的效应”在两个场景都变为显著（S = flat +0.041 [+0.004, +0.082]，S = nl +0.035 [+0.005, +0.065]）；
  GENTLE（S = flat）“nl 时大小写的效应”变为显著（+0.054 [+0.001, +0.108]）；修订 6 GENTLE 上 E1sm − E0 从 n.s. 变为刚好显著（下界 −0.0002 → +0.0001）。
- 论文提纲 `paper/outline.md` 中只有 LinES 的 Stack 格变了（S1 0.815 → 0.818，S1 − S0 −0.053 → −0.050），已更新。

**修订 11：论文正文的补充分析**（全部事后分析，不做取舍；`src/a11_analysis.py`；作业 23695119 生成证据缓存，23696612 分析；
完整表 `results/a11/summary.md`，例子 `results/a11/examples.md`）。评估器阶梯：PuncEv → Probe → Stack → + casing → + layout → 最终（B-qb）。
核对：各版本在 GUM 上的全局 Spearman 复现修订 9–10（Stack 0.818、+ casing 0.877、最终 0.903；带版式 + layout 0.939、最终 0.948），
EWT 上 PuncEv / Probe 复现修订 4（0.338 / 0.411）；扰动生成器在原 7 类、原 4 个比例上与修订 8 逐行相同（单元测试）。

- **E1 系统排名一致性**（4 个系统按平均分排序 vs 按 Gold F1 排序，Kendall τ_b）。领域平均 τ / 文档平均 τ：GUM 带版式 最终 0.933 / 0.894，
  共识 M0 0.489 / 0.458（差 +0.444 [+0.378, +0.533] / +0.435）；GENTLE 带版式 最终 0.773 / 0.758，M0 0.300 / 0.383；
  EWT 带版式 最终 0.800 / 0.668，M0 0.333 / 0.474。**共识给系统排序很差**（它奖励与多数一致的系统），证据类评估器好得多。
  语料层面最终评估器在 GUM、GENTLE、LinES、EWT 上都把 SaT 排第一（GENTLE 语料 τ 0.667，EWT 无换行 0.333）。
- **E2 评分公式**：归一化在所有语料、所有版本上都有用或无害（GUM：PuncEv +0.235，Stack +0.037，最终 +0.053）；证据强时归一化 Soft F1 ≈
  硬伪参考（GUM 最终 +0.008 n.s.，带版式 +0.015 显著）。**但 EWT 上硬伪参考远好于 Soft F1**（最终 0.888 vs 0.572，带版式）。
- **新发现：Soft F1 在单句文档上退化**。EWT 有 33/316 篇文档只有一句话：正确的“不切”Soft F1 = 0（硬 F1 按惯例给 1），正确答案被排到最后。
  事后变体 Q_end（把必然正确的文档末尾边界也计入，归一化时允许 k = 0）：GUM / GENTLE / LinES 变化 ≤ 0.007，**EWT 最终评估器
  0.522 → 0.787（无换行）、0.572 → 0.894（带版式，+0.321 [+0.229, +0.420]）**，修订 4 的 Probe 0.411 → 0.647、PuncEv 0.338 → 0.508；
  修正后 EWT 上 Soft ≈ 硬（+0.005），且远超共识（0.894 vs 0.591）。**以前“EWT 上共识胜过证据”的结论主要是这个退化造成的。**
  这是在复用测试集上的事后发现（P），没有改用；建议另写修订并用全新的 EWT dev 一次性检验（见下面 E5）。
- **人工扰动 v2**（在 Gold 上注入 1 个错误，分数是否下降 = 检出率；GUM，括号为 95% CI）。所有版本对“句末标点处漏切”“逗号等处多切”
  “无标点处多切”“边界错位”的检出率 ≈ 1。难的是两类：
  - **无句末标点处漏切**：在段落边界处 Stack 0.67 → + casing 0.81 → + layout 0.98 → 最终 0.98；但在**段落内部** Stack 0.52、
    + casing 0.61、最终（无换行）0.62，**带版式的 + layout 0.05、最终 0.07 [0.03, 0.13]**——学会“无标点边界 ≈ 换行”之后，段落内部的
    无标点边界反而被惩罚（GENTLE 0.23、EWT 0.24，同样的现象；斜率 −0.59）。这解释了带版式评估器在 LinES 上的退化。共识对段落边界处的
    漏切也不敏感（0.46，GUM）。
  - **缩写 / 编号后的误切**：PuncEv 0.22，Probe 0.77，Stack 0.73，最终 0.81–0.84；共识 0.61。
- **E3 真实错误定价**（只修正某类错误时分数增量 / 真实 F1 增量，1 = 按真实损失计价；GUM 带版式）：句末标点处漏切 ≈ 1；段落边界处
  无标点漏切 Stack 0.34 → 最终 1.01；段落内部 Stack −0.18 → 最终 −0.58（修正它反而扣分）；缩写后误切 Stack −0.17 → 最终 0.12；
  共识对无标点漏切 −0.14（看不见共同错误）。
- **E4 例子**：`results/a11/examples.md`（每个语料最多 25 对评估器与 Gold 顺序相反的系统，附不同空隙的上下文）。例：GENTLE 诗歌中
  标注把分号、破折号后算作句子边界，证据 p 只有 0.02–0.4，于是不切这些位置的 Punkt 被高估。
- **E5 全新测试集核查**（`results/e5_candidates.md`，只看结构）：EWT dev（318 篇、750 处段落、5 个领域）最合适；PUD 对 Stanza 被污染、文档很短；
  ParTUT、ESLSpok 没有文档标记，CTeTex 几乎都是单句；LinES train + dev 可把 LinES 的结论扩到 27 篇。

**修订 12：采用 Q_end，并在两个全新语料上做一次性测试**（用户同意；预注册 `1440748` 和测试代码 `dff0d27` 都在处理数据之前提交；
`src/a12_test.py`；GPU 作业 23740230（h200_tandon，8 分钟；general 账户的孪生作业 23740228 被自动取消），CPU 作业 23740307；
完整结果 `results/a12/summary.md`）。评分函数改为 Q_end（文档末尾边界计入，归一化时 k = 0..n）。测试对象：在全部 GUM train 上拟合的最终评估器
（无换行 = B-qb + ref91 特征；带版式 = B-qb + lower-nl 特征）。完整性核对：重新拟合的 5 个模型在 GENTLE 上逐值复现修订 11（最大差 0）；
分数全部冻结之后才生成 Gold。语料：EWT dev（318 篇，其中 42 篇单句；有段落版式）、LinES train + dev（18 篇，文档编号改为 lines-train-K / lines-dev-K）。

| 预先写定的假设 | 估计 [95% CI] | 结果 |
|---|---:|---|
| H1（主）EWT dev 带版式：最终 − 共识 M0 | +0.227 [+0.158, +0.298]（0.851 vs 0.625） | 成立 |
| H2 EWT dev：最终 − Probe（带版式 / 无换行） | +0.195 [+0.133, +0.256] / +0.202 [+0.146, +0.261] | 成立 |
| H3 EWT dev：最终评估器 Q_end − Q | +0.348 [+0.251, +0.449]（0.851 vs 0.503） | 成立 |
| H4 EWT dev：系统排名领域平均 τ，最终 − 共识 | +0.533 [+0.067, +0.600]（1.000 vs 0.467；文档平均 0.783 vs 0.541） | 成立 |
| H5 LinES train + dev：最终 − Probe | −0.088 [−0.245, +0.075]（0.563 vs 0.651） | 不成立：LinES 的失败再次出现 |
| H6 LinES：带版式评估器 − 无换行评估器 | −0.095 [−0.299, +0.050] | 方向符合，不显著 |

- EWT dev 上阶梯每一步都有增益（带版式，Q_end）：Probe 0.656 → Stack 0.776 → + casing 0.831 → + layout 0.847 → 最终 0.851；
  无换行：0.595 → 0.739 → 0.800 → 最终 0.797（共识 0.657）。软 ≈ 硬（+0.009 / +0.005）。最终评估器胜在文档之内（0.802 vs 共识 0.539），
  共识胜在文档之间（0.919 vs 0.876）。
- LinES train + dev：在 GUM 上得到的改进都不迁移（Stack 0.560，+ casing 0.541，最终 0.563，Probe 0.651），但最终评估器仍胜过共识
  （+0.095 [+0.007, +0.210]），语料层面把 4 个系统排对（τ = 1）。Q_end 在 LinES 上无影响（没有单句文档）。
- 扰动 v2 在 EWT dev 上复现：段落内部的无标点漏切，最终（无换行）检出 0.84，最终（带版式）0.35；LinES 上缩写后的误切检出只有 0.19–0.36。
- 论文提纲更新为 v4（`paper/outline.md`）：主表改为 Q_end，并加入两个全新测试集的 T 列。

**修订 13：自然基线对照 + 带对照组的敏感度测试**（用户要求；事后分析，不做取舍；`src/a13_baselines.py`；作业 23788753（SaT-12l-sm 在 CPU 上补跑
EWT dev、LinES train+dev，3 分钟）、23790903（同族偏差与平均集成）、23791114（修正 z 的处理后重跑）；结果 `results/a13/summary.md`、`extra.md`）。
基线：B1 句末标点规则（伪参考）、B2 句长先验、B3 遮蔽语言模型（PuncEv）+ 我们的评分、B4 SaT-12l-sm 的切分当伪参考、B5 SaT-12l-sm 的概率 + 我们的评分、
B6 共识、B7 DocAgree；带版式时基线也看到段落边界。核对：由特征复现的 PuncEv 与修订 4 的 EWT 证据一致。

- **全局 Spearman（无换行）**：我们 GUM 0.903、GENTLE 0.907、LinES test 0.873、EWT test 0.787、**EWT dev 0.797**、LinES train+dev 0.563。
  最强基线是 B5（SaT-12l-sm 概率 + 我们的评分）：0.936（GUM，被污染）、0.891、0.873、0.778、0.762、0.630。我们 − B5 在所有留出语料上都不显著
  （EWT dev +0.035 [−0.014, +0.088]，GENTLE +0.015，LinES train+dev −0.066）；带版式时 EWT dev 0.851 vs 0.835（+0.016，n.s.），GENTLE 0.903 vs 0.929。
  我们显著好于 B4（SaT 切分当参考）：EWT dev +0.060 / +0.054。对其余基线优势很大：EWT dev 上比规则 +0.42、比遮蔽 LM +0.33、比共识 +0.14 / +0.23；
  但带版式时“标点 + 换行”规则已经很强（EWT dev 0.796），我们只高 +0.055（显著）。
- **系统排名**：我们与 B5 都几乎完全排对 4 个系统（EWT dev 带版式领域平均 τ 1.0），共识 0.47。**同族偏差**：SaT 系评估器略高估 SaT 候选
  （秩残差 +0.02 到 +0.03；GUM、EWT dev），我们带版式时接近 0。**不拟合权重的平均集成**（我们 + B5）：EWT dev 0.808 / 0.862、EWT test 0.818 / 0.910，
  比 B5 显著好、比我们小幅好；LinES 上更差。
- **敏感度（注入 1 个错误的检出率，EWT dev；对照组对所有方法都是 0.50）**：所有读文本的方法都能察觉带标点的错误；区分方法的是“无句末标点的漏切”
  和“缩写 / 编号后的误切”。规则完全看不见前者，句长先验只认漏切、看不见任何误切，遮蔽 LM 和共识在两者上都接近 0.5，DocAgree 处处 0.50
  （但 Spearman 仍有 0.51——说明只看 Spearman 不够）。我们（无换行）对无标点漏切最敏感（段落边界处 0.93 / 段落内 0.84，B5 为 0.85 / 0.64），
  B5 对缩写后误切更敏感（0.75 vs 0.66）——两者盲区互补；带版式的我们在段落内只有 0.31。
- **度量的事后调整**：z（零假设标准化反应）在方法之间不可比（取决于各自对无关改动的波动 σ₀；硬参考类方法 σ₀ 可能为 0），所以正文用检出率
  （与尺度无关的“优势概率”），z 进附录；已写入预注册的修订 13 补记。
- 论文提纲更新为 **v5**：正文主表改为“基线 vs 我们”（T2），新增敏感度表（T3），开发阶梯、训练目标对比移到附录。

## GitHub 持续同步

本环境的系统 `.git` 是只读空挂载，因此仓库元数据持久保存在被忽略的 `.gitdata/`。使用以下包装命令代替普通 `git`：

```bash
scripts/git_project.sh status
scripts/git_project.sh add <files>
scripts/git_project.sh commit -m "message"
```

Login node 当前不能解析 GitHub；提交后通过 CPU short job 安全地 fetch/rebase/push
当前 review branch。必须显式传入分支名，脚本会拒绝推送 `main`：

```bash
sbatch --export=ALL,GIT_SYNC_BRANCH=experiment/gum-real-splitter-eval scripts/sync_github.slurm
```

远程 URL 不含任何凭据。项目专用 HTTPS credential store 位于 `.runtime/git/credentials`，权限为 `0600`，并被 `.gitignore` 排除。

## BERT 双向上下文对照

`google-bert/bert-base-uncased` 在每个候选空白处插入 `[MASK]`，使用左右各约 126 token 的双向窗口，并与 Qwen 使用相同的 GUM train Platt calibration、dev 候选和 Soft F1：

```bash
sbatch scripts/run_bert_comparison.slurm
# 已有完整概率缓存时只重算稳定校准和比较，不使用 GPU：
source scripts/env.sh
/scratch/wj2301/conda_envs/jit/bin/python -m src.run_bert_comparison --analyze-only
```

结果位于 `results/bert_vs_qwen_summary.md`、`results/bert_vs_qwen_metrics.csv` 和 `results/bert_vs_qwen.json`。
