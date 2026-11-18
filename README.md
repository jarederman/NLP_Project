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
