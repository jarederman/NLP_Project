# Reference-Free Sentence Segmentation Evaluation

本项目在无待评文档 gold boundary 输入的条件下，用 Qwen3.5 的 next-token 句末标点概率作为 proxy，比较候选 sentence segmentation。正式评价只使用 GUM dev；GUM train 仅拟合一个冻结的两参数 Platt calibration；不读取 test。

## 项目结构

- `src/data.py`：CoNLL-U 文档级读取、`SpaceAfter=No` 处理、受控去句末标点文本与 boundary alignment。
- `src/lm.py`：`Qwen3_5ForCausalLM` BF16 forward、token/character alignment、候选位置 logits 和文档级缓存。
- `src/candidates.py`：固定种子的 Gold/Over/Under/Shift/Mixed 候选。
- `src/scoring.py`：raw/calibrated BCE、weighted BCE、Soft F1/Fβ、句长 baseline、缩写规则。
- `src/evaluate.py`：Spearman、同文档 pairwise ranking、gold-win、类型/genre 汇总和实例选择。
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

## GitHub 持续同步

本环境的系统 `.git` 是只读空挂载，因此仓库元数据持久保存在被忽略的 `.gitdata/`。使用以下包装命令代替普通 `git`：

```bash
scripts/git_project.sh status
scripts/git_project.sh add <files>
scripts/git_project.sh commit -m "message"
```

Login node 当前不能解析 GitHub；提交后通过 CPU short job 安全地 fetch/rebase/push：

```bash
sbatch scripts/sync_github.slurm
```

远程 URL 不含任何凭据。项目专用 HTTPS credential store 位于 `.runtime/git/credentials`，权限为 `0600`，并被 `.gitignore` 排除。
