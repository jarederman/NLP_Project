# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Reference-free sentence segmentation evaluation. It scores candidate segmentations of a document without reading that document's gold boundaries, using LM evidence at each candidate gap: Qwen3.5-9B next-token terminal-punctuation probability in the preliminary study, and `google-bert/bert-base-uncased` `[MASK]` probability in the real-splitter study. The scores are Platt-calibrated on GUM train, and candidates are ranked by Soft F1 against those probabilities. The goal is to pick the best real splitter (Punkt, Stanza, WtP `wtp-bert-mini`, SaT `sat-3l-sm`) per document. README.md is in Chinese and is the authoritative experiment log.

## Environment (always do this first)

```bash
source /scratch/wj2301/NLP_Project/scripts/env.sh
```

This sets `HOME`, all HF/Torch/NLTK/Stanza/pip/tmp caches to paths inside the project (`.cache/`, `.runtime/`, `.tmp/`), and sets `PYTHONPATH` to the project root. Never write to the real `$HOME`.

- Real-splitter pipeline interpreter: `.venv/bin/python`. This venv uses `--system-site-packages` on top of `/scratch/wj2301/conda_envs/jit`; set it up via `scripts/setup_real_environment.slurm`.
- Preliminary Qwen/BERT-comparison code: `/scratch/wj2301/conda_envs/jit/bin/python` (PyTorch 2.10, Transformers 5.7).
- Don't modify the shared conda env.
- GPU work runs through Slurm (`h200_tandon` partition, account `torch_pr_676_tandon_advanced`). Logs go to `logs/`.

## Tests

The tests are CPU-only and need no model:

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest tests.test_selection -v                    # one module
python3 -m unittest tests.test_selection.SelectionTests.test_macro_and_micro_are_reported_separately -v   # one test
```

## Running experiments

The codebase is considered stable: submit formal jobs directly, without a smoke job first (user decision, 2026-11-03). The formal and train-CV scripts run the unit tests as their first step, so a broken tree fails before any artifact is overwritten. Archive the previous outputs (e.g. `results/real_v1/`, `data/real_system_v1/`) before rerunning under a new protocol version.

```bash
# Real-splitter study (current focus)
sbatch scripts/setup_real_environment.slurm
sbatch scripts/run_real_formal.slurm     # GUM dev; also builds the train BERT cache
sbatch scripts/run_train_cv.slurm        # P1: GUM-train 5-fold cross-fitting (needs the formal run first)
# scripts/run_real_smoke.slurm still exists for debugging GPU-only code paths

# Preliminary Qwen study (synthetic perturbation candidates)
sbatch scripts/run_smoke.slurm           # check status=PASS
sbatch scripts/run_formal.slurm
python -m src.analyze_cached             # CPU-only recompute from data/lm_cache/

# BERT vs Qwen comparison
sbatch scripts/run_bert_comparison.slurm
python -m src.run_bert_comparison --analyze-only
```

Data is pinned to UD commit `1fe635509c649e376dfb449d528424ab78f4eaee`, which has 197 GUM train and 30 dev docs. Fetch it with `scripts/fetch_gum.sh` (GUM) and `scripts/fetch_ewt.sh` (EWT test). Each fetch writes `UD_RELEASE` and `SHA256SUMS`.

## Architecture: real-splitter pipeline and the gold-isolation invariant

`scripts/run_real_formal.slurm` runs six stages in sequence. Each one is a `python -m src.<module>` with argparse defaults under `data/real_system/` and `results/`. The smoke script runs the same stages with `--limit 2` and redirected paths, so it shows how to call each stage on custom paths.

1. `prepare_real_data`: `surface_data.py` rebuilds full documents from CoNLL-U surface tokens, MWTs, and `SpaceAfter=No`. It deliberately ignores `# text`, `# sent_id`, and blank lines between sentences. Output is `RawDocument` JSONL (`raw_dev.jsonl`, `surface_dev.jsonl`).
2. `run_real_splitters`: `real_splitters.py` (`RealSplitterSuite`) produces `CandidateSegmentation` artifacts and logs alignment failures.
3. `run_surface_bert`: `bert_surface.py` computes per-gap BERT evidence and the GUM-train Platt calibration. If a visible `.?!` (including one inside quotes) already sits before the gap, it replaces that punctuation with `[MASK]`. Otherwise it inserts `[MASK]` at the gap.
4. `score_real_splitters`: gold-free scoring. Output is `frozen_scores.jsonl`.
5. `materialize_dev_gold`: refuses to run unless `frozen_scores.jsonl` exists and is non-empty, and only then writes `gold_dev.jsonl`.
6. `analyze_real_splitters`: the only stage that loads dev gold. It calls `selection.py` (Macro Selection Regret is the primary metric, Micro is reported separately and never mixed with Macro, plus Direct Internal Splitter and Always-SaT baselines).

`contracts.py` defines `RawDocument`, `GoldBoundaries`, and `CandidateSegmentation` as separate JSONL artifacts that use character offsets. They are aligned by `doc_id` plus the text SHA256 (`validate_against`). Keep these rules when changing anything:

- Splitters and scorers must read only `RawDocument` artifacts.
- Gold must never be read before scores are frozen.
- Never read GUM test.
- Train is used only to fit the two-parameter Platt calibration.
- Frozen protocol settings live in `configs/real_splitters.json` (model, mask protocol, systems, seed). Changing any of them is a protocol change.

The preliminary study (`run_experiment.py`, `data.py`, `lm.py`, `candidates.py`, `scoring.py`, `evaluate.py`, `report.py`) works differently. It removes sentence-final `.?!` from the text using gold (only to build the benchmark), generates seeded Gold/Over/Under/Shift/Mixed candidates, and caches Qwen logits per document in `data/lm_cache/`. A known leak remains: the final period of an in-word abbreviation can't be removed without loss.

## Git: use the wrapper, not plain `git`

The system `.git` is a read-only empty mount. Repo metadata lives in the gitignored `.gitdata/`:

```bash
scripts/git_project.sh status | add | commit ...
```

The login node can't resolve GitHub, so push from a Slurm CPU job with an explicit branch. The script refuses to push `main`.

```bash
sbatch --export=ALL,GIT_SYNC_BRANCH=<branch> scripts/sync_github.slurm
```

Credentials are in `.runtime/git/credentials` (mode 0600, gitignored). Never put them in remote URLs.
