#!/usr/bin/env bash
#SBATCH --job-name=seg-real-smoke
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/real_smoke_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/real_smoke_%j.err
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
smoke_root="$PROJECT_ROOT/.tmp/real_smoke"
mkdir -p "$smoke_root"

scripts/fetch_gum.sh
"$python" -m unittest discover -s tests -v
"$python" -m src.prepare_real_data \
  --limit 2 --output-root "$smoke_root/data" --manifest "$smoke_root/data_manifest.json"
"$python" -m src.run_real_splitters \
  --data-root "$smoke_root/data" --output-root "$smoke_root/predictions" \
  --failures "$smoke_root/alignment_failures.jsonl"
"$python" -m src.run_surface_bert \
  --data-root "$smoke_root/data" --cache-root "$smoke_root/lm_cache" \
  --evidence "$smoke_root/bert_dev_evidence.jsonl" \
  --calibration "$smoke_root/calibration.json"
"$python" -m src.score_real_splitters \
  --data-root "$smoke_root/data" --predictions "$smoke_root/predictions" \
  --evidence "$smoke_root/bert_dev_evidence.jsonl" \
  --output "$smoke_root/frozen_scores.jsonl"
"$python" -m src.materialize_dev_gold \
  --data-root "$smoke_root/data" --manifest "$smoke_root/data_manifest.json" \
  --frozen-scores "$smoke_root/frozen_scores.jsonl"
"$python" -m src.analyze_real_splitters \
  --data-root "$smoke_root/data" --scores "$smoke_root/frozen_scores.jsonl" \
  --evidence "$smoke_root/bert_dev_evidence.jsonl" --results "$smoke_root/results"
test ! -s "$smoke_root/alignment_failures.jsonl"
echo '{"status":"REAL_SMOKE_PASS","dev_gold_materialized_only_after_score_freeze":true}'
