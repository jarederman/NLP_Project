#!/usr/bin/env bash
#SBATCH --job-name=seg-train-cv
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/train_cv_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/train_cv_%j.err
# P1: GUM-train 5-fold cross-fitting development set.
# Requires a completed run_real_formal.slurm for the current protocol (it writes
# data/real_system/{raw,surface,gold}_train.jsonl and the train BERT cache).
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python

"$python" -m unittest discover -s tests -v
# Same GPU/fp16 settings as the dev splitter run, so outputs are comparable.
"$python" -m src.run_real_splitters --split train
"$python" -m src.train_crossfit
"$python" -m src.analyze_evidence_gaps
"$python" -m src.evidence_ceiling
echo '{"status":"TRAIN_CV_COMPLETE"}'
