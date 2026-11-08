#!/usr/bin/env bash
#SBATCH --job-name=seg-cond-context
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/cond_context_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/cond_context_%j.err
# BERT context comparison on GUM train: raw text vs raw+candidate vs candidate-only.
# Requires a completed run_train_cv.slurm (frozen candidates and folds).
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python

"$python" -m unittest discover -s tests -v
"$python" -m src.conditioned_bert
"$python" -m src.conditioned_eval
