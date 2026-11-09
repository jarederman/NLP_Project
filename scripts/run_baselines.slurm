#!/usr/bin/env bash
#SBATCH --job-name=seg-baselines
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/baselines_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/baselines_%j.err
# Amendment 5: reference-free baselines (single-system pseudo-reference, leave-one-out consensus,
# heuristics) vs the frozen evidence metric on GUM train (LOGO) and EWT test. CPU only.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python

"$python" -m unittest discover -s tests
"$python" -m src.baselines --bootstrap 1000 --workers 16
"$python" -m src.baseline_decomposition --bootstrap 1000
