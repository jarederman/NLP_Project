#!/usr/bin/env bash
#SBATCH --job-name=seg-calib-search
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/calib_search_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/calib_search_%j.err
# P2: gold-free feature x model calibration experiments on GUM train (CPU only).
# Requires a completed run_train_cv.slurm (frozen candidates, folds, failures).
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$python" -m unittest discover -s tests -v
"$python" -m src.calibration_search "$@"
"$python" -m src.analyze_disagreements
