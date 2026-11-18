#!/usr/bin/env bash
#SBATCH --job-name=seg-ewt-analysis
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/ewt_analysis_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/ewt_analysis_%j.err
# Amendment 4a: re-run only the post-freeze EWT stages (Gold materialization + analysis) after the
# GPU job froze every score. Evidence and frozen scores are not recomputed.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python

"$python" -m unittest discover -s tests
"$python" -m src.ewt_test gold
"$python" -m src.ewt_test analyze --bootstrap 1000 --workers 16 --gpu-log logs/ewt_test_gpu_22337008.csv
