#!/usr/bin/env bash
#SBATCH --job-name=seg-a12-cpu
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a12_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a12_%x_%j.err
# PREREGISTRATION amendment 12 (CPU): refit + integrity check, freeze every score, then (only then) gold, then analysis.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a12_test freeze --threads 16
"$python" -m src.a12_test gold
"$python" -m src.a12_test analyze --bootstrap 1000
