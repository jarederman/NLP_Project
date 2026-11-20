#!/usr/bin/env bash
#SBATCH --job-name=seg-review-nested-cpu
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/review_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/review_%x_%j.err
# Code review 2026-11-20, finding 1 (diagnostic, GUM train LOGO only): nested n-grams, refit stacks and final
# B-qb evaluators on nested training features. Needs run_review_nested_gpu.slurm first. Writes results/review_fixes/nested.*
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests > "logs/review_nested_cpu_tests_${SLURM_JOB_ID:-local}.log" 2>&1 \
    || { echo "Unit tests failed; see logs/review_nested_cpu_tests_${SLURM_JOB_ID:-local}.log" >&2; exit 1; }
"$python" -m src.review_nested cpu --bootstrap 1000 --threads 16 --workers 8
