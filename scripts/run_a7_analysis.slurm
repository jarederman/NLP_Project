#!/usr/bin/env bash
#SBATCH --job-name=seg-a7-analysis
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a7_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a7_%x_%j.err
# PREREGISTRATION amendment 7 (analysis only). A7_STAGE=posthoc: 7.1-7.3 (GUM dev + post-test
# GENTLE/LinES, from frozen artifacts). A7_STAGE=seeds: 7.4, after run_a7_seeds.slurm.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a7_posthoc "${A7_STAGE:?set A7_STAGE=posthoc or seeds}" --bootstrap 1000 --workers 16
