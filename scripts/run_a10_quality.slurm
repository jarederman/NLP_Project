#!/usr/bin/env bash
#SBATCH --job-name=seg-a10-quality
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a10_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a10_%x_%j.err
# PREREGISTRATION amendment 10: quality-oriented fusion vs post-hoc QE, one setting per job.
#   sbatch --export=ALL,A10_SETTING=T3 scripts/run_a10_quality.slurm
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
setting="${A10_SETTING:?set A10_SETTING to T1, T2 or T3}"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a10_quality --setting "$setting" --bootstrap 1000 --threads 16 --workers 4
