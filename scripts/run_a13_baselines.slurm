#!/usr/bin/env bash
#SBATCH --job-name=seg-a13-baselines
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a13_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a13_%x_%j.err
# PREREGISTRATION amendment 13 (post-hoc): SaT-12l-sm on the fresh corpora (CPU), then baselines + sensitivity.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a13_baselines sat --threads 16
"$python" -m src.a13_baselines analyze --bootstrap 1000
