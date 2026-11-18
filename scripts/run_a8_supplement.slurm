#!/usr/bin/env bash
#SBATCH --job-name=seg-a8-supplement
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a8_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a8_%x_%j.err
# PREREGISTRATION amendment 8 (analysis only, CPU): Gold as a candidate, controlled perturbations
# by error type, error pricing, selection regret, trivial baselines, Stack ablation, document length.
# GUM = GUM-train LOGO (dev); GENTLE / LinES are post-test. Writes results/a8/ only.
# The main step runs with the default thread count, as amendments 6/7 did (exact reproduction of dev S1).
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a8_supplement --bootstrap 1000 --workers 16
