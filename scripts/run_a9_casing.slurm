#!/usr/bin/env bash
#SBATCH --job-name=seg-a9-casing-gpu
#SBATCH --account=torch_pr_676_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a9_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a9_%x_%j.err
# PREREGISTRATION amendment 9.1 (GPU, inference only): SaT {3l-sm, 12l-sm, 12l} x {cased, lower}
# per-gap probabilities on GUM train / GUM dev / GENTLE / LinES / EWT test. Writes */a9*/gaps only.
# Short GPU job: general account, no partition (scheduler picks the GPU type; user decision 2026-11-18).
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a9_casing gpu
