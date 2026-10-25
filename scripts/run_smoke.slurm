#!/usr/bin/env bash
#SBATCH --job-name=segmetric-smoke
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon,a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/smoke_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/smoke_%j.err
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
bash scripts/fetch_gum.sh
PY=/scratch/wj2301/conda_envs/jit/bin/python
"$PY" -m src.run_experiment --mode smoke --config configs/preliminary.json
