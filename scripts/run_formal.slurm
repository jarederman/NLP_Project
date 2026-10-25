#!/usr/bin/env bash
#SBATCH --job-name=segmetric-gum-dev
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon,a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/formal_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/formal_%j.err
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
bash scripts/fetch_gum.sh
PY=/scratch/wj2301/conda_envs/jit/bin/python
"$PY" -m src.run_experiment --mode formal --config configs/preliminary.json
