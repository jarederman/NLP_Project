#!/usr/bin/env bash
#SBATCH --job-name=segmetric-bert
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/bert_comparison_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/bert_comparison_%j.err
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests
/scratch/wj2301/conda_envs/jit/bin/python -m src.run_bert_comparison
