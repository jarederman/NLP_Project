#!/usr/bin/env bash
#SBATCH --job-name=seg-a13b-classic
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a13_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a13_%x_%j.err
# Amendment 13 addendum (post-hoc): classic baselines GPT-2 perplexity and Dawid-Skene. Downloads gpt2 once.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a13b_classic --bootstrap 1000 --threads 16
