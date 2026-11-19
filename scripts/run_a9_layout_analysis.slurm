#!/usr/bin/env bash
#SBATCH --job-name=seg-a9-layout-cpu
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a9_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a9_%x_%j.err
# PREREGISTRATION amendments 9.2 + 9.3 (CPU analysis, after run_a9_layout.slurm). Default thread
# count, as in amendments 6-9.1, so the (lower, flat) stack reproduces amendment 6. Writes results/a9/.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a9_layout analyze --bootstrap 1000
