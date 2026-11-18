#!/usr/bin/env bash
#SBATCH --job-name=seg-a9-casing-cpu
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a9_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a9_%x_%j.err
# PREREGISTRATION amendment 9.1 (CPU analysis, after run_a9_casing.slurm). Default thread count,
# as in amendments 6-8, so GUM LOGO components reproduce amendment 6 exactly. Writes results/a9/.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a9_casing analyze --bootstrap 1000 --workers 16
