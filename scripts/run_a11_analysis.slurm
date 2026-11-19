#!/usr/bin/env bash
#SBATCH --job-name=seg-a11-analysis
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a11_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a11_%x_%j.err
# PREREGISTRATION amendment 11 (post-hoc, no decisions): evidence ladder cache, then E1-E4 and perturbation test v2.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a11_analysis evidence --threads 16
"$python" -m src.a11_analysis analyze --bootstrap 1000
