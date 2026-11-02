#!/usr/bin/env bash
#SBATCH --job-name=seg-real-setup
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/real_setup_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/real_setup_%j.err
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
scripts/setup_real_environment.sh
