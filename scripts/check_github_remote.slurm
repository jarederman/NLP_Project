#!/usr/bin/env bash
#SBATCH --job-name=nlp-git-check
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/git_check_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/git_check_%j.err
set -euo pipefail

PROJECT_ROOT=/scratch/wj2301/NLP_Project
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/.runtime/ssh"
chmod 700 "$PROJECT_ROOT/.runtime/ssh"
cd "$PROJECT_ROOT"
scripts/git_project.sh ls-remote --heads origin
echo "REMOTE_CHECK_OK"
