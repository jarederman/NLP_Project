#!/usr/bin/env bash
#SBATCH --job-name=nlp-git-sync
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/git_sync_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/git_sync_%j.err
set -euo pipefail

PROJECT_ROOT=/scratch/wj2301/NLP_Project
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
branch="${GIT_SYNC_BRANCH:-}"
if [[ -z "$branch" ]]; then
  echo "Set GIT_SYNC_BRANCH explicitly when submitting this job." >&2
  exit 2
fi
if [[ "$branch" == "main" ]]; then
  echo "This project sync job does not push protected main; use a review branch." >&2
  exit 2
fi
scripts/git_sync.sh "$branch" origin
echo "GITHUB_SYNC_OK"
