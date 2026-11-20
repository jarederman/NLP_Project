#!/usr/bin/env bash
#SBATCH --job-name=seg-review-nested
#SBATCH --account=torch_pr_676_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/review_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/review_%x_%j.err
# Code review 2026-11-20, finding 1 (diagnostic, GUM train only): nested LOGO refits of the E0 probe from the
# cached frozen-BERT features. Writes data/lm_cache/review_fixes/e0_nested.npz; no test data, no gold of a test set.
# Short GPU job: general account, no partition (user decision 2026-11-18); may be twin-submitted (lock below).
# Also runs on CPU (the solver falls back to CPU, ~1-1.5 h on 16 cores):
#   sbatch --partition=cpu_short --gres=none --cpus-per-task=16 --mem=32G --time=04:00:00 scripts/run_review_nested_gpu.slurm
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
lock="$PROJECT_ROOT/.runtime/review_nested_gpu.lock"
if ! mkdir "$lock" 2>/dev/null; then
    echo "REVIEW_TWIN: $lock is held by job $(cat "$lock/job" 2>/dev/null); exiting" >&2
    exit 1
fi
echo "${SLURM_JOB_ID:-local}" > "$lock/job"
trap 'rm -rf "$lock"' EXIT
scancel --user="$(id -un)" --name="${SLURM_JOB_NAME:-seg-review-nested}" --state=PENDING || true
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests > "logs/review_nested_tests_${SLURM_JOB_ID:-local}.log" 2>&1 \
    || { echo "Unit tests failed; see logs/review_nested_tests_${SLURM_JOB_ID:-local}.log" >&2; exit 1; }
"$python" -m src.review_nested gpu
