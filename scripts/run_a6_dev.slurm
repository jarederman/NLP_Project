#!/usr/bin/env bash
#SBATCH --job-name=seg-a6-dev
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a6_dev_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a6_dev_%j.err
# PREREGISTRATION amendment 6, development on GUM train (LOGO). Gold-free GPU stages first
# (SaT zero-shot, stability), the unit tests on CPU concurrently; then the fine-tuning (GUM-train
# labels) and the CPU combination / selection, which also fits the components for the test.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
python=.venv/bin/python
gpu_log="logs/a6_dev_gpu_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 5 > "$gpu_log" &
smi_pid=$!
trap 'kill $smi_pid 2>/dev/null || true' EXIT

OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests > "logs/a6_dev_tests_${SLURM_JOB_ID}.log" 2>&1 &
tests_pid=$!
"$python" -m src.a6_gpu sat --corpus gum
"$python" -m src.a6_gpu stability --corpus gum
if ! wait "$tests_pid"; then
    echo "Unit tests failed; see logs/a6_dev_tests_${SLURM_JOB_ID}.log" >&2
    exit 1
fi
"$python" -m src.a6_gpu finetune --corpus gum
"$python" -m src.a6_combine dev --bootstrap 1000 --workers 16
