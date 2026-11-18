#!/usr/bin/env bash
#SBATCH --job-name=seg-ewt-test
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/ewt_test_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/ewt_test_%j.err
# PREREGISTRATION amendment 4: one-shot EWT test of the frozen evidence (main: raw/probe+lr).
# Requires data/UD_English-EWT (scripts/fetch_ewt.sh) and the GUM-train probe features
# (scripts/run_boundary_probe.slurm). Order guarantees EWT Gold is read only after every score
# file is frozen, and the layout artifact only after the main (raw) scores are frozen.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
gpu_log="logs/ewt_test_gpu_${SLURM_JOB_ID}.csv"

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 > "$gpu_log" &
smi_pid=$!
trap 'kill $smi_pid 2>/dev/null || true' EXIT

# Unit tests on CPU, concurrently with the gold-free GPU stages; checked before anything is frozen.
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests > "logs/ewt_test_tests_${SLURM_JOB_ID}.log" 2>&1 &
tests_pid=$!

"$python" -m src.ewt_test prepare
"$python" -m src.run_real_splitters --split test --data-root data/real_system_ewt
"$python" -m src.ewt_test evidence --variant raw
if ! wait "$tests_pid"; then
    echo "Unit tests failed; see logs/ewt_test_tests_${SLURM_JOB_ID}.log" >&2
    exit 1
fi
"$python" -m src.ewt_test score --variant raw
"$python" -m src.ewt_test layout
"$python" -m src.ewt_test evidence --variant para
"$python" -m src.ewt_test score --variant para
"$python" -m src.ewt_test gold
"$python" -m src.ewt_test analyze --bootstrap 1000 --workers 16 --gpu-log "$gpu_log"
