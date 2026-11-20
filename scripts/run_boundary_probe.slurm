#!/usr/bin/env bash
#SBATCH --job-name=seg-boundary-probe
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/boundary_probe_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/boundary_probe_%j.err
# PREREGISTRATION amendment 3: [MASK]-state probe (1), insertion likelihood ratio (2), both (1+2),
# each with and without paragraph breaks as [SEP] (layout control, 4). GUM train only.
# Requires a completed run_train_cv.slurm (frozen candidates and folds).
#
# GPU utilisation: the unit tests run on CPU *concurrently* with the GPU extraction (which writes
# only to new paths); the job stops before cross-fitting if they fail. nvidia-smi samples the GPU
# every 2 s into logs/boundary_probe_gpu_<job>.csv.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
gpu_log="logs/boundary_probe_gpu_${SLURM_JOB_ID}.csv"

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 > "$gpu_log" &
smi_pid=$!
trap 'kill $smi_pid 2>/dev/null || true' EXIT

OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests > "logs/boundary_probe_tests_${SLURM_JOB_ID}.log" 2>&1 \
    || { echo "Unit tests failed; see logs/boundary_probe_tests_${SLURM_JOB_ID}.log" >&2; exit 1; }
"$python" -m src.layout_data

"$python" -m src.boundary_probe --stop-after-extraction
"$python" -m src.boundary_probe --from-features
"$python" -m src.boundary_probe_eval --workers 16 --gpu-log "$gpu_log"
