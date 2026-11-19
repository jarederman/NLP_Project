#!/usr/bin/env bash
#SBATCH --job-name=seg-a12-gpu
#SBATCH --account=torch_pr_676_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a12_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a12_%x_%j.err
# PREREGISTRATION amendment 12 (GPU, gold-free): prepare EWT dev (flat + newline) and LinES train+dev,
# run the four systems, compute Probe / SaT-12l / ModernBERT evidence with frozen models. No gold is written.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a12_test prepare
for root in data/real_system_ewtdev data/real_system_ewtdev_nl data/real_system_linesx; do
    "$python" -m src.run_real_splitters --split test --data-root "$root" \
        --output-root "$root/predictions" --failures "$root/alignment_failures.jsonl"
    echo "{\"status\": \"A12_SYSTEMS\", \"root\": \"$root\"}"
done
"$python" -m src.a12_test gpu
