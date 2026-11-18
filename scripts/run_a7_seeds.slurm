#!/usr/bin/env bash
#SBATCH --job-name=seg-a7-seeds
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a7_seeds_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a7_seeds_%j.err
# PREREGISTRATION amendment 7.4: E2 (ft-modernbert) LOGO re-trained with fold seeds
# 20261102 + 100k + g, k = 1, 2, 3. LOGO only; no final model, no test component touched.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests > "logs/a7_seeds_tests_${SLURM_JOB_ID}.log" 2>&1 &
tests_pid=$!
for k in 1 2 3; do
    "$python" -m src.a6_gpu finetune --corpus gum --models ft-modernbert --seed-offset "$k"
done
if ! wait "$tests_pid"; then
    echo "Unit tests failed; see logs/a7_seeds_tests_${SLURM_JOB_ID}.log" >&2
    exit 1
fi
