#!/usr/bin/env bash
#SBATCH --job-name=seg-a6-test
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a6_test_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a6_test_%j.err
# PREREGISTRATION amendment 6: one-shot test on GENTLE (main) and LinES (secondary). Runs ONCE,
# only after a successful development run (results/a6/dev_report.json: dev_success = true).
# Every score is frozen before any test Gold is materialized.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
python=.venv/bin/python
"$python" -c "import json,sys; sys.exit(0 if json.load(open('results/a6/dev_report.json'))['dev_success'] else 1)" \
    || { echo "Development criteria not met; the test is not run" >&2; exit 1; }
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests > "logs/a6_test_tests_${SLURM_JOB_ID}.log" 2>&1 \
    || { echo "Unit tests failed; see logs/a6_test_tests_${SLURM_JOB_ID}.log" >&2; exit 1; }
for corpus in gentle lines; do
    root="data/real_system_${corpus}"
    "$python" -m src.a6_test prepare --corpus "$corpus"
    "$python" -m src.run_real_splitters --split test --data-root "$root" \
        --output-root "$root/predictions" --failures "$root/alignment_failures.jsonl"
    "$python" -m src.a6_test e0 --corpus "$corpus"
    "$python" -m src.a6_gpu sat --corpus "$corpus"
    "$python" -m src.a6_gpu finetune --corpus "$corpus"
    "$python" -m src.a6_gpu stability --corpus "$corpus"
done
for corpus in gentle lines; do
    "$python" -m src.a6_combine freeze --corpus "$corpus"
done
for corpus in gentle lines; do
    "$python" -m src.a6_test gold --corpus "$corpus"
done
"$python" -m src.a6_combine analyze --corpora gentle lines --bootstrap 1000 --workers 16
