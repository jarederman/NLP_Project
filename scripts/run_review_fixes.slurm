#!/usr/bin/env bash
#SBATCH --job-name=seg-review-fixes
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/review_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/review_%x_%j.err
# Code review 2026-11-20: rerun the CPU analysis stages affected by the fixes (Dawid-Skene leave-one-out,
# LOGO PuncEv on GUM, Q_end in the perturbation test, tie-aware top-1). Frozen models, evidence and one-shot
# frozen scores are never rewritten (a12 runs only its analyze stage). The pre-fix reports are archived once
# in results/review_fixes/before/. Then the diagnostics (review_checks) and the old/new comparison.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests > "logs/review_fixes_tests_${SLURM_JOB_ID:-local}.log" 2>&1 \
    || { echo "Unit tests failed; see logs/review_fixes_tests_${SLURM_JOB_ID:-local}.log" >&2; exit 1; }
before=results/review_fixes/before
if [ ! -d "$before" ]; then
    mkdir -p "$before"/{a8,a11,a12,a13}
    cp results/a8/report.json results/a8/summary.md "$before/a8/"
    cp results/a11/report.json results/a11/summary.md results/a11/examples.md "$before/a11/"
    cp results/a12/report.json results/a12/summary.md "$before/a12/"
    cp results/a13/*.json results/a13/*.md "$before/a13/"
    echo '{"status": "REVIEW_ARCHIVED"}'
fi
"$python" -m src.a8_supplement --bootstrap 1000 --workers 16
"$python" -m src.a11_analysis evidence --threads 16
"$python" -m src.a11_analysis analyze --bootstrap 1000
"$python" -m src.a12_test analyze --bootstrap 1000
"$python" -m src.a13_baselines analyze --bootstrap 1000
"$python" -m src.a13_baselines extra --bootstrap 1000
"$python" -m src.a13b_classic --bootstrap 1000 --threads 16
"$python" -m src.a13c_softhard --bootstrap 1000
"$python" -m src.review_checks --bootstrap 1000 --workers 16 --threads 16
"$python" -m src.review_compare
echo '{"status": "REVIEW_FIXES_DONE"}'
