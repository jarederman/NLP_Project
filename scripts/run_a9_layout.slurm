#!/usr/bin/env bash
#SBATCH --job-name=seg-a9-layout-gpu
#SBATCH --account=torch_pr_676_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/a9_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/a9_%x_%j.err
# PREREGISTRATION amendments 9.2 + 9.3 (GPU): newline corpora (prepare), the four systems on
# newline text, sat-12l on newline text, ModernBERT E2 for cells (cased,flat), (lower,nl), (cased,nl).
# Short-ish GPU job: general account, no partition (user decision 2026-11-18).
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
# Twin submission (2026-11-18, queue congestion): the same script may be queued under two accounts
# (general and tandon_advanced/h200_tandon). The first twin to start cancels the other pending one;
# the lock directory stops both from running if they start at the same moment.
lock="$PROJECT_ROOT/.runtime/a9_layout_gpu.lock"
if ! mkdir "$lock" 2>/dev/null; then
    echo "A9_LAYOUT_TWIN: $lock is held by job $(cat "$lock/job" 2>/dev/null); exiting" >&2
    exit 1
fi
echo "${SLURM_JOB_ID:-local}" > "$lock/job"
trap 'rm -rf "$lock"' EXIT
scancel --user="$(id -un)" --name="${SLURM_JOB_NAME:-seg-a9-layout-gpu}" --state=PENDING || true
export HF_HUB_OFFLINE=1
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
"$python" -m src.a9_layout prepare
for pair in gum:train:data/real_system_nl gentle:test:data/real_system_gentle_nl \
            lines:test:data/real_system_lines_nl ewt:test:data/real_system_ewt_nl; do
    IFS=: read -r corpus split root <<< "$pair"
    "$python" -m src.run_real_splitters --split "$split" --data-root "$root" \
        --output-root "$root/predictions" --failures "$root/alignment_failures.jsonl"
    echo "{\"status\": \"A9_NL_SYSTEMS\", \"corpus\": \"$corpus\"}"
done
"$python" -m src.a9_layout gpu
