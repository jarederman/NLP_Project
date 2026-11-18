#!/usr/bin/env bash
#SBATCH --job-name=fetch-wtpsplit
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/fetch_wtpsplit_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/fetch_wtpsplit_%j.err
# Clone the SaT / WtP source (segment-any-text/wtpsplit, MIT) into third_party/ for analysis only.
# third_party/ is gitignored; the exact commit is recorded in third_party/wtpsplit.COMMIT.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
mkdir -p third_party
if [[ -d third_party/wtpsplit/.git ]]; then
  git -C third_party/wtpsplit fetch --tags origin
else
  git clone https://github.com/segment-any-text/wtpsplit.git third_party/wtpsplit
fi
git -C third_party/wtpsplit log -1 --format='%H %cd %s' > third_party/wtpsplit.COMMIT
git -C third_party/wtpsplit tag --list | tail -20 >> third_party/wtpsplit.COMMIT
cat third_party/wtpsplit.COMMIT
echo FETCH_WTPSPLIT_OK
