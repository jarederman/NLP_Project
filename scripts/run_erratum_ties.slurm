#!/usr/bin/env bash
#SBATCH --job-name=seg-erratum-ties
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/erratum_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/erratum_%x_%j.err
# Erratum 2026-11-19: tie-exact re-evaluation of amendments 6-9 (CPU analysis stages only; no GPU,
# no test inference, frozen components untouched). Untied reports archived in results/erratum_ties/untied.
#   sbatch --export=ALL,ERRATUM_STAGE=a6 scripts/run_erratum_ties.slurm      (first)
#   then ERRATUM_STAGE=a7 / a8 / a9 (after a6; they read results/a6/dev_report.json)
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python
OMP_NUM_THREADS=4 "$python" -m unittest discover -s tests
case "${ERRATUM_STAGE:?set ERRATUM_STAGE to a6, a7, a8 or a9}" in
  a6) "$python" -m src.a6_combine dev --erratum --bootstrap 1000 --workers 16
      "$python" -m src.a6_combine analyze --corpora gentle lines --bootstrap 1000 --workers 16 ;;
  a7) "$python" -m src.a7_posthoc posthoc --bootstrap 1000 --workers 16
      "$python" -m src.a7_posthoc seeds --bootstrap 1000 --workers 16 ;;
  a8) "$python" -m src.a8_supplement --bootstrap 1000 --workers 16 ;;
  a9) "$python" -m src.a9_casing analyze --bootstrap 1000 --workers 16
      "$python" -m src.a9_layout analyze --bootstrap 1000 ;;
  *) echo "unknown stage" >&2; exit 2 ;;
esac
echo "{\"status\": \"ERRATUM_STAGE_DONE\", \"stage\": \"$ERRATUM_STAGE\"}"
