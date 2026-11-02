#!/usr/bin/env bash
#SBATCH --job-name=seg-real-formal
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/real_formal_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/real_formal_%j.err
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
python=.venv/bin/python

"$python" -m src.prepare_real_data
"$python" -m src.run_real_splitters
"$python" -m src.run_surface_bert
"$python" -m src.score_real_splitters
"$python" -m src.analyze_real_splitters
"$python" - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
versions = {name: metadata.version(name) for name in
            ["nltk", "stanza", "wtpsplit", "torch", "transformers", "numpy", "scipy"]}
payload = {
    "status": "REAL_FORMAL_COMPLETE",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu": os.environ.get("SLURM_JOB_GPUS"),
    "versions": versions,
}
Path("results/real_run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
PY
