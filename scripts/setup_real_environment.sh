#!/usr/bin/env bash
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"

base_python=/scratch/wj2301/conda_envs/jit/bin/python
if [[ ! -x .venv/bin/python ]]; then
  "$base_python" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --upgrade -r requirements-real.txt
.venv/bin/python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab
.venv/bin/python - <<'PY'
import os
import stanza
stanza.download("en", model_dir=os.environ["STANZA_RESOURCES_DIR"], processors="tokenize", verbose=False)
PY
scripts/fetch_gum.sh
.venv/bin/python - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
names = ["nltk", "stanza", "wtpsplit", "torch", "transformers", "numpy", "scipy"]
versions = {name: metadata.version(name) for name in names}
Path(".runtime/real_env_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
print(json.dumps({"status": "REAL_ENVIRONMENT_READY", "versions": versions}))
PY
