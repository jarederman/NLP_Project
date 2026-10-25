#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/scratch/wj2301/NLP_Project
GIT_METADATA="$PROJECT_ROOT/.gitdata"

if [[ ! -d "$GIT_METADATA" ]]; then
  echo "Missing $GIT_METADATA; repository metadata has not been initialized." >&2
  exit 1
fi

exec git --git-dir="$GIT_METADATA" --work-tree="$PROJECT_ROOT" "$@"
