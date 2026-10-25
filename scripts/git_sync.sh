#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/scratch/wj2301/NLP_Project
GIT="$PROJECT_ROOT/scripts/git_project.sh"
cd "$PROJECT_ROOT"

branch="${1:-main}"
remote="${2:-origin}"

# Fetch first so remote changes are never overwritten. Rebase only when the
# remote branch already exists; otherwise this is the initial publication.
"$GIT" fetch "$remote"
if "$GIT" show-ref --verify --quiet "refs/remotes/$remote/$branch"; then
  "$GIT" rebase "$remote/$branch"
fi
"$GIT" push --set-upstream "$remote" "$branch"
