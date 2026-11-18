#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/scratch/wj2301/NLP_Project
GIT="$PROJECT_ROOT/scripts/git_project.sh"
cd "$PROJECT_ROOT"

branch="${1:-main}"
remote="${2:-origin}"
current_branch="$($GIT branch --show-current)"
if [[ "$current_branch" != "$branch" ]]; then
  echo "Refusing to sync '$branch' while work tree is on '$current_branch'." >&2
  exit 2
fi

# Fetch first so remote changes are never overwritten. Rebase only when the
# remote branch already exists; otherwise this is the initial publication.
"$GIT" fetch "$remote"
if "$GIT" show-ref --verify --quiet "refs/remotes/$remote/$branch"; then
  "$GIT" rebase "$remote/$branch"
elif [[ "$branch" != "main" ]] && "$GIT" show-ref --verify --quiet "refs/remotes/$remote/main"; then
  # New experiment branches must include the latest protected main before
  # their first publication.
  "$GIT" rebase "$remote/main"
fi
"$GIT" push --set-upstream "$remote" "HEAD:refs/heads/$branch"
