#!/usr/bin/env bash
#SBATCH --job-name=seg-e5-candidates
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/wj2301/NLP_Project/logs/e5_%x_%j.out
#SBATCH --error=/scratch/wj2301/NLP_Project/logs/e5_%x_%j.err
# E5 (2026-11-19): fetch candidate fresh test treebanks (UD r2.18) and report STRUCTURE ONLY
# (documents, paragraphs, sentences, tokens, genres). No splitter, evidence or score is computed,
# and no boundary-quality statistic is reported, so the choice cannot depend on difficulty.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
release=r2.18
base=data/UD_candidates
declare -A files=(
  [UD_English-ParTUT]="en_partut-ud-train.conllu en_partut-ud-dev.conllu en_partut-ud-test.conllu"
  [UD_English-PUD]="en_pud-ud-test.conllu"
  [UD_English-EWT]="en_ewt-ud-dev.conllu"
  [UD_English-LinES]="en_lines-ud-train.conllu en_lines-ud-dev.conllu"
  [UD_English-ESLSpok]="en_eslspok-ud-train.conllu en_eslspok-ud-dev.conllu en_eslspok-ud-test.conllu"
  [UD_English-CTeTex]="en_ctetex-ud-train.conllu en_ctetex-ud-dev.conllu en_ctetex-ud-test.conllu"
)
for tb in "${!files[@]}"; do
  mkdir -p "$base/$tb"
  for f in ${files[$tb]}; do
    url="https://raw.githubusercontent.com/UniversalDependencies/$tb/$release/$f"
    if curl --fail --silent --location --retry 3 --output "$base/$tb/$f.part" "$url"; then
      mv "$base/$tb/$f.part" "$base/$tb/$f"; echo "FETCHED $tb/$f"
    else
      rm -f "$base/$tb/$f.part"; echo "MISSING $tb/$f"
    fi
  done
  echo "$release" > "$base/$tb/UD_RELEASE"
  (cd "$base/$tb" && sha256sum *.conllu > SHA256SUMS 2>/dev/null || true)
done
.venv/bin/python - <<'PY'
import json, re
from collections import Counter
from pathlib import Path
base = Path("data/UD_candidates"); out = {}
for f in sorted(base.glob("*/*.conllu")):
    docs, pars, sents, toks, text_missing, doc_sizes, genres = 0, 0, 0, 0, 0, [], Counter()
    cur = 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("# newdoc"):
            if cur: doc_sizes.append(cur)
            docs += 1; cur = 0
            m = re.search(r"id\s*=\s*([A-Za-z_]+?)[-_0-9]", line); genres[m.group(1) if m else "?"] += 1
        elif line.startswith("# newpar"):
            pars += 1
        elif line.startswith("# sent_id"):
            sents += 1; cur += 1
        elif line and line[0].isdigit() and "\t" in line and not re.match(r"^\d+[-.]", line):
            toks += 1
            if line.split("\t")[1] == "_": text_missing += 1
    if cur: doc_sizes.append(cur)
    out[str(f.relative_to(base))] = {"sentences": sents, "tokens": toks, "newdoc": docs, "newpar": pars,
        "tokens_without_form": text_missing, "sentences_per_doc_median": (sorted(doc_sizes)[len(doc_sizes)//2] if doc_sizes else None),
        "docs_with_2plus_sentences": sum(s >= 2 for s in doc_sizes), "genre_prefixes": dict(genres.most_common(8))}
Path("results/e5_candidates.json").write_text(json.dumps(out, indent=2) + "\n")
for k, v in out.items(): print(k, json.dumps(v))
PY
