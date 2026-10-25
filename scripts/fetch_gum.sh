#!/usr/bin/env bash
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
target="$PROJECT_ROOT/data/UD_English-GUM"
if [[ -s "$target/en_gum-ud-train.conllu" && -s "$target/en_gum-ud-dev.conllu" ]]; then
  echo "GUM already present at $target"
  exit 0
fi
mkdir -p "$target"
for split in train dev; do
  url="https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/en_gum-ud-${split}.conllu"
  part="$target/en_gum-ud-${split}.conllu.part"
  curl --fail --location --retry 3 --output "$part" "$url"
  test -s "$part"
  mv "$part" "$target/en_gum-ud-${split}.conllu"
done
echo "Downloaded official UD English GUM; test split will not be read by the experiment."
