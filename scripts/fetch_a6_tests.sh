#!/usr/bin/env bash
# Amendment 6 one-shot test corpora: UD English GENTLE (main) and LinES (secondary) test, UD r2.18.
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
ud_release="r2.18"
for spec in GENTLE:en_gentle LinES:en_lines; do
  name=${spec%%:*}; file="${spec##*:}-ud-test.conllu"
  target="$PROJECT_ROOT/data/UD_English-$name"
  if [[ -s "$target/$file" && -s "$target/UD_RELEASE" && "$(<"$target/UD_RELEASE")" == "$ud_release" ]]; then
    echo "$name $ud_release test data already present"; continue
  fi
  mkdir -p "$target"
  curl --fail --location --retry 3 --output "$target/$file.part" \
    "https://raw.githubusercontent.com/UniversalDependencies/UD_English-$name/${ud_release}/$file"
  test -s "$target/$file.part"
  mv "$target/$file.part" "$target/$file"
  printf '%s\n' "$ud_release" > "$target/UD_RELEASE"
  sha256sum "$target/$file" > "$target/SHA256SUMS"
  echo "Downloaded UD English $name $ud_release test data and recorded its SHA256 checksum."
done
