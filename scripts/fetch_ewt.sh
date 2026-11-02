#!/usr/bin/env bash
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"

target="$PROJECT_ROOT/data/UD_English-EWT"
ud_release="r2.18"
release_marker="$target/UD_RELEASE"
if [[ -s "$target/en_ewt-ud-test.conllu" && -s "$release_marker" && \
      "$(<"$release_marker")" == "$ud_release" ]]; then
  echo "EWT $ud_release test data already present at $target"
  exit 0
fi

mkdir -p "$target"
url="https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/${ud_release}/en_ewt-ud-test.conllu"
part="$target/en_ewt-ud-test.conllu.part"
curl --fail --location --retry 3 --output "$part" "$url"
test -s "$part"
mv "$part" "$target/en_ewt-ud-test.conllu"
printf '%s\n' "$ud_release" > "$release_marker"
sha256sum "$target/en_ewt-ud-test.conllu" > "$target/SHA256SUMS"
echo "Downloaded frozen UD English EWT $ud_release test data and recorded its SHA256 checksum."
