#!/usr/bin/env bash
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
target="$PROJECT_ROOT/data/UD_English-GUM"
ud_release="r2.18"
release_marker="$target/UD_RELEASE"
if [[ -s "$target/en_gum-ud-train.conllu" && -s "$target/en_gum-ud-dev.conllu" && \
      -s "$release_marker" && "$(<"$release_marker")" == "$ud_release" ]]; then
  echo "GUM $ud_release already present at $target"
  exit 0
fi
mkdir -p "$target"
for split in train dev; do
  url="https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/${ud_release}/en_gum-ud-${split}.conllu"
  part="$target/en_gum-ud-${split}.conllu.part"
  curl --fail --location --retry 3 --output "$part" "$url"
  test -s "$part"
  mv "$part" "$target/en_gum-ud-${split}.conllu"
done
printf '%s\n' "$ud_release" > "$release_marker"
sha256sum "$target"/en_gum-ud-{train,dev}.conllu > "$target/SHA256SUMS"
echo "Downloaded official UD English GUM $ud_release and recorded SHA256 checksums; test remains unread."
