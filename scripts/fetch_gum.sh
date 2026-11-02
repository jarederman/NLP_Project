#!/usr/bin/env bash
set -euo pipefail
source /scratch/wj2301/NLP_Project/scripts/env.sh
cd "$PROJECT_ROOT"
target="$PROJECT_ROOT/data/UD_English-GUM"
ud_release="r2.18"
expected_dev_documents=32
release_marker="$target/UD_RELEASE"
if [[ -s "$target/en_gum-ud-train.conllu" && -s "$target/en_gum-ud-dev.conllu" && \
      -s "$release_marker" && "$(<"$release_marker")" == "$ud_release" && \
      "$(grep -c '^# newdoc id =' "$target/en_gum-ud-dev.conllu")" == "$expected_dev_documents" ]]; then
  echo "GUM $ud_release already present at $target"
  exit 0
fi
mkdir -p "$target"
for split in train dev; do
  # The fixed query key avoids a stale CDN object observed for the moved r2.18
  # tag while retaining an immutable release ref in the actual path.
  url="https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/${ud_release}/en_gum-ud-${split}.conllu?release=gum-v12.1-r2.18"
  part="$target/en_gum-ud-${split}.conllu.part"
  curl --fail --location --retry 3 --output "$part" "$url"
  test -s "$part"
  mv "$part" "$target/en_gum-ud-${split}.conllu"
done
actual_dev_documents="$(grep -c '^# newdoc id =' "$target/en_gum-ud-dev.conllu")"
if [[ "$actual_dev_documents" != "$expected_dev_documents" ]]; then
  echo "Expected $expected_dev_documents GUM dev documents in $ud_release, found $actual_dev_documents" >&2
  exit 1
fi
printf '%s\n' "$ud_release" > "$release_marker"
sha256sum "$target"/en_gum-ud-{train,dev}.conllu > "$target/SHA256SUMS"
echo "Downloaded official UD English GUM $ud_release and recorded SHA256 checksums; test remains unread."
