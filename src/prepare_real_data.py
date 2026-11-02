from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .contracts import GoldBoundaries as ContractGoldBoundaries
from .contracts import RawDocument, write_jsonl as write_contracts
from .surface_data import (
    extract_gold_boundaries,
    load_surface_documents,
    save_jsonl as write_surfaces,
)


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/real_data_manifest.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    data_root = ROOT / "data/UD_English-GUM"
    release_marker = data_root / "UD_RELEASE"
    if not release_marker.exists() or release_marker.read_text().strip() != config["gum_release"]:
        raise RuntimeError(
            f"Expected frozen GUM {config['gum_release']}; run scripts/fetch_gum.sh first"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "protocol_version": config["protocol_version"],
        "source": "UniversalDependencies/UD_English-GUM",
        "release": config["gum_release"],
        "test_read": False,
        "splits": {},
    }
    for split in ("train", "dev"):
        source = data_root / f"en_gum-ud-{split}.conllu"
        if not source.exists():
            raise FileNotFoundError(source)
        surfaces = load_surface_documents(source, split)
        evaluation_gold = extract_gold_boundaries(source, split, surfaces)
        if args.limit:
            surfaces = surfaces[:args.limit]
            selected = {doc.doc_id for doc in surfaces}
            evaluation_gold = [gold for gold in evaluation_gold if gold.doc_id in selected]
        if split == "dev" and args.limit is None and len(surfaces) != config["expected_dev_documents"]:
            raise RuntimeError(
                f"Expected {config['expected_dev_documents']} GUM dev documents, found {len(surfaces)}"
            )

        raw_documents = [RawDocument.create(
            doc_id=doc.doc_id,
            corpus="UD_English-GUM",
            split=split,
            genre=doc.genre,
            text=doc.text,
            source_ref=f"UniversalDependencies/UD_English-GUM@{config['gum_release']}",
        ) for doc in surfaces]
        raw_by_id = {doc.doc_id: doc for doc in raw_documents}
        gold_contracts = [ContractGoldBoundaries(
            doc_id=gold.doc_id,
            text_sha256=raw_by_id[gold.doc_id].sha256,
            offsets=tuple(gold.boundary_offsets),
            annotation_source=f"UD_English-GUM@{config['gum_release']}",
        ) for gold in evaluation_gold]
        for record in gold_contracts:
            record.validate_against(raw_by_id[record.doc_id])

        write_surfaces(surfaces, args.output_root / f"surface_{split}.jsonl")
        write_contracts(raw_documents, args.output_root / f"raw_{split}.jsonl")
        write_contracts(gold_contracts, args.output_root / f"gold_{split}.jsonl")
        manifest["splits"][split] = {
            "file": str(source.relative_to(ROOT)),
            "file_sha256": file_sha256(source),
            "documents": len(surfaces),
            "sentences": int(sum(gold.sentence_count for gold in evaluation_gold)),
            "surface_tokens": int(sum(len(doc.tokens) for doc in surfaces)),
            "candidate_positions": int(sum(len(doc.candidate_offsets) for doc in surfaces)),
            "gold_boundaries": int(sum(len(gold.boundary_offsets) for gold in evaluation_gold)),
            "document_ids": [doc.doc_id for doc in surfaces],
        }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "REAL_DATA_PREPARED", **manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
