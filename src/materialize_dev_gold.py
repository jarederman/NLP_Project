"""Final-only GUM Dev Gold materialization after predictions/scores are frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import GoldBoundaries, read_raw_documents, write_jsonl
from .surface_data import extract_gold_boundaries, load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/real_data_manifest.json")
    parser.add_argument(
        "--frozen-scores", type=Path,
        default=ROOT / "data/real_system/frozen_scores.jsonl",
    )
    args = parser.parse_args()
    if not args.frozen_scores.exists() or not args.frozen_scores.stat().st_size:
        raise RuntimeError("Refusing to read Dev Gold before gold-free scores are frozen")

    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    source = ROOT / "data/UD_English-GUM/en_gum-ud-dev.conllu"
    surfaces = load_surface_jsonl(args.data_root / "surface_dev.jsonl")
    raw = {doc.doc_id: doc for doc in read_raw_documents(args.data_root / "raw_dev.jsonl")}
    internal_gold = extract_gold_boundaries(source, "dev", surfaces)
    contracts = [GoldBoundaries(
        doc_id=record.doc_id,
        text_sha256=raw[record.doc_id].sha256,
        offsets=tuple(record.boundary_offsets),
        annotation_source=f"UD_English-GUM@{config['gum_release']}",
    ) for record in internal_gold]
    for record in contracts:
        record.validate_against(raw[record.doc_id])
    write_jsonl(contracts, args.data_root / "gold_dev.jsonl")

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        manifest["dev_gold_materialized"] = True
        manifest["dev_gold_materialized_after_frozen_scores"] = True
        manifest["splits"]["dev"]["sentences"] = int(
            sum(record.sentence_count for record in internal_gold)
        )
        manifest["splits"]["dev"]["gold_boundaries"] = int(
            sum(len(record.boundary_offsets) for record in internal_gold)
        )
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": "DEV_GOLD_MATERIALIZED_AFTER_FREEZE",
        "documents": len(contracts),
        "boundaries": sum(len(record.offsets) for record in contracts),
        "frozen_scores": str(args.frozen_scores),
    }), flush=True)


if __name__ == "__main__":
    main()
