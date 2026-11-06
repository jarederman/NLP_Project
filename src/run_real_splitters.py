from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import read_raw_documents, write_jsonl
from .real_splitters import RealSplitterSuite
from .surface_data import load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--split", choices=["dev", "train"], default="dev",
                        help="train = GUM-train development set for cross-fitting (P1)")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--systems", nargs="*")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    defaults = {
        "dev": (ROOT / "data/real_system/predictions", ROOT / "results/alignment_failures.jsonl"),
        "train": (ROOT / "data/real_system/train_cv/predictions",
                  ROOT / "data/real_system/train_cv/alignment_failures.jsonl"),
    }[args.split]
    args.output_root = args.output_root or defaults[0]
    args.failures = args.failures or defaults[1]

    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    systems = args.systems or config["systems"]
    documents = read_raw_documents(args.data_root / f"raw_{args.split}.jsonl")
    surfaces = load_surface_jsonl(args.data_root / f"surface_{args.split}.jsonl")
    surface_by_id = {doc.doc_id: doc for doc in surfaces}
    if set(surface_by_id) != {doc.doc_id for doc in documents}:
        raise RuntimeError(f"Raw/surface {args.split} document sets differ")
    for document in documents:
        if surface_by_id[document.doc_id].text_sha256 != document.sha256:
            raise RuntimeError(f"Raw/surface checksum mismatch for {document.doc_id}")
    canonical = {
        doc.doc_id: set(surface_by_id[doc.doc_id].candidate_offsets)
        for doc in documents
    }

    suite = RealSplitterSuite(
        stanza_dir=str(ROOT / ".cache/stanza"), use_gpu=not args.cpu
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_failures: list[dict] = []
    for system in systems:
        predictions, failures = suite.predict(system, documents, canonical)
        for prediction in predictions:
            prediction.validate_against(next(
                doc for doc in documents if doc.doc_id == prediction.doc_id
            ))
        write_jsonl(predictions, args.output_root / f"{system}.jsonl")
        all_failures.extend(failure.to_dict() for failure in failures)
        print(json.dumps({
            "stage": "SPLITTER_COMPLETE",
            "system": system,
            "documents": len(predictions),
            "boundaries": sum(len(item.offsets) for item in predictions),
            "alignment_failures": len(failures),
        }), flush=True)

    args.failures.parent.mkdir(parents=True, exist_ok=True)
    with args.failures.open("w", encoding="utf-8") as out:
        for failure in all_failures:
            out.write(json.dumps(failure, ensure_ascii=False) + "\n")
    # Non-gap boundaries (inside a token, or between tokens with no whitespace)
    # are real system output: they are excluded from the
    # gold-free score (no evidence exists there) and charged as Gold false
    # positives by analyze_real_splitters, which reads this file.
    print(json.dumps({
        "status": "REAL_SPLITTERS_COMPLETE",
        "systems": systems,
        "non_gap_boundaries": len(all_failures),
        "failures_file": str(args.failures),
    }), flush=True)


if __name__ == "__main__":
    main()
