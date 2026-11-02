"""Gold-free scoring stage for frozen real-system outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import read_candidate_segmentations, read_raw_documents
from .scoring import soft_fbeta
from .surface_data import load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/real_system/predictions")
    parser.add_argument("--evidence", type=Path, default=ROOT / "data/real_system/bert_dev_evidence.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/real_system/frozen_scores.jsonl")
    args = parser.parse_args()

    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    documents = {doc.doc_id: doc for doc in read_raw_documents(args.data_root / "raw_dev.jsonl")}
    surfaces = {doc.doc_id: doc for doc in load_surface_jsonl(args.data_root / "surface_dev.jsonl")}
    evidence = {record["doc_id"]: record for record in read_jsonl(args.evidence)}
    rows: list[dict] = []
    for system in config["systems"]:
        predictions = read_candidate_segmentations(args.predictions / f"{system}.jsonl")
        for prediction in predictions:
            document = documents[prediction.doc_id]
            prediction.validate_against(document)
            surface = surfaces[prediction.doc_id]
            record = evidence[prediction.doc_id]
            if record["text_sha256"] != document.sha256:
                raise RuntimeError(f"Evidence checksum mismatch for {prediction.doc_id}")
            if record["candidate_offsets"] != surface.candidate_offsets:
                raise RuntimeError(f"Evidence candidate offsets mismatch for {prediction.doc_id}")
            predicted = set(prediction.offsets)
            vector = [int(offset in predicted) for offset in surface.candidate_offsets]
            score = soft_fbeta(
                vector, record["calibrated_boundary_probability"], config["soft_f1_beta"]
            )
            rows.append({
                "doc_id": prediction.doc_id,
                "text_sha256": prediction.text_sha256,
                "system": system,
                "candidate_boundary_offsets": list(prediction.offsets),
                "candidate_boundary_count": len(prediction.offsets),
                "soft_f1": score,
                "score_protocol": config["protocol_version"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "GOLD_FREE_SCORES_FROZEN",
        "rows": len(rows),
        "gold_file_opened": False,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
