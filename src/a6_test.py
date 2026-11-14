"""One-shot test corpora of amendment 6 (GENTLE main, LinES secondary): gold-free stages and the
Gold materialization that is allowed only after every score is frozen.

  prepare   CoNLL-U -> surface/raw artifacts (sentence boundaries are not used for the text)
  e0        the frozen main evidence (raw/probe+lr; the model fitted on all of GUM train in
            amendment 4, reused unchanged)
  gold      Gold artifact, only if ``a6_combine freeze`` wrote the frozen scores (amendment 4a
            handling of non-whitespace Gold boundaries)

Splitters run through ``src.run_real_splitters``; the other evidence through ``src.a6_gpu``;
scores through ``src.a6_combine freeze``; the analysis through ``src.a6_combine analyze``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

import numpy as np

from .a6_data import CORPORA, ROOT, TEST_SOURCES, UD_RELEASE, write_jsonl
from .contracts import GoldBoundaries, RawDocument, read_raw_documents, write_jsonl as write_contracts
from .surface_data import extract_gold_boundaries, load_surface_documents, load_surface_jsonl, save_jsonl


def source(name: str):
    folder, file = TEST_SOURCES[name]
    path = ROOT / "data" / folder / file
    if (ROOT / "data" / folder / "UD_RELEASE").read_text().strip() != UD_RELEASE:
        raise RuntimeError(f"Expected {folder} {UD_RELEASE}; run scripts/fetch_a6_tests.sh")
    return folder, path


def genre_of(name: str, doc_id: str) -> str:
    return doc_id.split("_")[1] if name == "gentle" else "lines"


def manifest_path(name: str):
    return ROOT / "results/a6" / f"{name}_manifest.json"


def prepare(args) -> None:
    corpus = CORPORA[args.corpus]
    folder, path = source(args.corpus)
    docs = load_surface_documents(path, "test")
    for d in docs:
        d.genre = genre_of(args.corpus, d.doc_id)
    raw = [RawDocument.create(doc_id=d.doc_id, corpus=folder, split="test", genre=d.genre, text=d.text,
                              source_ref=f"UniversalDependencies/{folder}@{UD_RELEASE}") for d in docs]
    corpus.data_root.mkdir(parents=True, exist_ok=True)
    save_jsonl(docs, corpus.data_root / "surface_test.jsonl")
    write_contracts(raw, corpus.data_root / "raw_test.jsonl")
    manifest = {"source": f"UniversalDependencies/{folder}", "release": UD_RELEASE, "file": str(path.relative_to(ROOT)),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "documents": len(docs),
                "genres": dict(sorted(Counter(d.genre for d in docs).items())),
                "candidate_positions": sum(len(d.candidate_offsets) for d in docs), "gold_materialized": False}
    manifest_path(args.corpus).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(args.corpus).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "A6_TEST_PREPARED", "corpus": args.corpus, **manifest}), flush=True)


def e0(args) -> None:
    """Frozen main evidence on the test corpus: same extraction as the EWT test, same model."""
    import torch

    from .boundary_probe import extract, predict_logistic
    from .ewt_test import _blocks

    corpus = CORPORA[args.corpus]
    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    docs = sorted(load_surface_jsonl(corpus.data_root / "surface_test.jsonl"), key=lambda d: d.doc_id)
    raw = {d.doc_id: d for d in read_raw_documents(corpus.data_root / "raw_test.jsonl")}
    for d in docs:
        if raw[d.doc_id].sha256 != d.text_sha256:
            raise RuntimeError(f"Raw/surface checksum mismatch for {d.doc_id}")
    ns = argparse.Namespace(mask_batch=512, lr_batch=1024, feature_root=corpus.a6_root / "features")
    features, _ = extract(ns, torch, config, docs, {}, None, variants=("raw",), feature_name="test.pt")
    model = torch.load(ROOT / "data/real_system_ewt/models/raw_probe+lr.pt", weights_only=False)
    model = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in model.items()}
    blocks = _blocks(torch, features["raw"], "cuda")
    p = predict_logistic(torch, model, torch.cat([blocks[b] for b in ("logit", "hidden", "lr")], dim=1))
    p = p.cpu().numpy().astype(float)
    rows, start = [], 0
    for d in docs:
        n = len(d.candidate_offsets)
        rows.append({"doc_id": d.doc_id, "text_sha256": d.text_sha256, "candidate_offsets": d.candidate_offsets,
                     "calibrated_boundary_probability": p[start:start + n].tolist(), "config": "probe+lr"})
        start += n
    write_jsonl(rows, corpus.path("e0_probe+lr.jsonl"))
    print(json.dumps({"status": "A6_TEST_E0", "corpus": args.corpus, "gaps": int(len(p))}), flush=True)


def gold(args) -> None:
    corpus = CORPORA[args.corpus]
    frozen = corpus.path("frozen_scores.jsonl")
    if not frozen.exists() or not frozen.stat().st_size:
        raise RuntimeError("Refusing to read test Gold before the scores are frozen")
    _, path = source(args.corpus)
    docs = load_surface_jsonl(corpus.data_root / "surface_test.jsonl")
    raw = {d.doc_id: d for d in read_raw_documents(corpus.data_root / "raw_test.jsonl")}
    internal = extract_gold_boundaries(path, "test", docs, allow_non_gap=True)       # amendment 4a
    gaps = {d.doc_id: set(d.candidate_offsets) for d in docs}
    non_gap = sum(1 for r in internal for o in r.boundary_offsets if o not in gaps[r.doc_id])
    contracts = [GoldBoundaries(doc_id=r.doc_id, text_sha256=raw[r.doc_id].sha256, offsets=tuple(r.boundary_offsets),
                                annotation_source=f"{TEST_SOURCES[args.corpus][0]}@{UD_RELEASE}") for r in internal]
    for record in contracts:
        record.validate_against(raw[record.doc_id])
    write_contracts(contracts, corpus.data_root / "gold_test.jsonl")
    manifest = json.loads(manifest_path(args.corpus).read_text())
    manifest.update({"gold_materialized": True, "gold_materialized_after_frozen_scores": True,
                     "sentences": int(sum(r.sentence_count for r in internal)),
                     "gold_boundaries": int(sum(len(r.boundary_offsets) for r in internal)),
                     "gold_boundaries_not_at_whitespace_gaps": int(non_gap)})
    manifest_path(args.corpus).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "A6_TEST_GOLD_AFTER_FREEZE", "corpus": args.corpus,
                      "non_gap_gold": int(non_gap)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "e0", "gold"])
    parser.add_argument("--corpus", choices=["gentle", "lines"], required=True)
    args = parser.parse_args()
    {"prepare": prepare, "e0": e0, "gold": gold}[args.stage](args)


if __name__ == "__main__":
    main()
