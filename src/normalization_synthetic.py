"""Normalized vs raw Soft F1 on the preliminary synthetic-perturbation candidates.

Controlled comparison (same evidence, same candidates, only the scorer changes)
in the regime with large quality differences: seeded Gold/Over/Under/Shift/Mixed
candidates on the preliminary study's controlled GUM dev texts, for both cached
evidence sources (Qwen left-context, BERT bidirectional), Platt fitted on train.
GUM dev was the preliminary study's evaluation set; this is a re-analysis of
existing caches and informs no configuration choice. CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .candidates import generate_candidates
from .data import load_documents
from .quality_agreement import build_items, evaluate, render
from .scoring import PlattCalibrator
from .selection import ScoredSegmentation


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"qwen_left_context": "qwen_punctuation.jsonl", "bert_bidirectional": "bert_bidirectional_punctuation.jsonl"}


def main() -> None:
    config = json.loads((ROOT / "configs/preliminary.json").read_text())
    train_docs = load_documents(ROOT / "data/processed/gum_train_documents.jsonl")
    dev_docs = load_documents(ROOT / "data/processed/gum_dev_documents.jsonl")
    train_y = np.concatenate([d.gold for d in train_docs])
    out_root = ROOT / "results/quality_agreement"
    for source, filename in SOURCES.items():
        cache = {}
        with (ROOT / "data/lm_cache" / filename).open() as handle:
            for line in handle:
                record = json.loads(line)
                cache[record["doc_id"]] = record
        calibrator = PlattCalibrator().fit(
            np.concatenate([cache[d.doc_id]["raw_punctuation_probability"] for d in train_docs]), train_y)
        records = []
        for doc in dev_docs:
            p = tuple(float(v) for v in calibrator.predict(np.asarray(cache[doc.doc_id]["raw_punctuation_probability"])))
            gold = tuple(int(v) for v in doc.gold)
            for candidate in generate_candidates(doc.gold, doc.doc_id, config["seed"], config["rates"]):
                name = f"{candidate.condition}_{candidate.rate:g}"
                records.append(ScoredSegmentation(doc.doc_id, name, tuple(int(v) for v in candidate.boundaries), gold, p))
        items = build_items(records, {d.doc_id: d.genre for d in dev_docs})
        result = evaluate(items, 1000, int(config["seed"]))
        out = out_root / f"synthetic_{source}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        note = (f"初步研究的合成扰动候选（受控 GUM dev 文本，{len(dev_docs)} 篇 × {len(records) // len(dev_docs)} 个候选）；"
                f"证据：{source}，Platt 在 train 上拟合。GUM dev 是初步研究原有的评测集，这里只是对已有缓存的再分析，不用于任何选型。")
        (out / "summary.md").write_text(render(result, f"跨文档质量一致性：合成扰动 / {source}", note))
        print(json.dumps({"status": "SYNTHETIC_NORMALIZATION_COMPLETE", "source": source, "results": str(out)}), flush=True)


if __name__ == "__main__":
    main()
