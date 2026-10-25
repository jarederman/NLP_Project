"""Regenerate all formal tables/reports from cached Qwen probabilities (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

from .data import load_documents
from .evaluate import aggregate, score_dev_documents, select_examples, write_jsonl, write_metrics
from .lm import read_cache
from .postprocess import diagnostics_and_cases
from .report import write_summary
from .scoring import PlattCalibrator


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads((ROOT / "configs/preliminary.json").read_text())
    docs = load_documents(ROOT / "data/processed/gum_dev_documents.jsonl")
    cache = read_cache(ROOT / "data/lm_cache/qwen_punctuation.jsonl")
    cal_data = json.loads((ROOT / "results/calibration.json").read_text())
    calibrator = PlattCalibrator(cal_data["slope"], cal_data["intercept"])
    rows, aux = score_dev_documents(docs, cache, calibrator, config)
    agg = aggregate(rows)
    examples = select_examples(docs, rows, aux, config)
    metadata = json.loads((ROOT / "results/run_metadata.json").read_text())
    diagnostics, cases = diagnostics_and_cases(docs, cache, calibrator, rows, config)
    metadata.update({"diagnostics": diagnostics, "cases": cases})
    rule_totals = metadata["rule_totals"]
    write_metrics(rows, ROOT / "results/metrics.csv")
    write_jsonl(examples, ROOT / "results/examples.jsonl")
    (ROOT / "results/aggregate.json").write_text(json.dumps(agg, indent=2, allow_nan=True) + "\n")
    (ROOT / "results/run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_summary(ROOT / "results/summary.md", metadata, agg, rule_totals, calibrator.to_dict())
    print(json.dumps({"status": "REANALYZED_FROM_CACHE", "rows": len(rows), "examples": len(examples)}))


if __name__ == "__main__":
    main()
