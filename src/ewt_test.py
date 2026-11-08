"""One-shot EWT test of the frozen evidence (PREREGISTRATION amendment 4).

Stages, run once and in order by ``scripts/run_ewt_test.slurm``:

  prepare           EWT test -> surface/raw artifacts (gold-free), data manifest
  (splitters)       ``python -m src.run_real_splitters --split test --data-root data/real_system_ewt``
  evidence raw      frozen BERT features on EWT; every configuration fitted once on all of GUM train
  score raw         Soft F1 / normalized Soft F1 of every (doc, system) written to disk = frozen
  layout            paragraph-break artifact (derived from sentence-initial ``# newpar``; supplementary
                    ``para`` variant only, so it is produced after the main scores are frozen)
  evidence para / score para
  gold              EWT Gold materialized only if both frozen score files exist
  analyze           cross-document quality agreement, selection diagnostics, evidence diagnostics

The main method is ``raw/probe+lr``; the primary comparison is against ``raw/platt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_evidence_gaps import left_context_category
from .boundary_probe import CONFIGS, extract, fit_logistic, load_features, predict_logistic
from .contracts import GoldBoundaries, RawDocument, read_gold_boundaries, read_raw_documents, write_jsonl
from .layout_data import paragraph_break_offsets
from .meta_eval import evaluate as meta_evaluate, max_soft_f1
from .quality_agreement import Items, evaluate, load_items, statistics
from .real_splitters import INTRA_TOKEN_REASON
from .scoring import soft_fbeta
from .selection import ScoredSegmentation
from .surface_data import extract_gold_boundaries, load_surface_documents, load_surface_jsonl, save_jsonl


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/UD_English-EWT"
EWT_RELEASE = "r2.18"
# Overridable only for the pipeline dry run on already-seen GUM dev (never needed for the real run).
SOURCE = Path(os.environ.get("EWT_TEST_SOURCE", SOURCE_DIR / "en_ewt-ud-test.conllu"))
DATA = Path(os.environ.get("EWT_TEST_DATA", ROOT / "data/real_system_ewt"))
RESULTS = Path(os.environ.get("EWT_TEST_RESULTS", ROOT / "results/ewt_test"))
GUM = ROOT / "data/real_system"
GUM_FEATURES = ROOT / "data/lm_cache/bert_probe_v1.1"
PRIMARY = "raw/probe+lr"
BASELINE = "raw/platt"
METRICS = (("global_spearman", "全局 Spearman"), ("large_gap_pairwise", "大差距配对"),
           ("cell_spearman", "领域×系统 Spearman"), ("within_doc_pairwise", "文档内配对"))


def ewt_genre(doc_id: str) -> str:
    return doc_id.split("-", 1)[0]


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    temporary.replace(path)


def test_surfaces():
    surfaces = sorted(load_surface_jsonl(DATA / "surface_test.jsonl"), key=lambda d: d.doc_id)
    for doc in surfaces:
        doc.genre = ewt_genre(doc.doc_id)
    return surfaces


# ---------------------------------------------------------------- prepare (gold-free)
def prepare(args) -> None:
    if SOURCE.parent == SOURCE_DIR and (SOURCE_DIR / "UD_RELEASE").read_text().strip() != EWT_RELEASE:
        raise RuntimeError(f"Expected frozen EWT {EWT_RELEASE}; run scripts/fetch_ewt.sh")
    surfaces = load_surface_documents(SOURCE, "test")
    for doc in surfaces:
        doc.genre = ewt_genre(doc.doc_id)
    raw = [RawDocument.create(doc_id=d.doc_id, corpus="UD_English-EWT", split="test", genre=d.genre,
                              text=d.text, source_ref=f"UniversalDependencies/UD_English-EWT@{EWT_RELEASE}")
           for d in surfaces]
    DATA.mkdir(parents=True, exist_ok=True)
    save_jsonl(surfaces, DATA / "surface_test.jsonl")
    write_jsonl(raw, DATA / "raw_test.jsonl")
    genres = defaultdict(int)
    for d in surfaces:
        genres[d.genre] += 1
    manifest = {"source": "UniversalDependencies/UD_English-EWT", "release": EWT_RELEASE,
                "file": str(SOURCE.relative_to(ROOT)) if SOURCE.is_relative_to(ROOT) else str(SOURCE),
                "file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "documents": len(surfaces), "genres": dict(sorted(genres.items())),
                "candidate_positions": sum(len(d.candidate_offsets) for d in surfaces),
                "gold_materialized": False}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "EWT_PREPARED", **manifest}), flush=True)


def layout(args) -> None:
    for variant_file in ("raw.jsonl",):
        if not (DATA / "frozen" / variant_file).stat().st_size:
            raise RuntimeError("Main (raw) scores must be frozen before the layout artifact is built")
    surfaces = test_surfaces()
    breaks = paragraph_break_offsets(SOURCE, surfaces)
    write_rows([{"doc_id": d.doc_id, "text_sha256": d.text_sha256, "paragraph_break_offsets": breaks[d.doc_id]}
                for d in surfaces], DATA / "layout_test.jsonl")
    print(json.dumps({"status": "EWT_LAYOUT_COMPLETE", "paragraph_breaks": sum(map(len, breaks.values()))}), flush=True)


# ---------------------------------------------------------------- evidence (GPU, gold-free on EWT)
def _blocks(torch, f: dict, device: str) -> dict:
    clipped = np.clip(f["p"], 1e-7, 1 - 1e-7)
    return {"logit": torch.as_tensor((np.log(clipped) - np.log1p(-clipped))[:, None], dtype=torch.float32, device=device),
            "lr": torch.as_tensor(f["lr"], dtype=torch.float32, device=device),
            "hidden": torch.as_tensor(f["hidden"], device=device).float()}


def evidence(args) -> None:
    import torch

    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    surfaces = test_surfaces()
    raw_docs = {d.doc_id: d for d in read_raw_documents(DATA / "raw_test.jsonl")}
    for d in surfaces:
        if raw_docs[d.doc_id].sha256 != d.text_sha256:
            raise RuntimeError(f"Raw/surface checksum mismatch for {d.doc_id}")
    layout_rows = {}
    if args.variant == "para":
        layout_rows = {r["doc_id"]: r for r in jsonl(DATA / "layout_test.jsonl")}
    started = time.time()
    extraction_args = argparse.Namespace(mask_batch=args.mask_batch, lr_batch=args.lr_batch, feature_root=DATA / "features")
    test_features, timings = extract(extraction_args, torch, config, surfaces, layout_rows, None,
                                     variants=(args.variant,), feature_name="test.pt")

    # Final models: fitted once on all of GUM train (labels = GUM-train Gold, the calibration set).
    gum_surfaces = sorted(load_surface_jsonl(GUM / "surface_train.jsonl"), key=lambda d: d.doc_id)
    gum_raw = {d.doc_id: d for d in read_raw_documents(GUM / "raw_train.jsonl")}
    gold = {}
    for record in read_gold_boundaries(GUM / "gold_train.jsonl"):
        record.validate_against(gum_raw[record.doc_id])
        gold[record.doc_id] = set(record.offsets)
    y = torch.as_tensor(np.concatenate([[int(o in gold[d.doc_id]) for o in d.candidate_offsets] for d in gum_surfaces]),
                        dtype=torch.float32, device="cuda")
    train_features, _ = load_features(torch, GUM_FEATURES, gum_surfaces, variants=(args.variant,))
    train_blocks = _blocks(torch, train_features[args.variant], "cuda")
    test_blocks = _blocks(torch, test_features[args.variant], "cuda")
    fits = {}
    for name, (parts, lambdas) in CONFIGS.items():
        lam = lambdas[0]
        x_train = torch.cat([train_blocks[b] for b in parts], dim=1)
        model = fit_logistic(torch, x_train, y, lam)
        p = predict_logistic(torch, model, torch.cat([test_blocks[b] for b in parts], dim=1)).cpu().numpy().astype(float)
        fits[name] = {"lambda": lam, "train_loss": model["loss"], "max_grad": model["max_grad"]}
        (DATA / "models").mkdir(parents=True, exist_ok=True)
        torch.save({k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in model.items()},
                   DATA / "models" / f"{args.variant}_{name}.pt")
        rows, start = [], 0
        for d in surfaces:
            n = len(d.candidate_offsets)
            rows.append({"doc_id": d.doc_id, "text_sha256": d.text_sha256, "candidate_offsets": d.candidate_offsets,
                         "calibrated_boundary_probability": p[start:start + n].tolist(),
                         "variant": args.variant, "config": name, "lambda": lam})
            start += n
        write_rows(rows, DATA / "evidence" / args.variant / f"{name}.jsonl")
        del x_train
    info = {"timings": timings, "fits": fits, "seconds": round(time.time() - started, 1)}
    (DATA / f"evidence_{args.variant}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"status": "EWT_EVIDENCE_COMPLETE", "variant": args.variant, **info}), flush=True)


# ---------------------------------------------------------------- score (gold-free) = freeze
def score(args) -> None:
    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    surfaces = {d.doc_id: d for d in test_surfaces()}
    raw_docs = {d.doc_id: d for d in read_raw_documents(DATA / "raw_test.jsonl")}
    predictions = {}
    for system in config["systems"]:
        for row in jsonl(DATA / "predictions" / f"{system}.jsonl"):
            if row["text_sha256"] != raw_docs[row["doc_id"]].sha256:
                raise RuntimeError(f"Prediction checksum mismatch for {row['doc_id']}")
            predictions[(row["doc_id"], system)] = set(row["offsets"])
    rows = []
    for name in CONFIGS:
        for record in jsonl(DATA / "evidence" / args.variant / f"{name}.jsonl"):
            surface = surfaces[record["doc_id"]]
            if record["candidate_offsets"] != surface.candidate_offsets or record["text_sha256"] != surface.text_sha256:
                raise RuntimeError(f"Evidence does not match {record['doc_id']}")
            p = record["calibrated_boundary_probability"]
            maximum = max_soft_f1(tuple(p))
            for system in config["systems"]:
                chosen = predictions[(record["doc_id"], system)]
                vector = [int(o in chosen) for o in surface.candidate_offsets]
                soft = soft_fbeta(vector, p, config["soft_f1_beta"])
                rows.append({"doc_id": record["doc_id"], "text_sha256": record["text_sha256"], "system": system,
                             "variant": args.variant, "config": name,
                             "candidate_boundary_offsets": sorted(chosen),
                             "soft_f1": soft, "normalized_soft_f1": soft / maximum})
    write_rows(rows, DATA / "frozen" / f"{args.variant}.jsonl")
    print(json.dumps({"status": "EWT_SCORES_FROZEN", "variant": args.variant, "rows": len(rows),
                      "gold_file_opened": False}), flush=True)


# ---------------------------------------------------------------- gold (only after freezing)
def materialize_gold(args) -> None:
    for variant in ("raw", "para"):
        path = DATA / "frozen" / f"{variant}.jsonl"
        if not path.exists() or not path.stat().st_size:
            raise RuntimeError("Refusing to read EWT Gold before every score file is frozen")
    surfaces = test_surfaces()
    raw = {d.doc_id: d for d in read_raw_documents(DATA / "raw_test.jsonl")}
    internal = extract_gold_boundaries(SOURCE, "test", surfaces)
    contracts = [GoldBoundaries(doc_id=r.doc_id, text_sha256=raw[r.doc_id].sha256, offsets=tuple(r.boundary_offsets),
                                annotation_source=f"UD_English-EWT@{EWT_RELEASE}") for r in internal]
    for record in contracts:
        record.validate_against(raw[record.doc_id])
    write_jsonl(contracts, DATA / "gold_test.jsonl")
    manifest = json.loads((RESULTS / "data_manifest.json").read_text())
    manifest.update({"gold_materialized": True, "gold_materialized_after_frozen_scores": True,
                     "sentences": int(sum(r.sentence_count for r in internal)),
                     "gold_boundaries": int(sum(len(r.boundary_offsets) for r in internal))})
    (RESULTS / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "EWT_GOLD_MATERIALIZED_AFTER_FREEZE", "documents": len(contracts)}), flush=True)


# ---------------------------------------------------------------- analyze
def _records(evidence_path: Path, systems: list[str]) -> list[ScoredSegmentation]:
    gold = {r.doc_id: set(r.offsets) for r in read_gold_boundaries(DATA / "gold_test.jsonl")}
    chosen = defaultdict(dict)
    for row in jsonl(DATA / "frozen" / "raw.jsonl"):
        chosen[row["doc_id"]][row["system"]] = set(row["candidate_boundary_offsets"])
    non_gap = defaultdict(int)
    for failure in jsonl(DATA / "alignment_failures.jsonl"):
        if failure["reason"] != INTRA_TOKEN_REASON:
            raise RuntimeError(f"Unexpected alignment failure record: {failure}")
        non_gap[(failure["doc_id"], failure["system"])] += 1
    records = []
    for row in jsonl(evidence_path):
        d, offsets = row["doc_id"], row["candidate_offsets"]
        labels = tuple(int(o in gold[d]) for o in offsets)
        p = tuple(row["calibrated_boundary_probability"])
        for s in systems:
            records.append(ScoredSegmentation(d, s, tuple(int(o in chosen[d][s]) for o in offsets), labels, p,
                                              non_gap_fp=non_gap[(d, s)]))
    return records


def analyze(args) -> None:
    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    seed, systems = int(config["fixed_seed"]), sorted(config["systems"])
    fixed = config["baselines"]["fixed_system"]
    paths = {f"{v}/{c}": DATA / "evidence" / v / f"{c}.jsonl" for v in ("raw", "para") for c in CONFIGS}
    item_kwargs = dict(data_root=DATA, split="test", candidates=DATA / "frozen" / "raw.jsonl",
                       failures=DATA / "alignment_failures.jsonl")

    # Frozen-score integrity: the analysis recomputes exactly the frozen numbers.
    frozen = {(r["variant"], r["config"], r["doc_id"], r["system"]): r
              for v in ("raw", "para") for r in jsonl(DATA / "frozen" / f"{v}.jsonl")}
    per_name, merged = {}, {}
    for name, path in paths.items():
        items, n_docs = load_items(path, **item_kwargs)
        variant, cfg = name.split("/")
        for i in range(len(items.gold)):
            row = frozen[(variant, cfg, items.doc[i], items.system[i])]
            if abs(row["soft_f1"] - items.scores["soft_f1"][i]) > 1e-9 or \
                    abs(row["normalized_soft_f1"] - items.scores["normalized_soft_f1"][i]) > 1e-9:
                raise RuntimeError(f"Analysis disagrees with frozen scores for {name}")
        per_name[name] = items
    base = per_name[BASELINE]
    report = {"items": int(len(base.gold)), "documents": n_docs, "results": {}, "per_domain": {}, "selection": {},
              "diagnostics": {}}
    for scorer in ("normalized_soft_f1", "soft_f1"):
        merged = Items(base.doc, base.domain, base.system, base.gold, {n: it.scores[scorer] for n, it in per_name.items()})
        report["results"][scorer] = evaluate(merged, args.bootstrap, seed, BASELINE, args.workers)
        print(json.dumps({"status": "EWT_AGREEMENT", "scorer": scorer}), flush=True)
        for domain in sorted(set(base.domain.tolist())):
            idx = np.flatnonzero(base.domain == domain)
            report["per_domain"].setdefault(domain, {"documents": int(len(set(base.doc[idx].tolist()))), "items": int(len(idx))})
            stats = statistics(merged, idx)
            report["per_domain"][domain][scorer] = {n: stats[n]["global_spearman"] for n in (BASELINE, PRIMARY, "para/probe+lr")}
    report["system_gold_macro_f1"] = {s: float(base.gold[base.system == s].mean()) for s in systems}

    domain_by_doc = {d.doc_id: d.genre for d in read_raw_documents(DATA / "raw_test.jsonl")}
    for name in (BASELINE, PRIMARY, "para/probe+lr"):
        result = meta_evaluate(_records(paths[name], systems), domain_by_doc, fixed,
                               config["baselines"]["direct_internal_threshold"], n_resamples=args.bootstrap, seed=seed)
        soft = result["point"]["scorers"]["soft_f1"]
        boot = result["bootstrap"]
        report["selection"][name] = {
            "domain_mean_selected_minus_fixed": soft["domain_mean"]["selected_minus_fixed"],
            "domain_mean_selected_minus_fixed_ci": boot.get("soft_f1.domain_mean_selected_minus_fixed"),
            "gold_win_rate": result["point"]["gold_win_rate"], "gold_win_rate_ci": boot.get("gold_win_rate"),
            "within_doc_pairwise": soft["pairwise"]["accuracy"],
            "domain_mean_kendall_tau": soft["domain_mean"]["kendall_tau"],
            "domains_selecting_fixed": sum(d["selected"] == fixed for d in soft["domains"].values()),
            "domains": len(soft["domains"]), "corpus_selected": soft["corpus_macro"]["selected"],
            "corpus_oracle": soft["corpus_macro"]["oracle"],
            "direct_internal_macro_gold_f1": result["point"]["direct_internal"]["macro_gold_f1"]}
        print(json.dumps({"status": "EWT_SELECTION", "config": name}), flush=True)

    gold = {r.doc_id: set(r.offsets) for r in read_gold_boundaries(DATA / "gold_test.jsonl")}
    texts = {d.doc_id: d.text for d in read_raw_documents(DATA / "raw_test.jsonl")}
    for name, path in paths.items():
        cells = defaultdict(list)
        for row in jsonl(path):
            for o, p in zip(row["candidate_offsets"], row["calibrated_boundary_probability"]):
                terminal = left_context_category(texts[row["doc_id"]], o) == "terminal"
                cells[("gold" if o in gold[row["doc_id"]] else "non_gold") + ("_terminal" if terminal else "_no_terminal")].append(p)
        report["diagnostics"][name] = {k: {"n": len(v), "mean_p": float(np.mean(v))} for k, v in sorted(cells.items())}

    gum = json.loads((ROOT / "results/boundary_probe/report.json").read_text())["results"]["logo/normalized_soft_f1"]["point"]
    report["gum_logo_reference"] = {n: gum[n]["global_spearman"] for n in paths if n in gum}
    if args.gpu_log and args.gpu_log.exists():
        util = [float(line.split(",")[0]) for line in args.gpu_log.read_text().splitlines()
                if line.strip() and line.split(",")[0].strip().replace(".", "").isdigit()]
        busy = [u for u in util if u > 0]
        report["gpu_utilization"] = {"samples": len(util), "mean_all": float(np.mean(util)) if util else None,
                                     "mean_nonidle": float(np.mean(busy)) if busy else None}
    manifest = json.loads((RESULTS / "data_manifest.json").read_text())
    report["manifest"] = manifest
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "summary.md").write_text(render(report, args.bootstrap))
    primary = report["results"]["normalized_soft_f1"]["paired_vs_baseline"][PRIMARY]["global_spearman"]
    print(json.dumps({"status": "EWT_TEST_COMPLETE", "primary_paired_ci": primary}), flush=True)


def render(report: dict, bootstrap: int) -> str:
    m = report["manifest"]
    lines = ["# EWT test：冻结证据的一次性跨语料测试（修订 4）", "",
             f"UD English EWT {m['release']} test：{m['documents']} 篇文档、{m['gold_boundaries']} 个 Gold 边界、"
             f"{m['candidate_positions']} 个候选空隙；领域 {m['genres']}。所有配置只在 GUM train 上拟合一次；"
             "分数先冻结、后物化 Gold。主方法 `raw/probe+lr`，基线 `raw/platt`；`para` 为补充实验。"
             f"CI：领域内分层按文档重采样 {bootstrap} 次。", ""]
    for scorer in ("normalized_soft_f1", "soft_f1"):
        result = report["results"][scorer]
        lines += [f"## {scorer}" + ("（主）" if scorer == "normalized_soft_f1" else ""), "",
                  "| 配置 | " + " | ".join(label for _, label in METRICS) + " | 全局 Spearman 配对差 vs raw/platt | GUM LOGO 参照 |",
                  "|---|" + "---:|" * (len(METRICS) + 2)]
        for name, values in result["point"].items():
            diff = result["paired_vs_baseline"].get(name)
            cell = "—" if diff is None else (f"{values['global_spearman'] - result['point'][BASELINE]['global_spearman']:+.3f} "
                                             f"[{diff['global_spearman'][0]:+.3f}, {diff['global_spearman'][1]:+.3f}]")
            ref = report["gum_logo_reference"].get(name) if scorer == "normalized_soft_f1" else None
            lines.append(f"| {name} | " + " | ".join(f"{values[k]:.3f}" for k, _ in METRICS) + f" | {cell} | "
                         + (f"{ref:.3f}" if ref is not None else "—") + " |")
        lines.append("")
    lines += ["## 按领域的全局 Spearman（归一化，点估计）", "", f"| 领域 | 文档 | {BASELINE} | {PRIMARY} | para/probe+lr |", "|---|---:|---:|---:|---:|"]
    for domain, entry in report["per_domain"].items():
        v = entry["normalized_soft_f1"]
        lines.append(f"| {domain} | {entry['documents']} | {v[BASELINE]:.3f} | {v[PRIMARY]:.3f} | {v['para/probe+lr']:.3f} |")
    lines += ["", "## 选择（次要诊断，原版 Soft F1）", "",
              "| 配置 | 领域平均 所选 − 永远 SaT [CI] | Gold-win [CI] | 文档内配对 | 领域 τ | 选 SaT 的领域 | 语料级选择 / oracle | 证据自身 F1 |",
              "|---|---:|---:|---:|---:|---:|---|---:|"]
    for name, s in report["selection"].items():
        ci = s["domain_mean_selected_minus_fixed_ci"] or {}
        gw = s["gold_win_rate_ci"] or {}
        lines.append(f"| {name} | {s['domain_mean_selected_minus_fixed']:+.4f} [{ci.get('low', float('nan')):+.4f}, {ci.get('high', float('nan')):+.4f}] | "
                     f"{s['gold_win_rate']:.3f} [{gw.get('low', float('nan')):.3f}, {gw.get('high', float('nan')):.3f}] | "
                     f"{s['within_doc_pairwise']:.3f} | {s['domain_mean_kendall_tau']:.3f} | {s['domains_selecting_fixed']}/{s['domains']} | "
                     f"{s['corpus_selected']} / {s['corpus_oracle']} | {s['direct_internal_macro_gold_f1']:.3f} |")
    lines += ["", "系统 Gold Macro F1（EWT）：" + "，".join(f"{k} {v:.3f}" for k, v in report["system_gold_macro_f1"].items()), "",
              "## 证据诊断（平均 p）", "", "| 配置 | 无终止标点 Gold | 有终止标点 Gold | 有终止标点非 Gold | 无终止标点非 Gold |", "|---|---:|---:|---:|---:|"]
    for name, d in report["diagnostics"].items():
        get = lambda k: f"{d[k]['mean_p']:.3f}（{d[k]['n']}）" if k in d else "—"
        lines.append(f"| {name} | {get('gold_no_terminal')} | {get('gold_terminal')} | {get('non_gold_terminal')} | {get('non_gold_no_terminal')} |")
    if "gpu_utilization" in report:
        g = report["gpu_utilization"]
        lines += ["", f"GPU 利用率（nvidia-smi 采样 {g['samples']} 次）：全程平均 {g['mean_all']:.1f}%，非空闲平均 {g['mean_nonidle']:.1f}%。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "evidence", "score", "layout", "gold", "analyze"])
    parser.add_argument("--variant", choices=["raw", "para"], default="raw")
    parser.add_argument("--mask-batch", type=int, default=512)
    parser.add_argument("--lr-batch", type=int, default=1024)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-log", type=Path, default=None)
    args = parser.parse_args()
    {"prepare": prepare, "evidence": evidence, "score": score, "layout": layout,
     "gold": materialize_gold, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
