"""Reference-free baselines vs the frozen evidence metric (PREREGISTRATION amendment 5).

B1 single-system pseudo-reference, B2 leave-one-out consensus, B3 heuristics (punctuation-rule
pseudo-reference, sentence-length plausibility), compared with ``raw/probe+lr`` normalized Soft F1
on GUM train (LOGO evidence; development set) and EWT test (frozen evidence; baselines added after
the one-shot test, nothing is tuned). Also motivation analyses: errors shared by every system,
per-system cross-document ranking, per-domain agreement, and rank-average complementarity.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from .analyze_evidence_gaps import left_context_category
from .contracts import read_gold_boundaries, read_raw_documents
from .quality_agreement import Items, evaluate, load_items
from .surface_data import load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("punkt", "sat", "stanza", "wtp")
OURS, PLATT = "ours: probe+lr", "ours-baseline: platt"
CORPORA = {
    "gum": dict(data_root=ROOT / "data/real_system", split="train",
                ours=ROOT / "data/real_system/train_cv/boundary_probe/raw/probe+lr_lam0.001/logo.jsonl",
                platt=ROOT / "data/real_system/train_cv/boundary_probe/raw/platt_lam1e-06/logo.jsonl",
                candidates=ROOT / "data/real_system/train_cv/frozen_scores_cv.jsonl",
                failures=ROOT / "data/real_system/train_cv/alignment_failures.jsonl",
                predictions=ROOT / "data/real_system/train_cv/predictions",
                label="GUM train（LOGO 交叉拟合证据，开发集）"),
    "ewt": dict(data_root=ROOT / "data/real_system_ewt", split="test",
                ours=ROOT / "data/real_system_ewt/evidence/raw/probe+lr.jsonl",
                platt=ROOT / "data/real_system_ewt/evidence/raw/platt.jsonl",
                candidates=ROOT / "data/real_system_ewt/frozen/raw.jsonl",
                failures=ROOT / "data/real_system_ewt/alignment_failures.jsonl",
                predictions=ROOT / "data/real_system_ewt/predictions",
                label="EWT test（冻结证据；基线为测试后补加的比较）"),
}
METRICS = (("global_spearman", "全局 Spearman"), ("large_gap_pairwise", "大差距配对"),
           ("cell_spearman", "领域×系统 Spearman"), ("within_doc_pairwise", "文档内配对"))


def boundary_f1(system: set[int], reference: set[int]) -> float:
    if not system and not reference:
        return 1.0
    return 2 * len(system & reference) / (len(system) + len(reference))


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gum_median_sentence_length() -> float:
    """L0 for the length baseline: median Gold sentence length (surface tokens) on GUM train."""
    surfaces = {d.doc_id: d for d in load_surface_jsonl(ROOT / "data/real_system/surface_train.jsonl")}
    lengths = []
    for record in read_gold_boundaries(ROOT / "data/real_system/gold_train.jsonl"):
        ends = np.asarray(surfaces[record.doc_id].token_ends)
        cuts = [0] + [int(np.searchsorted(ends, o, side="right")) for o in record.offsets] + [len(ends)]
        lengths += [b - a for a, b in zip(cuts, cuts[1:]) if b > a]
    return float(np.median(lengths))


def subset(items: Items, idx: np.ndarray, names: list[str]) -> Items:
    return Items(items.doc[idx], items.domain[idx], items.system[idx], items.gold[idx],
                 {n: items.scores[n][idx] for n in names})


def corpus_scores(name: str, spec: dict, length0: float) -> tuple[Items, dict]:
    """All scorers aligned to the (doc, system) items of ``load_items`` plus gap-level data."""
    kwargs = dict(data_root=spec["data_root"], split=spec["split"], candidates=spec["candidates"],
                  failures=spec["failures"])
    ours, _ = load_items(spec["ours"], **kwargs)
    platt, _ = load_items(spec["platt"], **kwargs)
    if not ((ours.doc == platt.doc).all() and (ours.system == platt.system).all()):
        raise RuntimeError("Item order differs")
    surfaces = {d.doc_id: d for d in load_surface_jsonl(spec["data_root"] / f"surface_{spec['split']}.jsonl")}
    texts = {d.doc_id: d.text for d in read_raw_documents(spec["data_root"] / f"raw_{spec['split']}.jsonl")}
    full = defaultdict(dict)                     # every predicted boundary, incl. non-whitespace positions
    for system in SYSTEMS:
        for row in jsonl(spec["predictions"] / f"{system}.jsonl"):
            full[row["doc_id"]][system] = set(row["offsets"])
    for failure in jsonl(spec["failures"]):
        full[failure["doc_id"]][failure["system"]].add(failure["offset"])
    rule = {d: {o for o in s.candidate_offsets if left_context_category(texts[d], o) == "terminal"}
            for d, s in surfaces.items()}

    scores = {OURS: ours.scores["normalized_soft_f1"], PLATT: platt.scores["normalized_soft_f1"]}
    consensus, rule_f1, length, pseudo = [], [], [], {r: [] for r in SYSTEMS}
    for d, s in zip(ours.doc, ours.system):
        others = [full[d][o] for o in SYSTEMS if o != s]
        votes = defaultdict(int)
        for boundaries in others:
            for offset in boundaries:
                votes[offset] += 1
        consensus.append(boundary_f1(full[d][s], {o for o, v in votes.items() if v >= 2}))
        rule_f1.append(boundary_f1(full[d][s], rule[d]))
        mean_length = len(surfaces[d].tokens) / (len(full[d][s]) + 1)
        length.append(-abs(math.log(mean_length / length0)))
        for r in SYSTEMS:
            pseudo[r].append(boundary_f1(full[d][s], full[d][r]))
    scores["B2 consensus (leave-one-out)"] = np.array(consensus)
    scores["B3a punctuation-rule reference"] = np.array(rule_f1)
    scores["B3b sentence-length plausibility"] = np.array(length)
    for r in SYSTEMS:
        scores[f"B1 pseudo-reference = {r}"] = np.array(pseudo[r])
    scores["ours + consensus (rank average)"] = (rankdata(scores[OURS]) + rankdata(scores["B2 consensus (leave-one-out)"])) / 2
    items = Items(ours.doc, ours.domain, ours.system, ours.gold, scores)

    # Gap-level data for the shared-error analysis.
    gold = {r.doc_id: set(r.offsets) for r in read_gold_boundaries(spec["data_root"] / f"gold_{spec['split']}.jsonl")}
    chosen = defaultdict(dict)
    for row in jsonl(spec["candidates"]):
        chosen[row["doc_id"]][row["system"]] = set(row["candidate_boundary_offsets"])
    evidence = {n: {r["doc_id"]: r["calibrated_boundary_probability"] for r in jsonl(spec[k])}
                for n, k in ((OURS, "ours"), (PLATT, "platt"))}
    shared = {"all_miss": defaultdict(list), "all_false": defaultdict(list), "gold": defaultdict(list),
              "non_gold": defaultdict(list)}
    errors = {"shared": 0, "total": 0}
    for d, surface in surfaces.items():
        for i, o in enumerate(surface.candidate_offsets):
            splits = sum(o in chosen[d][s] for s in SYSTEMS)
            is_gold = o in gold[d]
            wrong = (len(SYSTEMS) - splits) if is_gold else splits
            errors["total"] += wrong
            bucket = None
            if is_gold and splits == 0:
                bucket = "all_miss"
            elif not is_gold and splits == len(SYSTEMS):
                bucket = "all_false"
            if bucket:
                errors["shared"] += wrong
            for n in (OURS, PLATT):
                p = evidence[n][d][i]
                shared["gold" if is_gold else "non_gold"][n].append(p)
                if bucket:
                    shared[bucket][n].append(p)
    shared_summary = {"errors": errors}
    for bucket, by_scorer in shared.items():
        shared_summary[bucket] = {n: {"n": len(v), "mean_p": float(np.mean(v)) if v else float("nan"),
                                      "share_above_half": float(np.mean(np.array(v) > 0.5)) if v else float("nan")}
                                  for n, v in by_scorer.items()}
    return items, shared_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/baselines")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    seed = int(json.loads((ROOT / "configs/real_splitters.json").read_text())["fixed_seed"])
    length0 = gum_median_sentence_length()
    report = {"length_L0_tokens": length0, "corpora": {}}
    for name, spec in CORPORA.items():
        items, shared = corpus_scores(name, spec, length0)
        entry = {"label": spec["label"], "items": int(len(items.gold)), "shared_errors": shared}
        main_names = [OURS, PLATT, "B2 consensus (leave-one-out)", "B3a punctuation-rule reference",
                      "B3b sentence-length plausibility", "ours + consensus (rank average)"] + \
                     [f"B1 pseudo-reference = {r}" for r in SYSTEMS]
        entry["all_items"] = evaluate(subset(items, np.arange(len(items.gold)), main_names), args.bootstrap, seed,
                                      OURS, args.workers)
        print(json.dumps({"status": "BASELINE_EVAL", "corpus": name, "set": "all"}), flush=True)
        entry["excluding_reference"] = {}
        for r in SYSTEMS:
            idx = np.flatnonzero(items.system != r)
            names = [OURS, PLATT, "B2 consensus (leave-one-out)", "B3a punctuation-rule reference",
                     f"B1 pseudo-reference = {r}"]
            entry["excluding_reference"][r] = evaluate(subset(items, idx, names), args.bootstrap, seed, OURS, args.workers)
            print(json.dumps({"status": "BASELINE_EVAL", "corpus": name, "set": f"without {r}"}), flush=True)
        entry["per_system"] = {}
        for s in SYSTEMS:
            idx = items.system == s
            entry["per_system"][s] = {n: float(spearmanr(items.scores[n][idx], items.gold[idx]).statistic)
                                      for n in main_names if n != f"B1 pseudo-reference = {s}"}
        entry["per_domain"] = {}
        for domain in sorted(set(items.domain.tolist())):
            idx = items.domain == domain
            entry["per_domain"][domain] = {"documents": int(len(set(items.doc[idx].tolist()))),
                                           **{n: float(spearmanr(items.scores[n][idx], items.gold[idx]).statistic)
                                              for n in (OURS, PLATT, "B2 consensus (leave-one-out)",
                                                        "B3a punctuation-rule reference", "B1 pseudo-reference = sat")}}
        entry["system_gold_macro_f1"] = {s: float(items.gold[items.system == s].mean()) for s in SYSTEMS}
        # (5) post hoc: does adding the evidence to consensus help beyond consensus alone?
        entry["rank_average_vs_consensus"] = evaluate(
            subset(items, np.arange(len(items.gold)), ["B2 consensus (leave-one-out)", "ours + consensus (rank average)", OURS]),
            args.bootstrap, seed, "B2 consensus (leave-one-out)", args.workers)
        # (6) post hoc: which system does each scorer put first, corpus-wide and per domain?
        entry["selection"] = {}
        domains = sorted(set(items.domain.tolist()))
        for n in (OURS, PLATT, "B2 consensus (leave-one-out)", "ours + consensus (rank average)",
                  "B3a punctuation-rule reference"):
            mean = {s: float(items.scores[n][items.system == s].mean()) for s in SYSTEMS}
            regrets, picks = [], {}
            for domain in domains:
                in_domain = items.domain == domain
                score = {s: items.scores[n][in_domain & (items.system == s)].mean() for s in SYSTEMS}
                gold = {s: items.gold[in_domain & (items.system == s)].mean() for s in SYSTEMS}
                pick = max(SYSTEMS, key=lambda s: score[s])
                picks[domain] = pick
                regrets.append(max(gold.values()) - gold[pick])
            entry["selection"][n] = {"corpus_ranking": sorted(SYSTEMS, key=lambda s: -mean[s]),
                                     "domain_picks": picks, "domain_mean_regret": float(np.mean(regrets)),
                                     "domains_picking_sat": sum(v == "sat" for v in picks.values())}
        gold_rank = sorted(SYSTEMS, key=lambda s: -entry["system_gold_macro_f1"][s])
        entry["gold_ranking"] = gold_rank
        report["corpora"][name] = entry
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.results / "summary.md").write_text(render(report, args.bootstrap))
    print(json.dumps({"status": "BASELINES_COMPLETE"}), flush=True)


def _fmt_diff(result: dict, name: str, metric: str = "global_spearman") -> str:
    if name == OURS:
        return "—"
    point = result["point"][name][metric] - result["point"][OURS][metric]
    low, high = result["paired_vs_baseline"][name][metric]
    return f"{point:+.3f} [{low:+.3f}, {high:+.3f}]"


def render(report: dict, bootstrap: int) -> str:
    lines = ["# 无参考基线对照（修订 5）", "",
             f"主方法 `{OURS}` = raw/probe+lr 的归一化 Soft F1。配对差 = 基线 − 主方法（领域内分层按文档重采样 {bootstrap} 次）。"
             f"句长基线 L0 = {report['length_L0_tokens']:.0f} token（GUM train Gold 中位数）。", ""]
    for name, entry in report["corpora"].items():
        result = entry["all_items"]
        lines += [f"## {entry['label']}：全部 {entry['items']} 个条目", "",
                  "| 评分 | " + " | ".join(l for _, l in METRICS) + " | 全局 Spearman 差 vs 主方法 |",
                  "|---|" + "---:|" * (len(METRICS) + 1)]
        for scorer, values in result["point"].items():
            note = "（参考系统自身条目恒为 1，偏高）" if scorer.startswith("B1") else ""
            lines.append(f"| {scorer}{note} | " + " | ".join(f"{values[m]:.3f}" for m, _ in METRICS)
                         + f" | {_fmt_diff(result, scorer)} |")
        lines += ["", "**去掉参考系统自身条目后的公平比较**（B1 只能评价参考系统以外的系统）：", "",
                  "| 参考系统 R | 条目 | 主方法 | B1（R 作参考） | B1 − 主方法 [CI] | B2 共识 | B2 − 主方法 [CI] |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r, res in entry["excluding_reference"].items():
            b1 = f"B1 pseudo-reference = {r}"
            n_items = entry["items"] - entry["items"] // len(SYSTEMS)
            lines.append(f"| {r} | {n_items} | {res['point'][OURS]['global_spearman']:.3f} | "
                         f"{res['point'][b1]['global_spearman']:.3f} | {_fmt_diff(res, b1)} | "
                         f"{res['point']['B2 consensus (leave-one-out)']['global_spearman']:.3f} | "
                         f"{_fmt_diff(res, 'B2 consensus (leave-one-out)')} |")
        sh = entry["shared_errors"]
        lines += ["", "**所有系统一致的错误**（共识类基线对这些错误完全失明）：", "",
                  f"- 空隙级系统错误中，所有系统一致犯下的占 {sh['errors']['shared'] / sh['errors']['total']:.1%}"
                  f"（{sh['errors']['shared']} / {sh['errors']['total']}）。", "",
                  "| 位置 | 数量 | 主方法平均 p | 主方法 p > 0.5 | Platt 平均 p |", "|---|---:|---:|---:|---:|"]
        for bucket, label in (("all_miss", "所有系统都漏切的 Gold 边界"), ("gold", "全部 Gold 边界（参照）"),
                              ("all_false", "所有系统都多切的非边界"), ("non_gold", "全部非边界（参照）")):
            o, p = sh[bucket][OURS], sh[bucket][PLATT]
            lines.append(f"| {label} | {o['n']} | {o['mean_p']:.3f} | {o['share_above_half']:.1%} | {p['mean_p']:.3f} |")
        lines += ["", "**每个系统内部的跨文档 Spearman**（这个系统在哪些文档上切得差）：", "",
                  "| 系统 | 主方法 | Platt | B2 共识 | B3a 规则 | B1 = SaT |", "|---|---:|---:|---:|---:|---:|"]
        for s, v in entry["per_system"].items():
            sat = v.get("B1 pseudo-reference = sat")
            lines.append(f"| {s} | {v[OURS]:.3f} | {v[PLATT]:.3f} | {v['B2 consensus (leave-one-out)']:.3f} | "
                         f"{v['B3a punctuation-rule reference']:.3f} | " + (f"{sat:.3f}" if sat is not None else "—（自身）") + " |")
        lines += ["", "**按领域的全局 Spearman**：", "", "| 领域 | 文档 | 主方法 | Platt | B2 共识 | B3a 规则 | B1 = SaT（含自身） |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for domain, v in entry["per_domain"].items():
            lines.append(f"| {domain} | {v['documents']} | {v[OURS]:.3f} | {v[PLATT]:.3f} | {v['B2 consensus (leave-one-out)']:.3f} | "
                         f"{v['B3a punctuation-rule reference']:.3f} | {v['B1 pseudo-reference = sat']:.3f} |")
        rc = entry["rank_average_vs_consensus"]
        lines += ["", "**事后分析：主方法能否补充共识**（配对差 = 评分 − 共识）：", "",
                  "| 评分 | " + " | ".join(l for _, l in METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
        base = "B2 consensus (leave-one-out)"
        for scorer in rc["point"]:
            if scorer == base:
                lines.append(f"| {scorer} | " + " | ".join(f"{rc['point'][scorer][m]:.3f}" for m, _ in METRICS) + " |")
                continue
            cells = []
            for m, _ in METRICS:
                low, high = rc["paired_vs_baseline"][scorer][m]
                cells.append(f"{rc['point'][scorer][m]:.3f}（{rc['point'][scorer][m] - rc['point'][base][m]:+.3f} [{low:+.3f}, {high:+.3f}]）")
            lines.append(f"| {scorer} | " + " | ".join(cells) + " |")
        lines += ["", f"**事后分析：各评分把哪个系统排第一**（Gold 排名：{' > '.join(entry['gold_ranking'])}）：", "",
                  "| 评分 | 语料级排名 | 领域级选 SaT 的领域数 | 领域平均 regret |", "|---|---|---:|---:|"]
        for scorer, s in entry["selection"].items():
            lines.append(f"| {scorer} | {' > '.join(s['corpus_ranking'])} | {s['domains_picking_sat']}/{len(s['domain_picks'])} | "
                         f"{s['domain_mean_regret']:.4f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
