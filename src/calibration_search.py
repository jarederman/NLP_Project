"""P2: many small calibration experiments combining BERT evidence with gold-free NLP features.

For every configuration (feature groups x model) and every cross-fitting scheme
(the P1 5-fold split, and leave-one-genre-out), a calibrator is fitted only on
the training folds' Gold, applied to the held-out fold, and the four real
systems are scored with Soft F1. All gold-free scores are written to disk before
meta-evaluation (Stage A -> Stage B), as in the rest of the pipeline.

Selection follows results/calibration_search/PREREGISTRATION.md, which was
written before any of these experiments ran. GUM dev and test are never read.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from .bert_surface import PROTOCOLS
from .contracts import read_gold_boundaries, read_raw_documents
from .gap_features import CATEGORIES, GROUPS, document_features, lexical_ids, vote_features
from .meta_eval import doc_stats, point_report, record_stats
from .real_splitters import INTRA_TOKEN_REASON
from .scoring import soft_fbeta
from .selection import ScoredSegmentation


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "platt"


@dataclass(frozen=True)
class Config:
    name: str
    groups: tuple[str, ...]   # "Bz" = BERT logit only (Platt); "B" = logit + mask mode
    model: str = "lr"
    vote: bool = False        # uses candidate votes: comparison only, never eligible

    @property
    def eligible(self) -> bool:
        return not self.vote

    @property
    def size(self) -> int:
        """Number of feature groups beyond the BERT evidence (for the simplicity rule)."""
        return len([g for g in self.groups if g not in ("Bz", "B")]) + (1 if self.model != "lr" else 0)


CONFIGS = (
    Config("platt", ("Bz",)),
    Config("B", ("B",)),
    Config("B+C", ("B", "C")),
    Config("B+K", ("B", "K")),
    Config("B+A", ("B", "A")),
    Config("B+L", ("B", "L")),
    Config("B+C+K", ("B", "C", "K")),
    Config("B+C+K+A", ("B", "C", "K", "A")),
    Config("B+C+K+L", ("B", "C", "K", "L")),
    Config("B+C+K+A+L", ("B", "C", "K", "A", "L")),
    Config("B+K+A+L", ("B", "K", "A", "L")),
    Config("B+C+A+L", ("B", "C", "A", "L")),
    Config("B+C+K+A+L/hgb", ("B", "C", "K", "A", "L"), "hgb"),
    Config("C+K+A+L（无 LM）", ("C", "K", "A", "L")),
    Config("C+K+A+L/hgb（无 LM）", ("C", "K", "A", "L"), "hgb"),
    # Wave 2 (pre-registration amendment 1): orthographic statistics and lexical priors.
    Config("B+C+K+O", ("B", "C", "K", "O")),
    Config("B+C+K+W", ("B", "C", "K", "W")),
    Config("B+C+K+A+L+O", ("B", "C", "K", "A", "L", "O")),
    Config("B+C+K+A+L+O+W", ("B", "C", "K", "A", "L", "O", "W")),
    Config("B+C+K+A+L+O/hgb", ("B", "C", "K", "A", "L", "O"), "hgb"),
    Config("B+V（投票）", ("B",), vote=True),
    Config("B+C+K+A+L+V（投票）", ("B", "C", "K", "A", "L"), vote=True),
)


def make_model(kind: str, seed: int, n_lexical: int = 0):
    """``n_lexical`` trailing columns are categorical word ids, target-encoded with
    sklearn's internal cross-fitting so training rows never see their own label."""
    if kind == "lr" and n_lexical:
        def encoder(n_columns):
            dense = list(range(n_columns - n_lexical))
            lexical = list(range(n_columns - n_lexical, n_columns))
            return ColumnTransformer([
                ("dense", "passthrough", dense),
                ("lexical", TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=seed), lexical),
            ])
        return _LazyColumns(encoder, seed)
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    raise ValueError(kind)


class _LazyColumns:
    """Pipeline whose ColumnTransformer is built once the column count is known."""

    def __init__(self, encoder_factory, seed: int):
        self.encoder_factory, self.seed = encoder_factory, seed

    def fit(self, X, y):
        self.pipeline = make_pipeline(self.encoder_factory(X.shape[1]), StandardScaler(),
                                      LogisticRegression(C=1.0, max_iter=5000))
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


def columns_for(config: Config, slices: dict[str, slice], vote_column: int) -> np.ndarray:
    cols: list[int] = []
    for group in config.groups:
        if group == "Bz":
            cols.append(slices["B"].start)
        elif group != "W":
            cols.extend(range(slices[group].start, slices[group].stop))
    if "W" in config.groups:          # categorical columns go last (see make_model)
        cols.extend(range(slices["W"].start, slices["W"].stop))
    if config.vote:
        cols.append(vote_column)
    return np.array(cols)


def fit_predict(config: Config, X: np.ndarray, y: np.ndarray, row_fold: np.ndarray,
                cols: np.ndarray, seed: int) -> np.ndarray:
    """Cross-fitted probabilities: each fold predicted by a model that never saw it."""
    out = np.full(len(y), np.nan)
    for fold in np.unique(row_fold):
        train, test = row_fold != fold, row_fold == fold
        model = make_model(config.model, seed, 2 if "W" in config.groups else 0)
        model.fit(X[train][:, cols], y[train])
        out[test] = model.predict_proba(X[test][:, cols])[:, 1]
    if np.isnan(out).any():
        raise RuntimeError("Every gap must receive a held-out prediction")
    return out


def selection_arrays(Q: np.ndarray, G: np.ndarray, domain_ix: np.ndarray, fixed: int,
                     n_domains: int) -> float:
    """Domain-mean (selected - fixed) macro Gold F1; ties broken by system name order."""
    total = 0.0
    for d in range(n_domains):
        rows = domain_ix == d
        selected = int(np.argmax(Q[rows].mean(axis=0)))
        total += G[rows, selected].mean() - G[rows, fixed].mean()
    return total / n_domains


def paired_bootstrap(results: dict[str, dict], G: np.ndarray, domain_ix: np.ndarray, fixed: int,
                     n_resamples: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Same stratified document resamples for every configuration (so differences are paired)."""
    rng = np.random.default_rng(seed)
    n_domains = int(domain_ix.max()) + 1
    pools = [np.flatnonzero(domain_ix == d) for d in range(n_domains)]
    draws = {name: {"primary": np.empty(n_resamples), "gold_win": np.empty(n_resamples)} for name in results}
    for b in range(n_resamples):
        idx = np.concatenate([rng.choice(pool, size=len(pool), replace=True) for pool in pools])
        for name, entry in results.items():
            Q, gs = entry["Q"][idx], entry["gold_soft"][idx]
            draws[name]["primary"][b] = selection_arrays(Q, G[idx], domain_ix[idx], fixed, n_domains)
            draws[name]["gold_win"][b] = float(np.mean(gs >= Q.max(axis=1)))
    return draws


def ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def choose(rows: dict[str, dict], draws: dict, configs: dict[str, Config]) -> dict:
    """Pre-registered rule: best LOGO primary among eligible configs, then the simplest
    configuration that is not distinguishable from it (gap < 0.002 or paired CI spans 0)."""
    eligible = [n for n in rows if configs[n].eligible]
    best = max(eligible, key=lambda n: (rows[n]["primary"], rows[n]["gold_win"]))
    candidates = []
    for name in eligible:
        diff = draws[best]["primary"] - draws[name]["primary"]
        low, high = ci(diff)
        gap = rows[best]["primary"] - rows[name]["primary"]
        if name == best or gap < 0.002 or low <= 0.0 <= high:
            candidates.append(name)
    chosen = min(candidates, key=lambda n: (configs[n].size, -rows[n]["primary"], -rows[n]["gold_win"]))
    return {"best_primary": best, "indistinguishable_from_best": sorted(candidates), "chosen": chosen}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_splitters.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--cv-root", type=Path, default=ROOT / "data/real_system/train_cv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/real_system/train_cv/search")
    parser.add_argument("--results", type=Path, default=ROOT / "results/calibration_search")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--only", nargs="*", help="Run only these configuration names (debugging)")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    seed, beta = int(config["fixed_seed"]), float(config["soft_f1_beta"])
    systems = sorted(config["systems"])
    fixed_name = config["baselines"]["fixed_system"]
    fixed = systems.index(fixed_name)
    threshold = config["baselines"]["direct_internal_threshold"]
    protocol = config["bert"]["mask_protocol"]
    configs = {c.name: c for c in CONFIGS if not args.only or c.name in args.only}
    if BASELINE not in configs:
        configs = {BASELINE: next(c for c in CONFIGS if c.name == BASELINE), **configs}

    def jsonl(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    documents = {doc.doc_id: doc for doc in read_raw_documents(args.data_root / "raw_train.jsonl")}
    doc_ids = sorted(documents)
    cache = {r["doc_id"]: r for r in jsonl(ROOT / "data/lm_cache" / f"bert_surface_{PROTOCOLS[protocol]}" / "train.jsonl")}
    candidates: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
    for row in jsonl(args.cv_root / "frozen_scores_cv.jsonl"):
        offsets = cache[row["doc_id"]]["candidate_offsets"]
        chosen = set(row["candidate_boundary_offsets"])
        candidates[row["doc_id"]][row["system"]] = tuple(int(o in chosen) for o in offsets)
    non_gap: dict[tuple[str, str], int] = defaultdict(int)
    for failure in jsonl(args.cv_root / "alignment_failures.jsonl"):
        if failure["reason"] != INTRA_TOKEN_REASON:
            raise RuntimeError(f"Unexpected alignment failure record: {failure}")
        non_gap[(failure["doc_id"], failure["system"])] += 1
    kfold = json.loads((args.cv_root / "folds.json").read_text())["folds"]

    # ---- Gold-free feature matrix ------------------------------------------------
    blocks, vote_blocks, row_doc, slices = [], [], [], {}
    vocabulary: dict[str, int] = {}   # string -> integer id only; carries no label information
    for doc_id in doc_ids:
        record, document = cache[doc_id], documents[doc_id]
        if record["text_sha256"] != document.sha256 or record["protocol"] != protocol:
            raise RuntimeError(f"Cache mismatch for {doc_id}")
        if set(candidates[doc_id]) != set(systems):
            raise RuntimeError(f"Missing frozen candidates for {doc_id}")
        groups = document_features(document.text, record["candidate_offsets"],
                                   record["raw_punctuation_probability"], record["mask_modes"])
        left_words, right_words = lexical_ids(document.text, record["candidate_offsets"])
        groups["W"] = np.column_stack([[vocabulary.setdefault(w, len(vocabulary)) for w in left_words],
                                       [vocabulary.setdefault(w, len(vocabulary)) for w in right_words]])
        if not slices:
            start = 0
            for name in (*GROUPS, "W"):
                slices[name] = slice(start, start + groups[name].shape[1])
                start += groups[name].shape[1]
        blocks.append(np.hstack([groups[name] for name in (*GROUPS, "W")]).astype(float))
        vote_blocks.append(vote_features(record["candidate_offsets"], candidates[doc_id]))
        row_doc.extend([doc_id] * len(record["candidate_offsets"]))
    X = np.hstack([np.vstack(blocks), np.vstack(vote_blocks)])
    vote_column = X.shape[1] - 1
    row_doc = np.array(row_doc)
    doc_rows = {d: np.flatnonzero(row_doc == d) for d in doc_ids}
    categories_terminal = X[:, slices["C"].start: slices["C"].start + len(CATEGORIES) - 1].sum(axis=1) == 0

    # Gold labels: used only to FIT calibrators on training folds (and, in Stage B, to evaluate).
    gold_offsets = {}
    for record in read_gold_boundaries(args.data_root / "gold_train.jsonl"):
        record.validate_against(documents[record.doc_id])
        gold_offsets[record.doc_id] = set(record.offsets)
    y = np.concatenate([[int(o in gold_offsets[d]) for o in cache[d]["candidate_offsets"]] for d in doc_ids])

    domains = sorted({documents[d].genre for d in doc_ids})
    schemes = {
        "kfold5": np.array([kfold[d] for d in row_doc]),
        "logo": np.array([domains.index(documents[d].genre) for d in row_doc]),
    }

    # ---- Stage A: cross-fitted evidence and frozen gold-free scores ---------------
    tasks = [(scheme, name) for scheme in schemes for name in configs]
    predictions = Parallel(n_jobs=args.jobs)(
        delayed(fit_predict)(configs[name], X, y, schemes[scheme], columns_for(configs[name], slices, vote_column), seed)
        for scheme, name in tasks
    )
    evidence = dict(zip(tasks, predictions))
    for (scheme, name), probabilities in evidence.items():
        rows = []
        for doc_id in doc_ids:
            p = probabilities[doc_rows[doc_id]]
            for system in systems:
                rows.append({"doc_id": doc_id, "system": system, "scheme": scheme, "config": name,
                             "soft_f1": soft_fbeta(list(candidates[doc_id][system]), p, beta)})
        path = args.output_root / scheme / f"{name.replace('/', '_')}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        np.save(path.with_suffix(".npy"), probabilities)   # gap-level evidence, row order = gap_rows.json
    (args.output_root / "gap_rows.json").write_text(json.dumps(
        {"doc_ids": doc_ids, "gaps_per_doc": [len(doc_rows[d]) for d in doc_ids]}) + "\n")
    print(json.dumps({"status": "CALIBRATION_SEARCH_SCORES_FROZEN", "configs": len(configs),
                      "schemes": list(schemes)}), flush=True)

    # ---- Stage B: meta-evaluation -------------------------------------------------
    domain_ix = np.array([domains.index(documents[d].genre) for d in doc_ids])
    G = np.zeros((len(doc_ids), len(systems)))
    report: dict = {"protocol": protocol, "fixed_system": fixed_name, "n_documents": len(doc_ids),
                    "n_domains": len(domains), "schemes": {}}
    for scheme in schemes:
        per_config: dict[str, dict] = {}
        rows_out: dict[str, dict] = {}
        for name, cfg in configs.items():
            p_all = evidence[(scheme, name)]
            stats, docs_meta = [], []
            Q = np.zeros((len(doc_ids), len(systems)))
            gold_soft = np.zeros(len(doc_ids))
            for i, doc_id in enumerate(doc_ids):
                p = tuple(float(v) for v in p_all[doc_rows[doc_id]])
                gold = tuple(int(v) for v in y[doc_rows[doc_id]])
                records = [ScoredSegmentation(doc_id, s, candidates[doc_id][s], gold, p,
                                              non_gap_fp=non_gap[(doc_id, s)]) for s in systems]
                for j, record in enumerate(records):
                    Q[i, j] = record.quality
                    G[i, j] = record.gold_counts.f1
                stats.extend(record_stats(r, documents[doc_id].genre) for r in records)
                meta = doc_stats(records[0], documents[doc_id].genre, threshold)
                docs_meta.append(meta)
                gold_soft[i] = meta.gold_soft_f1
            point = point_report(stats, docs_meta, fixed_name)
            soft = point["scorers"]["soft_f1"]
            primary = soft["domain_mean"]["selected_minus_fixed"]
            if abs(primary - selection_arrays(Q, G, domain_ix, fixed, len(domains))) > 1e-12:
                raise RuntimeError("Vectorised selection disagrees with meta_eval")
            non_terminal = ~categories_terminal
            per_config[name] = {"Q": Q, "gold_soft": gold_soft}
            rows_out[name] = {
                "groups": list(cfg.groups) + (["V"] if cfg.vote else []),
                "model": cfg.model,
                "eligible": cfg.eligible,
                "primary": primary,
                "gold_win": point["gold_win_rate"],
                "pairwise": soft["pairwise"]["accuracy"],
                "kendall_tau": soft["domain_mean"]["kendall_tau"],
                "regret": soft["domain_mean"]["regret"],
                "domains_selecting_fixed": sum(d["selected"] == fixed_name for d in soft["domains"].values()),
                "selected_by_domain": {k: v["selected"] for k, v in soft["domains"].items()},
                "direct_internal_macro_f1": point["direct_internal"]["macro_gold_f1"],
                "gap_log_loss": float(log_loss(y, np.clip(p_all, 1e-7, 1 - 1e-7))),
                "non_terminal_auc": float(roc_auc_score(y[non_terminal], p_all[non_terminal])),
                "non_terminal_gold_mean_p": float(p_all[non_terminal & (y == 1)].mean()),
                "terminal_non_gold_mean_p": float(p_all[categories_terminal & (y == 0)].mean()),
            }
        draws = paired_bootstrap(per_config, G, domain_ix, fixed, args.bootstrap, seed)
        for name, row in rows_out.items():
            row["primary_ci"] = ci(draws[name]["primary"])
            row["gold_win_ci"] = ci(draws[name]["gold_win"])
            row["primary_minus_platt_ci"] = ci(draws[name]["primary"] - draws[BASELINE]["primary"])
        report["schemes"][scheme] = {"configs": rows_out, "decision": choose(rows_out, draws, configs)}

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.results / "summary.md").write_text(render(report))
    print(json.dumps({"status": "CALIBRATION_SEARCH_COMPLETE",
                      "logo_choice": report["schemes"]["logo"]["decision"]["chosen"]}, ensure_ascii=False), flush=True)


def render(report: dict) -> str:
    fixed = report["fixed_system"]
    lines = [
        "# P2 证据组合实验（GUM train）", "",
        "选型规则见 `PREREGISTRATION.md`（实验前写定）。主指标：领域平均（所选 − Always-"
        f"{fixed}）Macro Gold F1；以 LOGO（留一领域）为准。CI 为按领域分层、按文档的配对 bootstrap。"
        "“投票”配置使用候选系统输出，只作对照。", "",
    ]
    titles = {"logo": "留一领域（LOGO，选型依据）", "kfold5": "5 折（领域内，参考）"}
    for scheme in ("logo", "kfold5"):
        entry = report["schemes"][scheme]
        decision = entry["decision"]
        lines += [f"## {titles[scheme]}", "",
                  f"主指标最高：**{decision['best_primary']}**；与之不可区分：{', '.join(decision['indistinguishable_from_best'])}；"
                  f"按规则选中：**{decision['chosen']}**。", "",
                  f"| 配置 | 主指标 [95% CI] | 相对 Platt 的差 [配对 CI] | Gold-win [CI] | Pairwise | 领域平均 τ | "
                  f"选 {fixed} 的领域 | 证据 DI F1 | gap log-loss | 无标点 AUC | 无标点 Gold 平均 p | 带标点非 Gold 平均 p |",
                  "|" + "---|" * 12]
        ordered = sorted(entry["configs"].items(), key=lambda kv: -kv[1]["primary"])
        for name, r in ordered:
            lines.append(
                f"| {name} | {r['primary']:+.4f} [{r['primary_ci'][0]:+.4f}, {r['primary_ci'][1]:+.4f}] | "
                f"[{r['primary_minus_platt_ci'][0]:+.4f}, {r['primary_minus_platt_ci'][1]:+.4f}] | "
                f"{r['gold_win']:.3f} [{r['gold_win_ci'][0]:.3f}, {r['gold_win_ci'][1]:.3f}] | "
                f"{r['pairwise']:.3f} | {r['kendall_tau']:.3f} | {r['domains_selecting_fixed']}/{report['n_domains']} | "
                f"{r['direct_internal_macro_f1']:.3f} | {r['gap_log_loss']:.4f} | {r['non_terminal_auc']:.3f} | "
                f"{r['non_terminal_gold_mean_p']:.3f} | {r['terminal_non_gold_mean_p']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
