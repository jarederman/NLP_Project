"""Amendments 9.2 + 9.3: 2 x 2 factorial (evidence case x evidence layout) for the evaluator, with
the evaluated systems run on flat text and on text with paragraph newlines (two scenarios).

  prepare   CPU: newline versions of GUM train / GENTLE / LinES / EWT (the space at every UD
            ``# newpar`` break becomes ``\\n``; same length, same gaps, new SHA256)
  gpu       GPU: sat-12l on newline text x {cased, lower}; ModernBERT E2 for the three new cells
            (LOGO on GUM + one final model, inference on the other corpora)
  analyze   CPU: stacks per cell, evaluator agreement per scenario, main effects / interaction,
            perturbation slopes, reproduction checks, pre-registered decision

The systems on newline text are produced by ``src.run_real_splitters`` (see the Slurm script).
GUM = development (LOGO); GENTLE / LinES / EWT are re-used tests (R). LinES has no paragraph
marks, so its newline text equals the original: a negative control.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .a6_combine import E0_DEV, FINAL, PLATT_LAMBDA, STACK_LAMBDA, consensus_scores, decomposition, dev_evidence, \
    e0_evidence, fairness, full_boundaries, gold_items, logit, ngram_matrix, raw_signal, rule_matrix, \
    soft_scores, stack_features
from .a6_data import CORPORA, ROOT, SEED, SYSTEMS, Corpus, doc_slices, gold_sets, jsonl, load_gap_array, \
    logo_folds, predictions, safe_lower, save_gap_array, surfaces, write_evidence, write_jsonl
from .a6_signals import fit_logistic_np, predict_logistic_np
from .a8_supplement import resample_plan
from .a9_casing import A9, CANDIDATES, corpus_items, gap_probabilities, logo_stack, macro_f1, slopes, \
    SLOPE_TYPES, thresholded
from .contracts import text_sha256
from .layout_data import paragraph_break_offsets
from .quality_agreement import Items, load_items
from .surface_data import load_surface_jsonl


RESULTS = ROOT / "results/a9"
NAMES = ("gum", "gentle", "lines", "ewt")
CONLLU = {"gum": ROOT / "data/UD_English-GUM/en_gum-ud-train.conllu",
          "gentle": ROOT / "data/UD_English-GENTLE/en_gentle-ud-test.conllu",
          "lines": ROOT / "data/UD_English-LinES/en_lines-ud-test.conllu",
          "ewt": ROOT / "data/UD_English-EWT/en_ewt-ud-test.conllu"}
NL_ROOT = {"gum": ROOT / "data/real_system_nl", "gentle": ROOT / "data/real_system_gentle_nl",
           "lines": ROOT / "data/real_system_lines_nl", "ewt": ROOT / "data/real_system_ewt_nl"}
NL = {n: Corpus(f"{n}-nl", NL_ROOT[n], A9[n].split, NL_ROOT[n] / "predictions",
                NL_ROOT[n] / "alignment_failures.jsonl", NL_ROOT[n] / "a9") for n in NAMES}
CELLS = (("lower", "flat"), ("cased", "flat"), ("lower", "nl"), ("cased", "nl"))     # simplest first
E2_MODEL = "answerdotai/ModernBERT-large"
EWT_E0 = ROOT / "data/real_system_ewt/evidence/raw/probe+lr.jsonl"
SCENARIOS = ("flat", "nl")
PRIMARY = "nl"
TOL = 0.01
LABEL = {"gum": "GUM train（LOGO，开发）", "gentle": "GENTLE（R）", "lines": "LinES（R，阴性对照）", "ewt": "EWT test（R）"}


def cell_name(case: str, layout: str) -> str:
    return f"{case}-{layout}"


def transform_for(case: str):
    return (lambda t: t) if case == "cased" else safe_lower


def source(layout: str) -> dict:
    return A9 if layout == "flat" else NL


# ------------------------------------------------------------------ prepare (CPU)
def newline_text(text: str, breaks) -> str:
    chars = list(text)
    for b in breaks:
        if chars[b] != " ":
            raise ValueError(f"Paragraph break at {b} is {chars[b]!r}, not a single space")
        chars[b] = "\n"
    return "".join(chars)


def prepare(args) -> None:
    manifest = {}
    for name in args.corpora:
        flat = A9[name]
        split = flat.split
        surf = load_surface_jsonl(flat.data_root / f"surface_{split}.jsonl")
        breaks = paragraph_break_offsets(CONLLU[name], surf)
        raw_rows = jsonl(flat.data_root / f"raw_{split}.jsonl")
        surf_rows = jsonl(flat.data_root / f"surface_{split}.jsonl")
        gold_rows = jsonl(flat.data_root / f"gold_{split}.jsonl")
        new_sha = {}
        for row in raw_rows:
            row["text"] = newline_text(row["text"], breaks[row["doc_id"]])
            row["sha256"] = new_sha[row["doc_id"]] = text_sha256(row["text"])
        for row in surf_rows:
            row["text"] = newline_text(row["text"], breaks[row["doc_id"]])
            if text_sha256(row["text"]) != new_sha[row["doc_id"]]:
                raise RuntimeError(f"Raw/surface newline text differs for {row['doc_id']}")
            row["text_sha256"] = new_sha[row["doc_id"]]
        for row in gold_rows:
            row["text_sha256"] = new_sha[row["doc_id"]]
        root = NL_ROOT[name]
        write_jsonl(raw_rows, root / f"raw_{split}.jsonl")
        write_jsonl(surf_rows, root / f"surface_{split}.jsonl")
        write_jsonl(gold_rows, root / f"gold_{split}.jsonl")
        write_jsonl([{"doc_id": d, "text_sha256": new_sha[d], "paragraph_break_offsets": sorted(b)}
                     for d, b in sorted(breaks.items())], root / "layout.jsonl")
        manifest[name] = {"documents": len(raw_rows), "paragraph_breaks": sum(map(len, breaks.values())),
                          "unchanged_documents": sum(1 for b in breaks.values() if not b)}
        print(json.dumps({"status": "A9_NL_PREPARED", "corpus": name, **manifest[name]}), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "layout_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def break_mask(name: str, docs) -> np.ndarray:
    rows = {r["doc_id"]: set(r["paragraph_break_offsets"]) for r in jsonl(NL_ROOT[name] / "layout.jsonl")}
    return np.concatenate([[o in rows[d.doc_id] for o in d.candidate_offsets] for d in docs]).astype(float)


# ------------------------------------------------------------------ GPU
def run_gpu(args) -> None:
    import torch
    from wtpsplit import SaT
    from .a6_gpu import MODEL_CACHE, Tagger, predict, train

    sat = SaT("sat-12l")
    sat.half().to("cuda")
    for name in NAMES:
        docs = surfaces(NL[name])
        for case in ("cased", "lower"):
            f = transform_for(case)
            probs = list(sat.predict_proba([f(d.text) for d in docs], stride=64, block_size=512, batch_size=64))
            out = []
            for d, prob in zip(docs, probs):
                prob = np.asarray(prob, dtype=float)
                if len(prob) != len(d.text):
                    raise RuntimeError(f"SaT length mismatch in {d.doc_id}")
                out.append(gap_probabilities(prob, d.candidate_offsets))
            save_gap_array(NL[name], f"sat12l-{case}", docs, np.concatenate(out))
            print(json.dumps({"status": "A9_NL_SAT", "corpus": name, "case": case}), flush=True)
    del sat
    torch.cuda.empty_cache()

    # E2 (lower, flat) on EWT with the amendment-6 final model (it never ran on EWT)
    saved = torch.load(MODEL_CACHE / "ft-modernbert_final.pt", weights_only=False)
    tagger = Tagger(torch, saved["model_name"])
    tagger.encoder.load_state_dict({k: v.float() for k, v in saved["encoder"].items()})
    tagger.head.load_state_dict(saved["head"])
    tagger.encoder.eval()
    docs = surfaces(A9["ewt"])
    save_gap_array(A9["ewt"], "ftmb-lower", docs, predict(tagger, docs, safe_lower))
    del tagger
    torch.cuda.empty_cache()

    for case, layout in CELLS[1:]:
        started = time.time()
        corp = source(layout)
        f = transform_for(case)
        docs = surfaces(corp["gum"])
        gold = gold_sets(corp["gum"])
        fold = logo_folds(docs)
        genres = sorted({d.genre for d in docs})
        out = np.zeros(len(fold))
        for g, genre in enumerate(genres):
            tagger = train(torch, E2_MODEL, [d for d in docs if d.genre != genre], gold, SEED + g, f)
            out[fold == g] = predict(tagger, [d for d in docs if d.genre == genre], f)
            del tagger
            torch.cuda.empty_cache()
        save_gap_array(corp["gum"], f"ftmb-{case}", docs, out)
        tagger = train(torch, E2_MODEL, docs, gold, SEED, f)
        torch.save({"encoder": {k: v.to(torch.bfloat16) for k, v in tagger.encoder.state_dict().items()},
                    "head": tagger.head.state_dict(), "model_name": E2_MODEL},
                   MODEL_CACHE / f"ft-modernbert-{cell_name(case, layout)}_final.pt")
        for name in NAMES[1:]:
            d_t = surfaces(corp[name])
            save_gap_array(corp[name], f"ftmb-{case}", d_t, predict(tagger, d_t, f))
        del tagger
        torch.cuda.empty_cache()
        print(json.dumps({"status": "A9_E2_CELL", "cell": cell_name(case, layout),
                          "seconds": round(time.time() - started, 1)}), flush=True)


# ------------------------------------------------------------------ evidence assembly
def e1_logit(name: str, case: str, layout: str, docs) -> np.ndarray:
    if (case, layout) == ("lower", "flat") and name != "ewt":
        return raw_signal(CORPORA[name], docs, "E1")                          # the amendment-6 array
    return logit(load_gap_array(source(layout)[name], f"sat12l-{case}", docs))


def e2_logit(name: str, case: str, layout: str, docs) -> np.ndarray:
    if (case, layout) == ("lower", "flat") and name != "ewt":
        return raw_signal(CORPORA[name], docs, "E2")
    return load_gap_array(source(layout)[name], f"ftmb-{case}", docs)


def platt_z(gum_z: np.ndarray, y: np.ndarray, fold, target_z) -> np.ndarray:
    if target_z is None:
        out = np.zeros_like(gum_z)
        for g in sorted(set(fold.tolist())):
            held = fold == g
            out[held] = predict_logistic_np(fit_logistic_np(gum_z[~held, None], y[~held], PLATT_LAMBDA), gum_z[held, None])
        return out
    return predict_logistic_np(fit_logistic_np(gum_z[:, None], y, PLATT_LAMBDA), target_z[:, None])


def design(e0, e1z, e2z, rules, ngrams, nl=None) -> np.ndarray:
    x = stack_features(e0, e1z, e2z, rules, ngrams)
    return x if nl is None else np.column_stack([x, nl])


# ------------------------------------------------------------------ fast bootstrap statistics
def fast_stats(scores: np.ndarray, gold: np.ndarray, n_sys: int = len(SYSTEMS)) -> np.ndarray:
    """(k, 2): global Spearman and within-document pairwise accuracy for k scorers. Items are
    grouped document-major in blocks of ``n_sys`` (as in every amendment-6 artifact)."""
    rg = rankdata(gold)
    out = np.zeros((len(scores), 2))
    g = gold.reshape(-1, n_sys)
    iu = np.triu_indices(n_sys, 1)
    dg = (g[:, :, None] - g[:, None, :])[:, iu[0], iu[1]]
    valid = dg != 0
    for k, s in enumerate(scores):
        rs = rankdata(s)
        out[k, 0] = np.corrcoef(rs, rg)[0, 1]
        sm = s.reshape(-1, n_sys)
        ds = (sm[:, :, None] - sm[:, None, :])[:, iu[0], iu[1]]
        credit = np.where(ds == 0, 0.5, (np.sign(ds) == np.sign(dg)).astype(float))
        out[k, 1] = credit[valid].mean() if valid.any() else np.nan
    return out


def bootstrap(scores: dict, gold: np.ndarray, genres: np.ndarray, n_boot: int) -> tuple[dict, np.ndarray, list]:
    names = list(scores)
    mat = np.array([scores[n] for n in names])
    n_sys = len(SYSTEMS)
    point = fast_stats(mat, gold)
    draws = []
    for idx in resample_plan(genres, n_boot):
        rows = (idx[:, None] * n_sys + np.arange(n_sys)).ravel()
        draws.append(fast_stats(mat[:, rows], gold[rows]))
    return dict(zip(names, point)), np.array(draws), names


def contrast(draws: np.ndarray, names: list, point: dict, plus, minus, metric: int = 0) -> list[float]:
    """Point and 95% CI of sum(plus) - sum(minus) for one metric (0 = global Spearman)."""
    ip = [names.index(n) for n in plus]
    im = [names.index(n) for n in minus]
    d = draws[:, ip, metric].sum(1) - draws[:, im, metric].sum(1)
    p = sum(point[n][metric] for n in plus) - sum(point[n][metric] for n in minus)
    return [float(p), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ------------------------------------------------------------------ items per scenario
def scenario_items(name: str, scenario: str, docs):
    if scenario == "flat":
        return corpus_items(name, docs)
    corpus = NL[name]
    preds = predictions(corpus)
    cand = corpus.path("candidates.jsonl")
    write_jsonl([{"doc_id": d.doc_id, "system": s, "candidate_boundary_offsets": sorted(preds[(d.doc_id, s)])}
                 for d in docs for s in SYSTEMS], cand)
    path = write_evidence(corpus, "items_placeholder", docs, np.full(sum(len(d.candidate_offsets) for d in docs), 0.5))
    items, _ = load_items(path, data_root=corpus.data_root, split=corpus.split, candidates=cand, failures=corpus.failures)
    if list(zip(items.doc.tolist(), items.system.tolist())) != [(d.doc_id, s) for d in docs for s in SYSTEMS]:
        raise RuntimeError(f"Item order mismatch in {name}/nl")
    return items


def doc_agree(docs, preds) -> np.ndarray:
    out = []
    for d in docs:
        sets = [preds[(d.doc_id, s)] & set(d.candidate_offsets) for s in SYSTEMS]
        vals = [1.0 if not a and not b else 2 * len(a & b) / (len(a) + len(b))
                for i, a in enumerate(sets) for b in sets[i + 1:]]
        out.extend([float(np.mean(vals))] * len(SYSTEMS))
    return np.array(out)


# ------------------------------------------------------------------ analysis
def analyze(args) -> None:
    gum = CORPORA["gum"]
    gum_docs = surfaces(gum)
    gum_gold = gold_sets(gum)
    ev_gum, extras = dev_evidence(gum, gum_docs, gum_gold)
    y, fold = extras["y"], logo_folds(gum_docs)
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    gum_nl = break_mask("gum", gum_docs)
    x_gum = {}
    for case, layout in CELLS:
        x_gum[(case, layout)] = design(ev_gum["E0"], e1_logit("gum", case, layout, gum_docs),
                                       e2_logit("gum", case, layout, gum_docs), extras["rules"], extras["ngrams"],
                                       gum_nl if layout == "nl" else None)
    x_gum["ref91"] = design(ev_gum["E0"], e1_logit("gum", "cased", "flat", gum_docs), e2_logit("gum", "lower", "flat", gum_docs),
                            extras["rules"], extras["ngrams"])
    report = {"note": "amendments 9.2+9.3; GUM = development (LOGO); other corpora re-used (R)",
              "manifest": json.loads((RESULTS / "layout_manifest.json").read_text()), "corpora": {}}
    for name in args.corpora:
        docs = surfaces(A9[name])
        gold = gold_sets(A9[name])
        genres = np.array([d.genre for d in docs])
        nl = break_mask(name, docs)
        if name == "gum":
            e0, rules, ngrams = ev_gum["E0"], extras["rules"], extras["ngrams"]
        else:
            e0 = (e0_evidence(EWT_E0, docs) if name == "ewt" else
                  np.concatenate([q["calibrated_boundary_probability"] for q in jsonl(CORPORA[name].path("evidence", "E0.jsonl"))]))
            rules, ngrams = rule_matrix(docs), ngram_matrix(final["ngram"], docs)
        ev, checks = {}, {}
        variants = list(CELLS) + ["ref91"]
        for v in variants:
            case, layout = ("cased", "flat") if v == "ref91" else v
            e1z = e1_logit(name, case, layout, docs)
            e2z = e2_logit(name, "lower" if v == "ref91" else case, layout, docs)
            label = "ref91" if v == "ref91" else cell_name(case, layout)
            if name == "gum":
                ev[f"stack:{label}"] = logo_stack(x_gum[v], y, fold)
                if v != "ref91":
                    ev[f"E1:{label}"] = platt_z(e1z, y, fold, None)
                    ev[f"E2:{label}"] = platt_z(e2z, y, fold, None)
            else:
                x = design(e0, e1z, e2z, rules, ngrams, nl if layout == "nl" else None)
                ev[f"stack:{label}"] = predict_logistic_np(fit_logistic_np(x_gum[v], y, STACK_LAMBDA), x)
                if v != "ref91":
                    ev[f"E1:{label}"] = platt_z(e1_logit("gum", case, layout, gum_docs), y, None, e1z)
                    ev[f"E2:{label}"] = platt_z(e2_logit("gum", case, layout, gum_docs), y, None, e2z)
        if name in ("gum", "gentle", "lines"):
            frozen = ev_gum["stack"] if name == "gum" else np.concatenate(
                [q["calibrated_boundary_probability"] for q in jsonl(CORPORA[name].path("evidence", "stack.jsonl"))])
            checks["lower-flat_stack_reproduces_a6"] = float(np.max(np.abs(ev["stack:lower-flat"] - frozen)))
        if name == "lines":     # negative control: same text, so same SaT input and same system outputs
            checks["lines_E1_nl_minus_flat_max"] = float(max(
                np.max(np.abs(e1_logit(name, c, "nl", docs) - e1_logit(name, c, "flat", docs))) for c in ("lower", "cased")))
            flat_p, nl_p = predictions(A9[name]), predictions(NL[name])
            checks["lines_system_outputs_identical"] = all(flat_p[k] == nl_p[k] for k in flat_p)
        out = {"documents": len(docs), "checks": checks, "scenarios": {}}
        for scenario in SCENARIOS:
            corpus = A9[name] if scenario == "flat" else NL[name]
            sdocs = docs if scenario == "flat" else surfaces(NL[name])
            preds = predictions(corpus)
            items = scenario_items(name, scenario, sdocs)
            scores = {k: soft_scores(sdocs, preds, p)[1] for k, p in ev.items()}
            scores["M0"] = consensus_scores(sdocs, full_boundaries(corpus, preds))
            scores["DocAgree"] = doc_agree(sdocs, preds)
            point, draws, names = bootstrap(scores, items.gold, genres, args.bootstrap)
            ci = {n: [float(np.percentile(draws[:, i, 0], 2.5)), float(np.percentile(draws[:, i, 0], 97.5))]
                  for i, n in enumerate(names)}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                table = {n: {"global_spearman": [float(point[n][0]), *ci[n]], "within_doc_pairwise": float(point[n][1]),
                             **decomposition(items, scores[n]), "rank_bias": fairness(items, scores[n])["rank_bias"]}
                         for n in names}
            effects = {}
            for kind in ("stack", "E1", "E2"):
                c = lambda case, layout: f"{kind}:{cell_name(case, layout)}"
                for m, mname in ((0, "global_spearman"), (1, "within_doc_pairwise")):
                    effects[f"{kind}|{mname}"] = {
                        "case @flat": contrast(draws, names, point, [c("cased", "flat")], [c("lower", "flat")], m),
                        "case @nl": contrast(draws, names, point, [c("cased", "nl")], [c("lower", "nl")], m),
                        "layout @lower": contrast(draws, names, point, [c("lower", "nl")], [c("lower", "flat")], m),
                        "layout @cased": contrast(draws, names, point, [c("cased", "nl")], [c("cased", "flat")], m),
                        "interaction": contrast(draws, names, point, [c("cased", "nl"), c("lower", "flat")],
                                                [c("lower", "nl"), c("cased", "flat")], m)}
            vs_ref = {cell_name(*v): contrast(draws, names, point, [f"stack:{cell_name(*v)}"], ["stack:ref91"])
                      for v in CELLS}
            vs_m0 = {cell_name(*v): contrast(draws, names, point, [f"stack:{cell_name(*v)}"], ["M0"]) for v in CELLS}
            sys_f1 = {s: float(macro_f1(sdocs, [preds[(d.doc_id, s)] & set(d.candidate_offsets) for d in sdocs],
                                        gold).mean()) for s in SYSTEMS}
            out["scenarios"][scenario] = {"evaluator": table, "effects": effects, "stack_minus_ref91": vs_ref,
                                          "stack_minus_M0": vs_m0, "system_macro_f1": sys_f1}
        out["evidence_as_segmenter"] = {k: float(macro_f1(docs, thresholded(docs, p, 0.5), gold).mean())
                                        for k, p in ev.items()}
        if name != "ewt" or args.ewt_slopes:
            sl = slopes(docs, gold, {k: p for k, p in ev.items() if k.startswith("stack:")})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["slopes"] = {k: {t: float(np.nanmean(v[:, j])) for j, t in enumerate(SLOPE_TYPES)} for k, v in sl.items()}
        report["corpora"][name] = out
        print(json.dumps({"status": "A9_LAYOUT_CORPUS_DONE", "corpus": name, "checks": checks}), flush=True)
    report["decision"] = decide(report["corpora"].get("gum"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "layout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "layout_summary.md").write_text(render(report))
    print(json.dumps({"status": "A9_LAYOUT_COMPLETE", "decision": report["decision"]}), flush=True)


def decide(gum: dict | None) -> dict:
    """Pre-registered rule on GUM LOGO, primary scenario (systems see newlines)."""
    if not gum:
        return {}
    sc = gum["scenarios"][PRIMARY]
    g = {cell_name(*v): sc["evaluator"][f"stack:{cell_name(*v)}"]["global_spearman"][0] for v in CELLS}
    best = max(g.values())
    pick = next(cell_name(*v) for v in CELLS if g[cell_name(*v)] >= best - TOL)
    ci = sc["stack_minus_ref91"][pick]
    return {"global_spearman": g, "best_within_tolerance": pick, "pick_minus_ref91": ci,
            "adopted": pick if ci[1] > 0 else "ref91 (unchanged)"}


# ------------------------------------------------------------------ rendering
def _ci(v, sign=True) -> str:
    f = (lambda x: f"{x:+.3f}") if sign else (lambda x: f"{x:.3f}")
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"


def render(r: dict) -> str:
    L = ["# 修订 9.2 + 9.3：大小写 × 版式（2×2，控制变量）", "",
         "格 = 证据输入的（大小写, 版式）；场景 = 被评系统是否也看到换行。GUM = LOGO（开发）；GENTLE / LinES / EWT 为复用的测试集（R）。"
         "ref91 = 修订 9.1 采用的 Stack-cased（E1 cased、E2 lower、flat）。CI：领域内按文档重采样 1000 次。", "",
         "段落换行：" + "，".join(f"{k} {v['paragraph_breaks']} 处（{v['documents']} 篇，其中 {v['unchanged_documents']} 篇无换行）"
                             for k, v in r["manifest"].items()), ""]
    d = r.get("decision")
    if d:
        L += [f"**预先写定的判定（GUM，主场景 = 系统看到换行）**：四格 stack 全局 Spearman "
              + "，".join(f"{k} {v:.3f}" for k, v in d["global_spearman"].items())
              + f"；容差内最简单的格 {d['best_within_tolerance']}，相对 ref91 {_ci(d['pick_minus_ref91'])} → 采用 **{d['adopted']}**", ""]
    for name, c in r["corpora"].items():
        L += [f"# {LABEL[name]}（{c['documents']} 篇）", "", "核对：" + (json.dumps(c["checks"]) if c["checks"] else "—"), ""]
        for scenario, s in c["scenarios"].items():
            L += [f"## 场景：系统{'看到换行' if scenario == 'nl' else '在无换行文本上'}", "",
                  "系统 macro F1：" + "，".join(f"{k} {v:.3f}" for k, v in s["system_macro_f1"].items()), "",
                  "| 评分 | 全局 Spearman [CI] | 文档之间 | 文档之内 | 文档内配对 | " + " | ".join(f"残差 {x}" for x in SYSTEMS) + " |",
                  "|---|---:|---:|---:|---:|" + "---:|" * len(SYSTEMS)]
            for k, v in s["evaluator"].items():
                L.append(f"| {k} | {_ci(v['global_spearman'], False)} | {v['between_doc']:.3f} | {v['within_doc']:.3f} | "
                         f"{v['within_doc_pairwise']:.3f} | " + " | ".join(f"{v['rank_bias'][x]:+.3f}" for x in SYSTEMS) + " |")
            L += ["", "主效应与交互作用（配对差 [CI]）：", "", "| 证据 · 指标 | 大小写 @flat | 大小写 @nl | 版式 @lower | 版式 @cased | 交互 |",
                  "|---|---:|---:|---:|---:|---:|"]
            for k, v in s["effects"].items():
                L.append(f"| {k} | " + " | ".join(_ci(v[e]) for e in ("case @flat", "case @nl", "layout @lower", "layout @cased", "interaction")) + " |")
            L += ["", "| stack 格 | − ref91 [CI] | − M0 [CI] |", "|---|---:|---:|"]
            for k in s["stack_minus_ref91"]:
                L.append(f"| {k} | {_ci(s['stack_minus_ref91'][k])} | {_ci(s['stack_minus_M0'][k])} |")
            L.append("")
        if "slopes" in c:
            L += ["## 扰动斜率（与场景无关；理想 = 1）", "", "| stack 格 | " + " | ".join(SLOPE_TYPES) + " |", "|---|" + "---:|" * len(SLOPE_TYPES)]
            for k, v in c["slopes"].items():
                L.append(f"| {k} | " + " | ".join(f"{v[t]:.3f}" for t in SLOPE_TYPES) + " |")
            L.append("")
        L += ["## 证据当切分器（p ≥ 0.5，macro F1，诊断）", "", "| 证据 | macro F1 |", "|---|---:|"]
        L += [f"| {k} | {v:.3f} |" for k, v in c["evidence_as_segmenter"].items()] + [""]
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "gpu", "analyze"])
    parser.add_argument("--corpora", nargs="*", default=list(NAMES))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--ewt-slopes", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "gpu": run_gpu, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
