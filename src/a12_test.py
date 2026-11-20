"""Amendment 12: one-shot tests of the final evaluators with Q_end scoring on two fresh corpora.

  prepare  CPU  CoNLL-U -> raw / surface (+ paragraph-newline version of EWT dev); no gold written
  gpu      GPU  Probe evidence; SaT-12l (lower / cased flat, lower newline); ModernBERT taggers (frozen)
  freeze   CPU  refit the frozen GUM models (integrity check against amendment 11), evidence ladder,
                item scores (Q and Q_end) for every evaluator -> frozen_scores; models + evidence saved
  gold     CPU  gold artifacts, refused unless every frozen score file exists
  analyze  CPU  pre-registered hypotheses H1-H6 and the full report -> results/a12/

Systems run through ``src.run_real_splitters`` (see scripts/run_a12_test_gpu.slurm).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .a6_combine import FINAL, STACK_LAMBDA, consensus_scores, decomposition, dev_evidence, fairness, full_boundaries, logit, \
    ngram_matrix, rule_matrix
from .a6_data import CORPORA, ROOT, SYSTEMS, Corpus, doc_slices, gold_sets, jsonl, load_gap_array, predictions, \
    safe_lower, save_gap_array, surfaces, write_evidence, write_jsonl
from .a6_signals import fit_logistic_np, predict_logistic_np
from .a8_supplement import category_codes, hard_f1, resample_plan, soft_f1
from .a9_casing import gap_probabilities
from .a9_layout import bootstrap, contrast, design, doc_agree, newline_text
from .a10_quality import B_VARIANTS, SETTINGS, load, prob
from .a11_analysis import _final_quality, doc_taus, perturbation, ranking_stats, soft_f1_end
from .contracts import GoldBoundaries, RawDocument, read_raw_documents, text_sha256, write_jsonl as write_contracts
from .ewt_test import ewt_genre
from .layout_data import paragraph_break_offsets
from .meta_eval import max_soft_f1
from .quality_agreement import load_items, tie_exact
from .surface_data import extract_gold_boundaries, load_surface_documents, load_surface_jsonl, save_jsonl

RESULTS = ROOT / "results/a12"
MODELS = ROOT / "data/lm_cache/a12/final_models.pkl"
UD = ROOT / "data/UD_candidates"
SOURCE = {"ewtdev": UD / "UD_English-EWT/en_ewt-ud-dev.conllu",
          "linesx": UD / "UD_English-LinES/en_lines-ud-traindev.conllu"}
LINES_PARTS = (("train", UD / "UD_English-LinES/en_lines-ud-train.conllu"), ("dev", UD / "UD_English-LinES/en_lines-ud-dev.conllu"))


def _corpus(name: str, folder: str) -> Corpus:
    root = ROOT / "data" / folder
    return Corpus(name, root, "test", root / "predictions", root / "alignment_failures.jsonl", root / "a12")


TEST = {"ewtdev": _corpus("ewtdev", "real_system_ewtdev"), "ewtdev-nl": _corpus("ewtdev-nl", "real_system_ewtdev_nl"),
        "linesx": _corpus("linesx", "real_system_linesx")}
SCENARIOS = {"ewtdev": {"flat": "ewtdev", "nl": "ewtdev-nl"}, "linesx": {"flat": "linesx"}}
# evaluator -> (a10 setting whose features it uses, kind)
EVALUATORS = {"Stack": ("T1", "stack"), "+casing": ("T2", "stack"), "+layout": ("T3", "stack"),
              "final (flat)": ("T2", "qb"), "final (layout)": ("T3", "qb")}
LADDER = {"flat": ("Probe", "Stack", "+casing", "final (flat)"),
          "nl": ("Probe", "Stack", "+casing", "+layout", "final (layout)")}
FINAL_OF = {"flat": "final (flat)", "nl": "final (layout)"}


def genre_of(name: str, doc_id: str) -> str:
    return ewt_genre(doc_id) if name == "ewtdev" else "lines"


# ------------------------------------------------------------------ prepare (no gold)
def combined_lines() -> Path:
    """LinES train + dev in one CoNLL-U with unique document ids (lines-train-K / lines-dev-K)."""
    out = []
    for part, path in LINES_PARTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"# newdoc id = (\S+)", line)
            out.append(f"# newdoc id = lines-{part}-{m.group(1)}" if m else line)
    target = SOURCE["linesx"]
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    return target


def prepare(args) -> None:
    combined_lines()
    manifest = {}
    for name in ("ewtdev", "linesx"):
        corpus, path = TEST[name], SOURCE[name]
        docs = load_surface_documents(path, "test")
        for d in docs:
            d.genre = genre_of(name, d.doc_id)
        corpus.data_root.mkdir(parents=True, exist_ok=True)
        raw = [RawDocument.create(doc_id=d.doc_id, corpus=path.parent.name, split="test", genre=d.genre, text=d.text,
                                  source_ref=f"UniversalDependencies/{path.parent.name}@r2.18") for d in docs]
        save_jsonl(docs, corpus.data_root / "surface_test.jsonl")
        write_contracts(raw, corpus.data_root / "raw_test.jsonl")
        manifest[name] = {"file": str(path.relative_to(ROOT)), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "documents": len(docs), "gaps": sum(len(d.candidate_offsets) for d in docs), "gold_materialized": False}
        if name == "ewtdev":                          # paragraph-newline version (input layout, not gold)
            breaks = paragraph_break_offsets(path, docs)
            nl = TEST["ewtdev-nl"]
            nl.data_root.mkdir(parents=True, exist_ok=True)
            surf_rows = jsonl(corpus.data_root / "surface_test.jsonl")
            raw_rows = jsonl(corpus.data_root / "raw_test.jsonl")
            sha = {}
            for row in raw_rows:
                row["text"] = newline_text(row["text"], breaks.get(row["doc_id"], []))
                row["sha256"] = sha[row["doc_id"]] = text_sha256(row["text"])
            for row in surf_rows:
                row["text"] = newline_text(row["text"], breaks.get(row["doc_id"], []))
                row["text_sha256"] = sha[row["doc_id"]]
            write_jsonl(raw_rows, nl.data_root / "raw_test.jsonl")
            write_jsonl(surf_rows, nl.data_root / "surface_test.jsonl")
            write_jsonl([{"doc_id": d, "text_sha256": sha[d], "paragraph_break_offsets": sorted(b)} for d, b in sorted(breaks.items())],
                        nl.data_root / "layout.jsonl")
            manifest[name]["paragraph_breaks"] = int(sum(map(len, breaks.values())))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "A12_PREPARED", **manifest}), flush=True)


# ------------------------------------------------------------------ GPU evidence (frozen models only)
def run_gpu(args) -> None:
    import torch
    from wtpsplit import SaT
    from .a6_gpu import MODEL_CACHE, Tagger, predict
    from .boundary_probe import extract, predict_logistic
    from .ewt_test import _blocks

    config = json.loads((ROOT / "configs/real_splitters.json").read_text())
    probe = torch.load(ROOT / "data/real_system_ewt/models/raw_probe+lr.pt", weights_only=False)
    probe = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in probe.items()}
    for name in ("ewtdev", "linesx"):                 # Probe (amendment 4 model), flat text
        corpus = TEST[name]
        docs = surfaces(corpus)
        ns = argparse.Namespace(mask_batch=512, lr_batch=1024, feature_root=corpus.a6_root / "features")
        features, _ = extract(ns, torch, config, docs, {}, None, variants=("raw",), feature_name="test.pt")
        blocks = _blocks(torch, features["raw"], "cuda")
        p = predict_logistic(torch, probe, torch.cat([blocks[b] for b in ("logit", "hidden", "lr")], dim=1))
        save_gap_array(corpus, "probe", docs, p.cpu().numpy().astype(float))
        print(json.dumps({"status": "A12_PROBE", "corpus": name}), flush=True)
    sat = SaT("sat-12l")
    sat.half().to("cuda")
    for cname, case in (("ewtdev", "lower"), ("ewtdev", "cased"), ("ewtdev-nl", "lower"), ("linesx", "lower"), ("linesx", "cased")):
        corpus = TEST[cname]
        docs = surfaces(corpus)
        texts = [d.text if case == "cased" else safe_lower(d.text) for d in docs]
        probs = list(sat.predict_proba(texts, stride=64, block_size=512, batch_size=64))
        out = []
        for d, pr in zip(docs, probs):
            pr = np.asarray(pr, dtype=float)
            if len(pr) != len(d.text):
                raise RuntimeError(f"SaT length mismatch in {d.doc_id}")
            out.append(gap_probabilities(pr, d.candidate_offsets))
        save_gap_array(corpus, f"sat12l-{case}", docs, np.concatenate(out))
        print(json.dumps({"status": "A12_SAT", "corpus": cname, "case": case}), flush=True)
    del sat
    torch.cuda.empty_cache()
    for model_file, targets in (("ft-modernbert_final.pt", ("ewtdev", "linesx")), ("ft-modernbert-lower-nl_final.pt", ("ewtdev-nl", "linesx"))):
        saved = torch.load(MODEL_CACHE / model_file, weights_only=False)
        tagger = Tagger(torch, saved["model_name"])
        tagger.encoder.load_state_dict({k: v.float() for k, v in saved["encoder"].items()})
        tagger.head.load_state_dict(saved["head"])
        tagger.encoder.eval()
        key = "ftmb-lower-flat" if model_file == "ft-modernbert_final.pt" else "ftmb-lower-nl"
        for cname in targets:
            docs = surfaces(TEST[cname])
            save_gap_array(TEST[cname], key, docs, predict(tagger, docs, safe_lower))
            print(json.dumps({"status": "A12_E2", "corpus": cname, "model": key}), flush=True)
        del tagger
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ freeze
def design_for(name: str, setting: str, docs, final: dict) -> np.ndarray:
    corpus = TEST[name]
    e0 = load_gap_array(corpus, "probe", docs)
    rules, ngrams = rule_matrix(docs), ngram_matrix(final["ngram"], docs)
    if setting == "T1":
        return design(e0, logit(load_gap_array(corpus, "sat12l-lower", docs)), load_gap_array(corpus, "ftmb-lower-flat", docs), rules, ngrams)
    if setting == "T2":
        return design(e0, logit(load_gap_array(corpus, "sat12l-cased", docs)), load_gap_array(corpus, "ftmb-lower-flat", docs), rules, ngrams)
    # T3: lower-nl features; LinES has no layout, so its "newline" text is the flat text (negative control)
    nl_corpus = TEST["ewtdev-nl"] if name == "ewtdev" else corpus
    nl_docs = surfaces(nl_corpus)
    if [d.candidate_offsets for d in nl_docs] != [d.candidate_offsets for d in docs]:
        raise RuntimeError("Newline and flat gaps differ")
    e1 = logit(load_gap_array(nl_corpus, "sat12l-lower", nl_docs))
    e2 = load_gap_array(nl_corpus if name == "ewtdev" else corpus, "ftmb-lower-nl", nl_docs)
    breaks = np.zeros(sum(len(d.candidate_offsets) for d in docs))
    if name == "ewtdev":
        rows = {r["doc_id"]: set(r["paragraph_break_offsets"]) for r in jsonl(nl_corpus.data_root / "layout.jsonl")}
        breaks = np.concatenate([[o in rows.get(d.doc_id, set()) for o in d.candidate_offsets] for d in docs]).astype(float)
    return design(e0, e1, e2, rules, ngrams, breaks)


def fit_models(threads: int) -> dict:
    import torch
    torch.set_num_threads(threads)
    gum_docs = surfaces(CORPORA["gum"])
    ev_gum, extras = dev_evidence(CORPORA["gum"], gum_docs, gold_sets(CORPORA["gum"]))
    with FINAL.open("rb") as handle:
        final = pickle.load(handle)
    ctx = {"ev_gum": ev_gum, "extras": extras, "final": final}
    models = {"final_components": final}
    for t in ("T1", "T2", "T3"):
        gum = load("gum", SETTINGS[t], ctx)
        models[f"stack-{t}"] = fit_logistic_np(gum.x, gum.y, STACK_LAMBDA)
        if t in ("T2", "T3"):
            models[f"qb-{t}"] = _final_quality(torch, gum, B_VARIANTS["B-qb"])
    # integrity: GENTLE evidence of amendment 11 must be reproduced exactly
    cache = np.load(ROOT / "results/a11/evidence/gentle.npz")
    check, gentle_x = {}, {}
    for ev_name, (t, kind) in EVALUATORS.items():
        if t not in gentle_x:
            gentle_x[t] = load("gentle", SETTINGS[t], ctx).x
        x = gentle_x[t]
        p = predict_logistic_np(models[f"stack-{t}"], x) if kind == "stack" else prob(*models[f"qb-{t}"], x)
        key = {"Stack": "Stack", "+casing": "+casing", "+layout": "+layout", "final (flat)": "final_flat", "final (layout)": "final_layout"}[ev_name]
        check[ev_name] = float(np.max(np.abs(p - cache[f"ev__{key}"])))
    models["integrity_max_abs_diff"] = check
    if max(check.values()) >= 1e-9:
        raise RuntimeError(f"Refit models do not reproduce amendment 11 on GENTLE: {check}")
    return models


def q_scores(rows: np.ndarray, q: np.ndarray, kind: str) -> np.ndarray:
    if kind == "Q_end":
        return soft_f1_end(rows, q)
    if not len(q):
        return np.ones(len(rows))
    if kind == "hard":
        return hard_f1(rows, q >= 0.5)
    return soft_f1(rows, q) / max_soft_f1(tuple(q))


def freeze(args) -> None:
    # Review fix (2026-11-20): never refit or re-freeze once any test Gold is on disk (cf. a6_combine.freeze).
    existing = [str(TEST[c].data_root / "gold_test.jsonl") for m in SCENARIOS.values() for c in m.values()
                if (TEST[c].data_root / "gold_test.jsonl").exists()]
    if existing:
        raise RuntimeError(f"Gold already materialized ({existing[0]}); refusing to (re)fit or (re)freeze scores")
    started = time.time()
    models = fit_models(args.threads)
    MODELS.parent.mkdir(parents=True, exist_ok=True)
    with MODELS.open("wb") as handle:
        pickle.dump(models, handle)
    print(json.dumps({"status": "A12_MODELS", "integrity": models["integrity_max_abs_diff"],
                      "seconds": round(time.time() - started, 1)}), flush=True)
    final = models["final_components"]
    for name, scen_map in SCENARIOS.items():
        corpus = TEST[name]
        docs = surfaces(corpus)
        ev = {"Probe": load_gap_array(corpus, "probe", docs)}
        for ev_name, (t, kind) in EVALUATORS.items():
            x = design_for(name, t, docs, final)
            ev[ev_name] = predict_logistic_np(models[f"stack-{t}"], x) if kind == "stack" else prob(*models[f"qb-{t}"], x)
        np.savez_compressed(corpus.path("evidence_frozen.npz"), **{k: v for k, v in ev.items()},
                            doc_ids=np.array([d.doc_id for d in docs]))
        for scen, cname in scen_map.items():
            sc_corpus = TEST[cname]
            sdocs = surfaces(sc_corpus)
            preds = predictions(sc_corpus)
            scores = {}
            for ev_name in LADDER[scen]:
                for kind in ("Q", "Q_end"):
                    scores[f"{ev_name} [{kind}]"] = []
            scores[f"H({FINAL_OF[scen]})"] = []
            for d, sl in zip(docs, doc_slices(docs)):
                rows = np.array([[o in preds[(d.doc_id, s)] for o in d.candidate_offsets] for s in SYSTEMS], dtype=bool)
                for ev_name in LADDER[scen]:
                    for kind in ("Q", "Q_end"):
                        scores[f"{ev_name} [{kind}]"].extend(q_scores(rows, ev[ev_name][sl], kind))
                scores[f"H({FINAL_OF[scen]})"].extend(q_scores(rows, ev[FINAL_OF[scen]][sl], "hard"))
            scores = {k: np.array(v) for k, v in scores.items()}
            scores["M0"] = consensus_scores(sdocs, full_boundaries(sc_corpus, preds))
            scores["DocAgree"] = doc_agree(sdocs, preds)
            out_rows = []
            for k, d in enumerate(d for d in docs for _ in SYSTEMS):
                s = SYSTEMS[k % len(SYSTEMS)]
                out_rows.append({"doc_id": d.doc_id, "system": s, "candidate_boundary_offsets": sorted(preds[(d.doc_id, s)]),
                                 "scores": {n: float(v[k]) for n, v in scores.items()}})
            write_jsonl(out_rows, sc_corpus.path("frozen_scores.jsonl"))
            print(json.dumps({"status": "A12_FROZEN", "corpus": cname, "items": len(out_rows), "gold_file_opened": False}), flush=True)


# ------------------------------------------------------------------ gold (after freeze)
def gold(args) -> None:
    for name, scen_map in SCENARIOS.items():
        for cname in scen_map.values():
            f = TEST[cname].path("frozen_scores.jsonl")
            if not f.exists() or not f.stat().st_size:
                raise RuntimeError(f"Refusing to read test Gold before the scores of {cname} are frozen")
    for name, scen_map in SCENARIOS.items():
        corpus = TEST[name]
        docs = load_surface_jsonl(corpus.data_root / "surface_test.jsonl")
        internal = extract_gold_boundaries(SOURCE[name], "test", docs, allow_non_gap=True)
        for cname in scen_map.values():
            raw = {d.doc_id: d for d in read_raw_documents(TEST[cname].data_root / "raw_test.jsonl")}
            contracts = [GoldBoundaries(doc_id=r.doc_id, text_sha256=raw[r.doc_id].sha256, offsets=tuple(r.boundary_offsets),
                                        annotation_source=f"{SOURCE[name].parent.name}@r2.18") for r in internal]
            for record in contracts:
                record.validate_against(raw[record.doc_id])
            write_contracts(contracts, TEST[cname].data_root / "gold_test.jsonl")
        print(json.dumps({"status": "A12_GOLD_AFTER_FREEZE", "corpus": name, "documents": len(internal)}), flush=True)


# ------------------------------------------------------------------ analysis
def items_for(cname: str, docs):
    corpus = TEST[cname]
    preds = predictions(corpus)
    cand = corpus.path("candidates.jsonl")
    write_jsonl([{"doc_id": d.doc_id, "system": s, "candidate_boundary_offsets": sorted(preds[(d.doc_id, s)])}
                 for d in docs for s in SYSTEMS], cand)
    path = write_evidence(corpus, "items_placeholder", docs, np.full(sum(len(d.candidate_offsets) for d in docs), 0.5))
    items, _ = load_items(path, data_root=corpus.data_root, split="test", candidates=cand, failures=corpus.failures)
    if list(zip(items.doc.tolist(), items.system.tolist())) != [(d.doc_id, s) for d in docs for s in SYSTEMS]:
        raise RuntimeError(f"Item order mismatch in {cname}")
    return items


def ranking(scores: dict, f1: np.ndarray, genre: np.ndarray, n_boot: int) -> tuple[dict, dict]:
    n = len(genre)
    plan = resample_plan(genre, n_boot)
    out, draws = {}, {}
    for name, v in scores.items():
        point = ranking_stats(v, f1, genre, np.arange(n))
        dt = doc_taus(v, f1, n)
        dr = np.array([ranking_stats(v, f1, genre, idx) for idx in plan])
        draws[name] = np.column_stack([dr[:, 1], [np.nanmean(dt[idx]) for idx in plan]])
        q = lambda col: [float(np.nanpercentile(draws[name][:, col], 2.5)), float(np.nanpercentile(draws[name][:, col], 97.5))]
        out[name] = {"corpus_tau": float(point[0]), "corpus_top1": float(point[2]), "genre_tau": [float(point[1]), *q(0)],
                     "doc_tau": [float(np.nanmean(dt)), *q(1)]}
    return out, draws


def analyze(args) -> None:
    report = {"note": "amendment 12; one-shot tests (T)", "manifest": json.loads((RESULTS / "manifest.json").read_text()),
              "integrity": None, "corpora": {}, "hypotheses": {}}
    with MODELS.open("rb") as handle:
        report["integrity"] = pickle.load(handle)["integrity_max_abs_diff"]
    for name, scen_map in SCENARIOS.items():
        corpus = TEST[name]
        docs = surfaces(corpus)
        genres = np.array([d.genre for d in docs])
        gold = gold_sets(corpus)
        out = {"documents": len(docs), "documents_without_internal_gold_boundary":
               int(sum(1 for d in docs if not (gold[d.doc_id] & set(d.candidate_offsets)))), "scenarios": {}}
        for scen, cname in scen_map.items():
            items = items_for(cname, surfaces(TEST[cname]))
            frozen = jsonl(TEST[cname].path("frozen_scores.jsonl"))
            scores = {n: tie_exact(np.array([r["scores"][n] for r in frozen])) for n in frozen[0]["scores"]}
            point, draws, names = bootstrap(scores, items.gold, genres, args.bootstrap)
            table = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, n in enumerate(names):
                    table[n] = {"global_spearman": [float(point[n][0]), float(np.percentile(draws[:, i, 0], 2.5)),
                                                    float(np.percentile(draws[:, i, 0], 97.5))],
                                "within_doc_pairwise": float(point[n][1]), **decomposition(items, scores[n]),
                                "rank_bias": fairness(items, scores[n])["rank_bias"]}
            fin = FINAL_OF[scen]
            contrasts = {f"{fin} [Q_end] - M0": contrast(draws, names, point, [f"{fin} [Q_end]"], ["M0"]),
                         f"{fin} [Q_end] - Probe [Q_end]": contrast(draws, names, point, [f"{fin} [Q_end]"], ["Probe [Q_end]"]),
                         f"{fin} [Q_end] - {fin} [Q]": contrast(draws, names, point, [f"{fin} [Q_end]"], [f"{fin} [Q]"]),
                         f"{fin} [Q_end] - H({fin})": contrast(draws, names, point, [f"{fin} [Q_end]"], [f"H({fin})"]),
                         f"{fin} [Q_end] - Stack [Q_end]": contrast(draws, names, point, [f"{fin} [Q_end]"], ["Stack [Q_end]"])}
            rank_scores = {k: scores[k] for k in (f"{fin} [Q_end]", "Probe [Q_end]", "Stack [Q_end]", "M0")}
            rank, rdraws = ranking(rank_scores, items.gold, genres, args.bootstrap)
            d = rdraws[f"{fin} [Q_end]"] - rdraws["M0"]
            rank_contrast = {"genre_tau": [rank[f"{fin} [Q_end]"]["genre_tau"][0] - rank["M0"]["genre_tau"][0],
                                           float(np.nanpercentile(d[:, 0], 2.5)), float(np.nanpercentile(d[:, 0], 97.5))],
                             "doc_tau": [rank[f"{fin} [Q_end]"]["doc_tau"][0] - rank["M0"]["doc_tau"][0],
                                         float(np.nanpercentile(d[:, 1], 2.5)), float(np.nanpercentile(d[:, 1], 97.5))]}
            out["scenarios"][scen] = {"evaluator": table, "contrasts": contrasts, "ranking": rank, "ranking_final_minus_M0": rank_contrast,
                                      "system_macro_f1": {s: float(items.gold[items.system == s].mean()) for s in SYSTEMS}}
        # perturbation test v2 on the frozen evidence (amendment 11 generator)
        ev = np.load(corpus.path("evidence_frozen.npz"))
        y = np.concatenate([[o in gold[d.doc_id] for o in d.candidate_offsets] for d in docs]).astype(bool)
        par = np.zeros(len(y), bool)
        if name == "ewtdev":
            rows = {r["doc_id"]: set(r["paragraph_break_offsets"]) for r in jsonl(TEST["ewtdev-nl"].data_root / "layout.jsonl")}
            par = np.concatenate([[o in rows.get(d.doc_id, set()) for o in d.candidate_offsets] for d in docs]).astype(bool)
        sl = doc_slices(docs)
        scen_for_cons = "nl" if "nl" in scen_map else "flat"
        cpreds = predictions(TEST[scen_map[scen_for_cons]])
        cmat = np.concatenate([np.array([[o in cpreds[(d.doc_id, s)] for o in d.candidate_offsets] for s in SYSTEMS], dtype=bool)
                               for d in docs], axis=1)
        keymap = {"Probe": "Probe", "Stack": "Stack", "+casing": "+casing", "+layout": "+layout",
                  "final_flat": "final (flat)", "final_layout": "final (layout)"}
        c = SimpleNamespace(name=name, ev={k: ev[v] for k, v in keymap.items()}, y=y, par=par,
                            cats=np.concatenate([category_codes(d.text, d.candidate_offsets) for d in docs]),
                            doc=np.array([d.doc_id for d in docs]), genre=genres, sl=sl, scen={scen_for_cons: {"cmat": cmat}})
        out["perturbation"] = perturbation(c, args.bootstrap)
        report["corpora"][name] = out
        print(json.dumps({"status": "A12_ANALYZED", "corpus": name}), flush=True)
    e = report["corpora"]["ewtdev"]["scenarios"]
    li = report["corpora"]["linesx"]["scenarios"]["flat"]
    ok = lambda v, sign=1: (v[1] > 0) if sign > 0 else (v[2] < 0)
    hyp = {
        "H1 final(layout) Q_end - M0 > 0 (EWT dev, layout)": e["nl"]["contrasts"]["final (layout) [Q_end] - M0"],
        "H2a final(layout) Q_end - Probe Q_end > 0 (EWT dev, layout)": e["nl"]["contrasts"]["final (layout) [Q_end] - Probe [Q_end]"],
        "H2b final(flat) Q_end - Probe Q_end > 0 (EWT dev, flat)": e["flat"]["contrasts"]["final (flat) [Q_end] - Probe [Q_end]"],
        "H3 final(layout) Q_end - Q > 0 (EWT dev, layout)": e["nl"]["contrasts"]["final (layout) [Q_end] - final (layout) [Q]"],
        "H4 genre-mean tau final(layout) - M0 > 0 (EWT dev, layout)": e["nl"]["ranking_final_minus_M0"]["genre_tau"],
        "H5 final(flat) Q_end - Probe Q_end (LinES train+dev)": li["contrasts"]["final (flat) [Q_end] - Probe [Q_end]"],
    }
    report["hypotheses"] = {k: {"estimate": v, "confirmed": bool(ok(v))} for k, v in hyp.items()}
    # H6: layout evaluator on text without layout (LinES), paired vs the flat final evaluator
    lines_items = items_for("linesx", surfaces(TEST["linesx"]))
    frozen = jsonl(TEST["linesx"].path("frozen_scores.jsonl"))
    ev = np.load(TEST["linesx"].path("evidence_frozen.npz"))
    docs = surfaces(TEST["linesx"])
    preds = predictions(TEST["linesx"])
    lay = []
    for d, sl in zip(docs, doc_slices(docs)):
        rows = np.array([[o in preds[(d.doc_id, s)] for o in d.candidate_offsets] for s in SYSTEMS], dtype=bool)
        lay.extend(q_scores(rows, ev["final (layout)"][sl], "Q_end"))
    sc = {"final (layout) [Q_end]": tie_exact(np.array(lay)),
          "final (flat) [Q_end]": tie_exact(np.array([r["scores"]["final (flat) [Q_end]"] for r in frozen]))}
    point, draws, names = bootstrap(sc, lines_items.gold, np.array([d.genre for d in docs]), args.bootstrap)
    h6 = contrast(draws, names, point, ["final (layout) [Q_end]"], ["final (flat) [Q_end]"])
    report["hypotheses"]["H6 final(layout) - final(flat) < 0 (LinES, no layout)"] = {"estimate": h6, "confirmed": bool(h6[2] < 0)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "summary.md").write_text(render(report))
    print(json.dumps({"status": "A12_COMPLETE", "hypotheses": {k: v["confirmed"] for k, v in report["hypotheses"].items()}}), flush=True)


def _ci(v, sign=False) -> str:
    f = (lambda x: f"{x:+.3f}") if sign else (lambda x: f"{x:.3f}")
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"


def render(r: dict) -> str:
    L = ["# 修订 12：一次性测试（EWT dev，LinES train + dev；T）", "",
         "完整性核对（重新拟合的模型 vs 修订 11 在 GENTLE 上的证据，最大 |Δp|）：" + json.dumps(r["integrity"]), "",
         "## 预先写定的假设", "", "| 假设 | 估计 [95% CI] | 成立 |", "|---|---:|:---:|"]
    for k, v in r["hypotheses"].items():
        L.append(f"| {k} | {_ci(v['estimate'], True)} | {'✓' if v['confirmed'] else '✗'} |")
    for name, c in r["corpora"].items():
        L += ["", f"# {name}（{c['documents']} 篇；没有内部 Gold 边界的文档 {c['documents_without_internal_gold_boundary']} 篇）"]
        for scen, s in c["scenarios"].items():
            L += ["", f"## 场景：{'带版式' if scen == 'nl' else '无换行'}", "",
                  "系统 macro F1：" + "，".join(f"{k} {v:.3f}" for k, v in s["system_macro_f1"].items()), "",
                  "| 评分 | 全局 Spearman [CI] | 文档之间 | 文档之内 | 文档内配对 | " + " | ".join(f"残差 {x}" for x in SYSTEMS) + " |",
                  "|---|---:|---:|---:|---:|" + "---:|" * len(SYSTEMS)]
            for n, v in s["evaluator"].items():
                L.append(f"| {n} | {_ci(v['global_spearman'])} | {v['between_doc']:.3f} | {v['within_doc']:.3f} | {v['within_doc_pairwise']:.3f} | "
                         + " | ".join(f"{v['rank_bias'][x]:+.3f}" for x in SYSTEMS) + " |")
            L += ["", "配对差："] + [f"- {k}：{_ci(v, True)}" for k, v in s["contrasts"].items()]
            L += ["", "系统排名一致性（Kendall τ_b）：", "", "| 评分 | 语料 τ | 领域平均 τ | 文档平均 τ |", "|---|---:|---:|---:|"]
            for n, v in s["ranking"].items():
                L.append(f"| {n} | {v['corpus_tau']:.3f} | {_ci(v['genre_tau'])} | {_ci(v['doc_tau'])} |")
        p = c["perturbation"]
        k1 = p["levels"].index("k=1")
        L += ["", "## 人工扰动 v2（只注入 1 个错误的检出率）", "", "| 评分 | " + " | ".join(p["types"]) + " |", "|---|" + "---:|" * len(p["types"])]
        for i, n in enumerate(p["scorers"]):
            L.append(f"| {n} | " + " | ".join(f"{p['detect']['point'][i][t][k1]:.2f}" for t in range(len(p["types"]))) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "gpu", "freeze", "gold", "analyze"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    {"prepare": prepare, "gpu": run_gpu, "freeze": freeze, "gold": gold, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
