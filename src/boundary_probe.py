"""Boundary evidence beyond "does a terminal mark fit here?" (PREREGISTRATION amendment 3).

One GPU job, GUM train only (GUM dev/test are never read):

- ``raw`` and ``para`` input variants. ``raw`` is the frozen v1.1 protocol (original text as
  context, the queried gap's terminal mark replaced by / [MASK] inserted at the gap). ``para``
  is the controlled layout experiment: identical, plus a ``[SEP]`` token at every paragraph
  break (``src.layout_data``).
- per gap, one [MASK] query: P(.?!) and the layer-8/12 hidden states at [MASK] (method 1);
- per gap, the insertion likelihood ratio (method 2): the window with a terminal mark at the
  gap (A: the original mark, else an inserted ".") vs without (B: the mark removed, else
  unchanged); each of the 2 left / 3 right tokens is masked in turn, Δ = log p_A − log p_B.
- an L2 logistic regression per configuration, cross-fitted on the P1 5 folds and on
  leave-one-genre-out folds, fitted on the GPU; writes per-gap evidence for
  ``src.boundary_probe_eval``.

All windows are gathered on the GPU from one flat token array, so the GPU is never waiting on
Python-side sequence construction.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_evidence_gaps import left_context_category
from .bert_surface import PROTOCOLS, load_surface_bert, plan_queries, read_cache
from .contracts import read_gold_boundaries, read_raw_documents
from .surface_data import load_surface_jsonl


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("raw", "para")
LEFT_TARGETS = 2
RIGHT_TARGETS = 3
PROBE_LAYERS = (8, 12)
PRIMARY_LAMBDA = 1e-3
SENSITIVITY_LAMBDAS = (1e-4, 1e-2)
CONFIGS = {                       # name -> (feature blocks, lambdas)
    "platt": (("logit",), (1e-6,)),
    "lr": (("logit", "lr"), (1e-6,)),
    "probe": (("logit", "hidden"), (PRIMARY_LAMBDA, *SENSITIVITY_LAMBDAS)),
    "probe+lr": (("logit", "hidden", "lr"), (PRIMARY_LAMBDA, *SENSITIVITY_LAMBDAS)),
}


def window_budgets(max_length: int) -> tuple[int, int]:
    left = (max_length - 3) // 2
    return left, max_length - 3 - left


def right_token_indices(tokenizer, doc) -> list[int]:
    """Index of the first token right of each candidate gap (the gap is one whitespace char)."""
    encoded = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True)
    start_to_token = {start: i for i, (start, end) in enumerate(encoded["offset_mapping"]) if end > start}
    out = []
    for offset in doc.candidate_offsets:
        index = start_to_token.get(offset + 1)
        if index is None:
            raise ValueError(f"Cannot align right context in {doc.doc_id} at {offset}")
        out.append(index)
    return out


def insert_separators(ids: list[int], slots: list[tuple[int, int]], right_indices: list[int],
                      breaks: list[bool], sep_id: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Insert ``sep_id`` before the right-context token of every paragraph-break gap.

    A slot boundary x moves to x + #{inserted positions p < x}, so a [SEP] inserted exactly at a
    slot's end ends up right of the slot: "... left [MASK] [SEP] right ...".
    """
    positions = sorted({r for r, b in zip(right_indices, breaks, strict=True) if b})
    out: list[int] = []
    cursor = 0
    for p in positions:
        out.extend(ids[cursor:p])
        out.append(sep_id)
        cursor = p
    out.extend(ids[cursor:])
    shift = np.asarray(positions, dtype=np.int64)

    def moved(x: int) -> int:
        return x + int(np.searchsorted(shift, x, side="left"))
    return out, [(moved(s), moved(e)) for s, e in slots]


class FlatCorpus:
    """All documents' token ids in one array, followed by a pool of special tokens that
    window construction can point into (an inserted [MASK] or ".", and [CLS]/[SEP]/[PAD])."""

    def __init__(self, documents: list[tuple[list[int], list[tuple[int, int]]]],
                 mask_id: int, period_id: int, cls_id: int, sep_id: int, pad_id: int):
        pieces, s, e, start, end, doc_of = [], [], [], [], [], []
        base = 0
        for d, (ids, slots) in enumerate(documents):
            pieces.append(np.asarray(ids, dtype=np.int64))
            for a, b in slots:
                s.append(base + a)
                e.append(base + b)
                start.append(base)
                end.append(base + len(ids))
                doc_of.append(d)
            base += len(ids)
        self.special_ids = {"mask": mask_id, "period": period_id, "cls": cls_id, "sep": sep_id, "pad": pad_id}
        self.at = {name: base + i for i, name in enumerate(self.special_ids)}
        self.src = np.concatenate(pieces + [np.asarray(list(self.special_ids.values()), dtype=np.int64)])
        self.s, self.e = np.asarray(s, dtype=np.int64), np.asarray(e, dtype=np.int64)
        self.start, self.end = np.asarray(start, dtype=np.int64), np.asarray(end, dtype=np.int64)
        self.doc_of = np.asarray(doc_of, dtype=np.int64)


def mask_specs(corpus: FlatCorpus, left_budget: int, right_budget: int) -> dict[str, np.ndarray]:
    """One row per gap: the slot replaced by a single [MASK] (the v1.1 query)."""
    nl = np.minimum(left_budget, corpus.s - corpus.start)
    nr = np.minimum(right_budget, corpus.end - corpus.e)
    n = len(corpus.s)
    return {"s": corpus.s, "e": corpus.e, "nl": nl, "nr": nr,
            "coff": np.full(n, corpus.at["mask"]), "c": np.ones(n, dtype=np.int64),
            "tpos": 1 + nl, "gap": np.arange(n)}


def lr_specs(corpus: FlatCorpus, left_budget: int, right_budget: int) -> dict[str, np.ndarray]:
    """Rows for the insertion likelihood ratio: for every gap, condition A (terminal mark at the
    gap) and B (none), each with one of the 2 left / 3 right tokens masked. Targets outside the
    window or on special tokens are dropped."""
    has_mark = corpus.e > corpus.s
    nl = np.minimum(left_budget, corpus.s - corpus.start)
    nr = np.minimum(right_budget, corpus.end - corpus.e)
    special = set(corpus.special_ids[k] for k in ("cls", "sep", "pad", "mask"))
    rows = defaultdict(list)
    for cond in (0, 1):                                   # 0 = A (mark), 1 = B (no mark)
        coff = np.where(has_mark, corpus.s, corpus.at["period"]) if cond == 0 else np.full(len(corpus.s), corpus.at["period"])
        c = np.where(has_mark, corpus.e - corpus.s, 1) if cond == 0 else np.zeros(len(corpus.s), dtype=np.int64)
        for j in range(LEFT_TARGETS + RIGHT_TARGETS):
            if j < LEFT_TARGETS:                          # j=0: token just left of the slot
                valid = nl >= j + 1
                source = corpus.s - (j + 1)
                tpos = nl - j
            else:
                k = j - LEFT_TARGETS                      # k=0: token just right of the slot
                valid = nr >= k + 1
                source = corpus.e + k
                tpos = nl + c + 1 + k
            source = np.where(valid, source, 0)
            valid &= ~np.isin(corpus.src[source], list(special))
            idx = np.flatnonzero(valid)
            for name, values in (("s", corpus.s), ("e", corpus.e), ("nl", nl), ("nr", nr), ("coff", coff),
                                 ("c", c), ("tpos", tpos), ("source", source)):
                rows[name].append(values[idx])
            rows["gap"].append(idx)
            rows["cond"].append(np.full(len(idx), cond))
            rows["target"].append(np.full(len(idx), j))
    return {k: np.concatenate(v).astype(np.int64) for k, v in rows.items()}


def gather_windows(torch, src, spec: dict, at: dict[str, int]):
    """Token ids and attention mask for a batch of row specs (all tensors on one device):
    [CLS] src[s-nl:s] content(c tokens from coff) src[e:e+nr] [SEP]."""
    nl, nr, c, s, e, coff = (spec[k] for k in ("nl", "nr", "c", "s", "e", "coff"))
    length = nl + c + nr + 2
    width = int(length.max())
    k = torch.arange(width, device=src.device)[None, :]
    nl_, c_, nr_ = nl[:, None], c[:, None], nr[:, None]
    idx = torch.full((len(nl), width), at["pad"], dtype=torch.long, device=src.device)
    idx = torch.where(k == 0, at["cls"], idx)
    idx = torch.where((k >= 1) & (k <= nl_), s[:, None] - nl_ + k - 1, idx)
    idx = torch.where((k > nl_) & (k <= nl_ + c_), coff[:, None] + k - nl_ - 1, idx)
    idx = torch.where((k > nl_ + c_) & (k <= nl_ + c_ + nr_), e[:, None] + k - nl_ - c_ - 1, idx)
    idx = torch.where(k == nl_ + c_ + nr_ + 1, at["sep"], idx)
    return src[idx], (k < length[:, None]).long()


def _batches(spec: dict, batch_size: int, torch, device):
    n = len(spec["nl"])
    tensors = {k: torch.as_tensor(v, device=device) for k, v in spec.items()}
    for start in range(0, n, batch_size):
        yield start, {k: v[start:start + batch_size] for k, v in tensors.items()}


def run_mask_rows(model, torch, src, spec, at, punct_ids, batch_size):
    """P(.?!) at [MASK] and the probe hidden states (fp16) for every gap."""
    n = len(spec["nl"])
    probability = torch.empty(n, device=src.device)
    hidden = torch.empty((n, model.config.hidden_size * len(PROBE_LAYERS)), dtype=torch.float16, device=src.device)
    punct = torch.as_tensor(punct_ids, device=src.device)
    with torch.inference_mode():
        for start, batch in _batches(spec, batch_size, torch, src.device):
            ids, attention = gather_windows(torch, src, batch, at)
            out = model.bert(input_ids=ids, attention_mask=attention, output_hidden_states=True, return_dict=True)
            rows = torch.arange(len(ids), device=src.device)
            states = [out.hidden_states[layer][rows, batch["tpos"]] for layer in PROBE_LAYERS]
            logits = model.cls(out.last_hidden_state[rows, batch["tpos"]]).float()
            probability[start:start + len(ids)] = torch.exp(
                torch.logsumexp(logits[:, punct], dim=-1) - torch.logsumexp(logits, dim=-1))
            hidden[start:start + len(ids)] = torch.cat(states, dim=-1).half()
    return probability, hidden


def run_lr_rows(model, torch, src, spec, at, mask_id, batch_size):
    """log p(original token) at the masked target of every likelihood-ratio row."""
    n = len(spec["nl"])
    logp = torch.empty(n, device=src.device)
    with torch.inference_mode():
        for start, batch in _batches(spec, batch_size, torch, src.device):
            ids, attention = gather_windows(torch, src, batch, at)
            rows = torch.arange(len(ids), device=src.device)
            labels = src[batch["source"]]
            if not torch.equal(ids[rows, batch["tpos"]], labels):
                raise RuntimeError("Likelihood-ratio target position does not hold its source token")
            ids[rows, batch["tpos"]] = mask_id
            hidden = model.bert(input_ids=ids, attention_mask=attention, return_dict=True).last_hidden_state
            logits = model.cls(hidden[rows, batch["tpos"]]).float()
            logp[start:start + len(ids)] = torch.log_softmax(logits, dim=-1)[rows, labels]
    return logp


def lr_features(spec: dict, logp: np.ndarray, n_gaps: int) -> np.ndarray:
    """Per gap: Δ_j = log p_A − log p_B for the 5 targets (0 if missing), validity flags,
    and the left / right sums of Δ."""
    targets = LEFT_TARGETS + RIGHT_TARGETS
    value = np.zeros((2, n_gaps, targets))
    seen = np.zeros((2, n_gaps, targets), dtype=bool)
    value[spec["cond"], spec["gap"], spec["target"]] = logp
    seen[spec["cond"], spec["gap"], spec["target"]] = True
    valid = seen[0] & seen[1]
    delta = np.where(valid, value[0] - value[1], 0.0)
    return np.concatenate([delta, valid.astype(float), delta[:, :LEFT_TARGETS].sum(1, keepdims=True),
                           delta[:, LEFT_TARGETS:].sum(1, keepdims=True)], axis=1)


def fit_logistic(torch, x, y, lam: float, max_iter: int = 50, tol: float = 1e-9):
    """L2 logistic regression on standardised features (mean BCE + lam/2 |w|^2, bias unpenalised),
    by damped Newton steps: each step is one d x d Hessian (a large matmul, GPU-friendly) solved in
    float64, with Armijo backtracking. Computed in float64 throughout."""
    x, y = x.double(), y.double()
    mu = x.mean(0)
    sd = x.std(0).clamp_min(1e-6)
    n, d = x.shape
    z = torch.cat([(x - mu) / sd, torch.ones((n, 1), dtype=x.dtype, device=x.device)], dim=1)
    reg = torch.full((d + 1,), float(lam), dtype=x.dtype, device=x.device)
    reg[-1] = 0.0
    theta = torch.zeros(d + 1, dtype=x.dtype, device=x.device)
    rate = y.mean().clamp(1e-4, 1 - 1e-4)
    theta[-1] = torch.log(rate / (1 - rate))

    def objective(t):
        s = z @ t
        return float((torch.nn.functional.softplus(s) - y * s).mean() + 0.5 * (reg * t * t).sum())

    grad = None
    for _ in range(max_iter):
        s = z @ theta
        p = torch.sigmoid(s)
        grad = z.T @ (p - y) / n + reg * theta
        if float(grad.abs().max()) < tol:
            break
        weights = (p * (1 - p) / n).clamp_min(1e-12)
        hessian = z.T @ (z * weights[:, None]) + torch.diag(reg + 1e-12)
        step = torch.linalg.solve(hessian, grad)
        f0, slope, t = objective(theta), float(grad @ step), 1.0
        while objective(theta - t * step) > f0 - 1e-4 * t * slope and t > 1e-8:
            t *= 0.5
        theta = theta - t * step
    s = z @ theta
    grad = z.T @ (torch.sigmoid(s) - y) / n + reg * theta
    return {"w": theta[:-1], "b": theta[-1], "mu": mu, "sd": sd, "loss": objective(theta),
            "max_grad": float(grad.abs().max())}


def predict_logistic(torch, model: dict, x):
    return torch.sigmoid(((x.double() - model["mu"]) / model["sd"]) @ model["w"] + model["b"]).to(x.dtype)


def extract(args, torch, config: dict, surfaces, layout: dict, reference: dict) -> tuple[dict, dict]:
    """GPU stage: [MASK] rows and likelihood-ratio rows for both input variants."""
    bert = config["bert"]
    protocol = bert["mask_protocol"]
    left_budget, right_budget = window_budgets(bert["max_length"])
    model, tokenizer, punct_ids, _ = load_surface_bert(bert["model_name"], os.environ["HF_HUB_CACHE"])
    ids_of = dict(mask=tokenizer.mask_token_id, period=tokenizer.convert_tokens_to_ids("."),
                  cls=tokenizer.cls_token_id, sep=tokenizer.sep_token_id, pad=tokenizer.pad_token_id)
    plans = {}
    for doc in surfaces:
        full_ids, plan = plan_queries(tokenizer, doc, protocol)
        plans[doc.doc_id] = (full_ids, [(p[2], p[3]) for p in plan], right_token_indices(tokenizer, doc))
    timings, features = {}, {}
    for variant in VARIANTS:
        started = time.time()
        documents = []
        for doc in surfaces:
            full_ids, slots, right = plans[doc.doc_id]
            if variant == "para":
                breaks = set(layout[doc.doc_id]["paragraph_break_offsets"])
                full_ids, slots = insert_separators(full_ids, slots, right,
                                                    [o in breaks for o in doc.candidate_offsets], ids_of["sep"])
            documents.append((full_ids, slots))
        corpus = FlatCorpus(documents, **{f"{k}_id": v for k, v in ids_of.items()})
        src = torch.as_tensor(corpus.src, device="cuda")
        m_spec = mask_specs(corpus, left_budget, right_budget)
        l_spec = lr_specs(corpus, left_budget, right_budget)
        probability, hidden = run_mask_rows(model, torch, src, m_spec, corpus.at, punct_ids, args.mask_batch)
        torch.cuda.synchronize()
        mask_seconds = time.time() - started
        logp = run_lr_rows(model, torch, src, l_spec, corpus.at, ids_of["mask"], args.lr_batch)
        torch.cuda.synchronize()
        p = probability.cpu().numpy().astype(float)
        if variant == "raw":
            cached = np.concatenate([reference[d.doc_id]["raw_punctuation_probability"] for d in surfaces])
            diff = np.abs(p - cached)
            timings["raw_vs_cache"] = {"max_abs_diff": float(diff.max()), "p99_abs_diff": float(np.quantile(diff, 0.99))}
            if diff.max() > 0.05:
                raise RuntimeError(f"GPU-gathered raw queries disagree with the frozen cache: {diff.max()}")
        lr = lr_features(l_spec, logp.cpu().numpy().astype(float), len(p))
        features[variant] = {"p": p, "hidden": hidden.cpu(), "lr": lr}
        timings[variant] = {"mask_rows": len(m_spec["nl"]), "lr_rows": len(l_spec["nl"]),
                            "mask_seconds": round(mask_seconds, 1),
                            "total_seconds": round(time.time() - started, 1)}
        print(json.dumps({"status": "PROBE_EXTRACTION", "variant": variant, **timings[variant]}), flush=True)
        destination = args.feature_root / variant
        destination.mkdir(parents=True, exist_ok=True)
        torch.save({"doc_ids": [d.doc_id for d in surfaces],
                    "gaps_per_doc": [len(d.candidate_offsets) for d in surfaces],
                    "p_punct": torch.as_tensor(p), "hidden": features[variant]["hidden"], "lr": torch.as_tensor(lr),
                    "probe_layers": PROBE_LAYERS, "protocol": protocol, "model_name": bert["model_name"],
                    "timings": timings[variant], "raw_vs_cache": timings.get("raw_vs_cache")},
                   destination / "train.pt")
        del src
    return features, timings


def load_features(torch, feature_root: Path, surfaces) -> tuple[dict, dict]:
    features, timings = {}, {}
    for variant in VARIANTS:
        saved = torch.load(feature_root / variant / "train.pt", weights_only=False)
        if saved["doc_ids"] != [d.doc_id for d in surfaces]:
            raise RuntimeError("Saved features do not match the documents")
        features[variant] = {"p": saved["p_punct"].numpy(), "hidden": saved["hidden"], "lr": saved["lr"].numpy()}
        timings[variant] = saved.get("timings")
        if saved.get("raw_vs_cache"):
            timings["raw_vs_cache"] = saved["raw_vs_cache"]
    return features, timings


def crossfit(args, torch, features: dict, surfaces, raw_docs: dict, layout: dict, device: str) -> dict:
    """Cross-fitted evidence for every (variant, configuration, lambda, scheme). Gold enters only
    here, as GUM-train labels; each document's evidence comes from folds that never saw it."""
    gold = {}
    for record in read_gold_boundaries(args.data_root / "gold_train.jsonl"):
        record.validate_against(raw_docs[record.doc_id])
        gold[record.doc_id] = set(record.offsets)
    doc_index = np.concatenate([[i] * len(d.candidate_offsets) for i, d in enumerate(surfaces)]).astype(int)
    y_np = np.concatenate([[int(o in gold[d.doc_id]) for o in d.candidate_offsets] for d in surfaces]).astype(float)
    terminal = np.concatenate([[left_context_category(d.text, o) == "terminal" for o in d.candidate_offsets]
                               for d in surfaces])
    breaks = np.concatenate([[o in set(layout[d.doc_id]["paragraph_break_offsets"]) for o in d.candidate_offsets]
                             for d in surfaces])
    folds5 = json.loads((args.cv_root / "folds.json").read_text())["folds"]
    genres = sorted({d.genre for d in surfaces})
    schemes = {"fold5": np.array([folds5[d.doc_id] for d in surfaces]),
               "logo": np.array([genres.index(d.genre) for d in surfaces])}
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)

    fit_log, diagnostics = [], {}
    started = time.time()
    for variant in VARIANTS:
        f = features[variant]
        clipped = np.clip(f["p"], 1e-7, 1 - 1e-7)
        blocks = {"logit": torch.as_tensor((np.log(clipped) - np.log1p(-clipped))[:, None], dtype=torch.float32, device=device),
                  "lr": torch.as_tensor(f["lr"], dtype=torch.float32, device=device),
                  "hidden": torch.as_tensor(f["hidden"], device=device).float()}
        for name, (parts, lambdas) in CONFIGS.items():
            x = torch.cat([blocks[b] for b in parts], dim=1)
            for lam in lambdas:
                for scheme, fold_of_doc in schemes.items():
                    fold_of_gap = torch.as_tensor(fold_of_doc[doc_index], device=device)
                    evidence = torch.empty_like(y)
                    for fold in np.unique(fold_of_doc):
                        held = fold_of_gap == int(fold)
                        fitted = fit_logistic(torch, x[~held], y[~held], lam)
                        evidence[held] = predict_logistic(torch, fitted, x[held])
                        fit_log.append({"variant": variant, "config": name, "lambda": lam, "scheme": scheme,
                                        "fold": int(fold), "loss": fitted["loss"], "max_grad": fitted["max_grad"]})
                    e = evidence.cpu().numpy().astype(float)
                    diagnostics[f"{variant}/{name}/lam={lam:g}/{scheme}"] = {
                        "mean_p_gold_no_terminal": float(e[(y_np == 1) & ~terminal].mean()),
                        "share_gold_no_terminal_above_half": float((e[(y_np == 1) & ~terminal] > 0.5).mean()),
                        "mean_p_gold_terminal": float(e[(y_np == 1) & terminal].mean()),
                        "mean_p_nongold_terminal": float(e[(y_np == 0) & terminal].mean()),
                        "mean_p_nongold_no_terminal": float(e[(y_np == 0) & ~terminal].mean()),
                        "mean_p_gold_no_terminal_not_paragraph": float(e[(y_np == 1) & ~terminal & ~breaks].mean()),
                        "heldout_bce": float(-np.mean(y_np * np.log(np.clip(e, 1e-7, 1)) +
                                                      (1 - y_np) * np.log(np.clip(1 - e, 1e-7, 1)))),
                    }
                    path = args.output_root / variant / f"{name}_lam{lam:g}" / f"{scheme}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    start = 0
                    with path.open("w") as out:
                        for d in surfaces:
                            n = len(d.candidate_offsets)
                            out.write(json.dumps({"doc_id": d.doc_id, "text_sha256": d.text_sha256,
                                                  "candidate_offsets": d.candidate_offsets,
                                                  "calibrated_boundary_probability": e[start:start + n].tolist(),
                                                  "variant": variant, "config": name, "lambda": lam,
                                                  "scheme": scheme}) + "\n")
                            start += n
            del x
        print(json.dumps({"status": "PROBE_CROSSFIT", "variant": variant,
                          "seconds": round(time.time() - started, 1)}), flush=True)
    corpus = {"gaps": int(len(y_np)), "gold": int(y_np.sum()), "gold_no_terminal": int(((y_np == 1) & ~terminal).sum()),
              "paragraph_breaks": int(breaks.sum()), "gold_paragraph": int((breaks & (y_np == 1)).sum()),
              "gold_no_terminal_paragraph": int((breaks & (y_np == 1) & ~terminal).sum())}
    return {"diagnostics": diagnostics, "corpus": corpus, "fits": fit_log,
            "fitting_seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_splitters.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/real_system")
    parser.add_argument("--cv-root", type=Path, default=ROOT / "data/real_system/train_cv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/real_system/train_cv/boundary_probe")
    parser.add_argument("--feature-root", type=Path, default=ROOT / "data/lm_cache/bert_probe_v1.1")
    parser.add_argument("--mask-batch", type=int, default=512)
    parser.add_argument("--lr-batch", type=int, default=1024)
    parser.add_argument("--from-features", action="store_true", help="skip the GPU extraction, reuse saved features")
    parser.add_argument("--stop-after-extraction", action="store_true",
                        help="save the features and stop (cross-fitting reads gold; run it after the tests pass)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    config = json.loads(args.config.read_text())
    version = PROTOCOLS[config["bert"]["mask_protocol"]]
    surfaces = sorted(load_surface_jsonl(args.data_root / "surface_train.jsonl"), key=lambda d: d.doc_id)
    raw_docs = {d.doc_id: d for d in read_raw_documents(args.data_root / "raw_train.jsonl")}
    layout = {}
    with (args.data_root / "layout_train.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            layout[row["doc_id"]] = row
    reference = read_cache(ROOT / "data/lm_cache" / f"bert_surface_{version}" / "train.jsonl")
    for doc in surfaces:
        if raw_docs[doc.doc_id].sha256 != doc.text_sha256 or layout[doc.doc_id]["text_sha256"] != doc.text_sha256:
            raise RuntimeError(f"Artifact hash mismatch for {doc.doc_id}")
        if reference[doc.doc_id]["candidate_offsets"] != doc.candidate_offsets:
            raise RuntimeError(f"Reference cache gaps differ for {doc.doc_id}")

    if args.from_features:
        features, timings = load_features(torch, args.feature_root, surfaces)
    else:
        features, timings = extract(args, torch, config, surfaces, layout, reference)
        if args.stop_after_extraction:
            print(json.dumps({"status": "PROBE_EXTRACTION_COMPLETE", **timings}), flush=True)
            return
    fitted = crossfit(args, torch, features, surfaces, raw_docs, layout, args.device)
    timings["fitting_seconds"] = fitted.pop("fitting_seconds")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "probe_run.json").write_text(json.dumps({"timings": timings, **fitted}, indent=2) + "\n")
    print(json.dumps({"status": "BOUNDARY_PROBE_COMPLETE", **timings}), flush=True)


if __name__ == "__main__":
    main()
