"""GPU stages of amendment 6 (all gold-free on the evaluated corpus).

  sat         zero-shot SaT character probabilities on the lower-cased raw text -> per-gap raw p
              (``sat-12l``: main candidate E1; ``sat-12l-sm``: analysis only, GUM-contaminated)
  finetune    E2/E2b: token classifiers fine-tuned on GUM train (lower-cased, punctuation dropout);
              on GUM: leave-one-genre-out out-of-fold logits + one final model on all of GUM train;
              on a test corpus: logits of the saved final model
  stability   every splitter re-run on 64-word chunks (start offsets 0 and 32)

Gold enters only in ``finetune --corpus gum`` as GUM-train training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np

from .a6_data import CORPORA, ROOT, SEED, SYSTEMS, doc_slices, gold_sets, logo_folds, safe_lower, \
    save_gap_array, surfaces, word_spans, write_jsonl


MODEL_CACHE = ROOT / "data/lm_cache/a6"
FT_MODELS = {"ft-modernbert": "answerdotai/ModernBERT-large", "ft-bert-uncased": "google-bert/bert-base-uncased"}
SAT_MODELS = {"sat12l": "sat-12l", "sat12l-sm": "sat-12l-sm"}
WINDOW, STRIDE = 510, 256
EPOCHS, LR, BATCH, WARMUP, DECAY, DROPOUT_P = 3, 2e-5, 16, 0.1, 0.01, 0.3
CHUNK_WORDS, CHUNK_OFFSETS = 64, (0, 32)
TERMINAL_TAIL = re.compile(r"[.?!]+(?=[\"'”’»)\]}]*$)")


# ------------------------------------------------------------------ SaT zero-shot
def run_sat(args) -> None:
    import torch
    from wtpsplit import SaT

    corpus = CORPORA[args.corpus]
    docs = surfaces(corpus)
    texts = [safe_lower(d.text) for d in docs]
    for name, model_name in SAT_MODELS.items():
        started = time.time()
        sat = SaT(model_name)
        sat.half().to("cuda")
        probs = list(sat.predict_proba(texts, stride=64, block_size=512, batch_size=64))
        p = []
        for d, prob in zip(docs, probs):
            prob = np.asarray(prob, dtype=float)
            if len(prob) != len(d.text):
                raise RuntimeError(f"SaT returned {len(prob)} probabilities for {len(d.text)} chars in {d.doc_id}")
            o = np.asarray(d.candidate_offsets)
            p.append(np.maximum(prob[o - 1], prob[o]))
        save_gap_array(corpus, f"{name}_raw", docs, np.concatenate(p))
        print(json.dumps({"status": "A6_SAT", "corpus": corpus.name, "model": model_name,
                          "seconds": round(time.time() - started, 1)}), flush=True)
        del sat
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ fine-tuned token classifiers
def corrupt(text: str, spans, labels, rng: random.Random) -> tuple[str, list[tuple[int, int]]]:
    """Drop the terminal marks before Gold boundaries with probability DROPOUT_P (training only)."""
    pieces, new_spans, cursor = [], [], 0
    for k, (s, e) in enumerate(spans):
        word = text[s:e]
        if k < len(labels) and labels[k] and rng.random() < DROPOUT_P:
            m = TERMINAL_TAIL.search(word)
            if m and m.start() > 0:
                word = word[:m.start()] + word[m.end():]
        sep = text[e:spans[k + 1][0]] if k + 1 < len(spans) else ""
        new_spans.append((cursor, cursor + len(word)))
        pieces.append(word + sep)
        cursor += len(word) + len(sep)
    return "".join(pieces), new_spans


def encode(tokenizer, text: str, spans) -> tuple[list[int], list[int]]:
    """Token ids of the text and, per word, the index of its last sub-token."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ends = np.array([e if e > s else -1 for s, e in enc["offset_mapping"]])
    starts = np.array([s for s, _ in enc["offset_mapping"]])
    word_token = []
    for s, e in spans:
        inside = np.flatnonzero((ends > s) & (ends <= e) & (starts < e))
        if not len(inside):
            raise RuntimeError(f"No token for word {text[s:e]!r}")
        word_token.append(int(inside[-1]))
    return enc["input_ids"], word_token


def windows(n_tokens: int, shift: int = 0) -> list[tuple[int, int]]:
    starts = sorted({max(0, s) for s in range(-shift, max(n_tokens - WINDOW, 0) + STRIDE, STRIDE)})
    out = [(s, min(s + WINDOW, n_tokens)) for s in starts if s < n_tokens]
    if out and out[-1][1] < n_tokens:
        out.append((max(0, n_tokens - WINDOW), n_tokens))
    return out


class Tagger:
    def __init__(self, torch, model_name: str):
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).cuda()
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1).cuda()
        self.cls, self.sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        self.pad = self.tokenizer.pad_token_id

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.head.parameters())

    def forward(self, ids, attention):
        torch = self.torch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.encoder(input_ids=ids, attention_mask=attention).last_hidden_state
            return self.head(hidden).squeeze(-1).float()

    def batch(self, rows):
        torch = self.torch
        width = max(len(r[0]) for r in rows) + 2
        ids = torch.full((len(rows), width), self.pad, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), width), -100.0)
        for i, (chunk, positions, targets) in enumerate(rows):
            seq = [self.cls] + chunk + [self.sep]
            ids[i, :len(seq)] = torch.as_tensor(seq)
            attention[i, :len(seq)] = 1
            for pos, y in zip(positions, targets):
                labels[i, pos + 1] = y
        return ids.cuda(), attention.cuda(), labels.cuda()


def training_rows(tagger: Tagger, docs, gold, rng: random.Random):
    rows = []
    for d in docs:
        text = safe_lower(d.text)
        spans = word_spans(text)
        labels = [int(o in gold[d.doc_id]) for o in d.candidate_offsets]
        if not labels:
            continue
        text, spans = corrupt(text, spans, labels, rng)
        ids, word_token = encode(tagger.tokenizer, text, spans[:len(labels)])
        for a, b in windows(len(ids), shift=rng.randrange(STRIDE)):
            pos = [(t - a, y) for t, y in zip(word_token, labels) if a <= t < b]
            if pos:
                rows.append((ids[a:b], [p for p, _ in pos], [y for _, y in pos]))
    rng.shuffle(rows)
    return rows


def train(torch, model_name: str, docs, gold, seed: int) -> Tagger:
    random.seed(seed)
    torch.manual_seed(seed)
    tagger = Tagger(torch, model_name)
    rng = random.Random(seed)
    steps_per_epoch = math.ceil(len(training_rows(tagger, docs, gold, random.Random(seed))) / BATCH)
    total = steps_per_epoch * EPOCHS
    optimizer = torch.optim.AdamW(tagger.parameters(), lr=LR, weight_decay=DECAY)
    warm = max(1, int(WARMUP * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    tagger.encoder.train()
    step = 0
    for _ in range(EPOCHS):
        rows = training_rows(tagger, docs, gold, rng)
        for start in range(0, len(rows), BATCH):
            ids, attention, labels = tagger.batch(rows[start:start + BATCH])
            logits = tagger.forward(ids, attention)
            mask = labels >= 0
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step >= total:
                break
    tagger.encoder.eval()
    return tagger


def predict(tagger: Tagger, docs) -> np.ndarray:
    """Per-gap logit; each token is predicted by the window where it is most central."""
    torch = tagger.torch
    out = []
    for d in docs:
        text = safe_lower(d.text)
        spans = word_spans(text)
        n_gaps = len(d.candidate_offsets)
        if n_gaps == 0:
            continue
        ids, word_token = encode(tagger.tokenizer, text, spans[:n_gaps])
        spans_w = windows(len(ids))
        token_logit = np.zeros(len(ids))
        rows = []
        for i, (a, b) in enumerate(spans_w):
            lo = a if i == 0 else a + (WINDOW - STRIDE) // 2
            hi = b if i == len(spans_w) - 1 else a + (WINDOW + STRIDE) // 2
            rows.append((a, b, lo, hi))
        for start in range(0, len(rows), 32):
            group = rows[start:start + 32]
            ids_t, attention, _ = tagger.batch([(ids[a:b], [], []) for a, b, _, _ in group])
            with torch.inference_mode():
                logits = tagger.forward(ids_t, attention).cpu().numpy()
            for (a, b, lo, hi), row in zip(group, logits):
                token_logit[lo:hi] = row[1 + lo - a:1 + hi - a]
        out.append(token_logit[np.asarray(word_token)])
    return np.concatenate(out)


def run_finetune(args) -> None:
    import torch

    corpus = CORPORA[args.corpus]
    docs = surfaces(corpus)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    k = args.seed_offset
    if k and corpus.name != "gum":
        raise RuntimeError("--seed-offset is a GUM-train LOGO analysis (amendment 7.4)")
    for name in args.models:
        model_name = FT_MODELS[name]
        out_name = f"{name}-seed{k}" if k else name
        started = time.time()
        if corpus.name == "gum":
            gold = gold_sets(corpus)
            genre_of_gap = logo_folds(docs)
            genres = sorted({d.genre for d in docs})
            logit = np.zeros(len(genre_of_gap))
            slices = doc_slices(docs)
            for g, genre in enumerate(genres):
                train_docs = [d for d in docs if d.genre != genre]
                held = [d for d in docs if d.genre == genre]
                tagger = train(torch, model_name, train_docs, gold, SEED + 100 * k + g)
                pred = predict(tagger, held)
                idx = np.concatenate([np.arange(sl.start, sl.stop) for d, sl in zip(docs, slices) if d.genre == genre])
                logit[idx] = pred
                print(json.dumps({"status": "A6_FT_FOLD", "model": out_name, "genre": genre,
                                  "seconds": round(time.time() - started, 1)}), flush=True)
                del tagger
                torch.cuda.empty_cache()
            save_gap_array(corpus, f"{out_name}_logit", docs, logit)
            if k:                     # amendment 7.4: LOGO only, the test components stay untouched
                print(json.dumps({"status": "A6_FT_COMPLETE", "corpus": corpus.name, "model": out_name,
                                  "seconds": round(time.time() - started, 1)}), flush=True)
                continue
            tagger = train(torch, model_name, docs, gold, SEED)
            torch.save({"encoder": {k: v.to(torch.bfloat16) for k, v in tagger.encoder.state_dict().items()},
                        "head": tagger.head.state_dict(), "model_name": model_name}, MODEL_CACHE / f"{name}_final.pt")
        else:
            saved = torch.load(MODEL_CACHE / f"{name}_final.pt", weights_only=False)
            tagger = Tagger(torch, saved["model_name"])
            tagger.encoder.load_state_dict({k: v.float() for k, v in saved["encoder"].items()})
            tagger.head.load_state_dict(saved["head"])
            tagger.encoder.eval()
            save_gap_array(corpus, f"{name}_logit", docs, predict(tagger, docs))
        print(json.dumps({"status": "A6_FT_COMPLETE", "corpus": corpus.name, "model": name,
                          "seconds": round(time.time() - started, 1)}), flush=True)
        del tagger
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ perturbation stability
def chunk_plan(doc, offset: int) -> list[tuple[int, int]]:
    """Word-index ranges [a, b) of 64-word chunks, the first one shortened by ``offset``."""
    n_words = len(doc.candidate_offsets) + 1
    cuts = [0] + list(range(offset or CHUNK_WORDS, n_words, CHUNK_WORDS)) + [n_words]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def run_stability(args) -> None:
    from .real_splitters import RealSplitterSuite

    corpus = CORPORA[args.corpus]
    docs = surfaces(corpus)
    suite = RealSplitterSuite(stanza_dir=str(ROOT / ".cache/stanza"), use_gpu=True)
    rows = []
    for system in SYSTEMS:
        started = time.time()
        for offset in CHUNK_OFFSETS:
            texts, meta = [], []
            for d in docs:
                spans = word_spans(d.text)
                for a, b in chunk_plan(d, offset):
                    texts.append(d.text[spans[a][0]:spans[b - 1][1]])
                    meta.append((d.doc_id, spans[a][0], spans[b - 1][1]))
            raw = suite.predict_raw_offsets(system, texts)
            found, edges = {}, {}
            for (doc_id, start, end), offsets in zip(meta, raw):
                found.setdefault(doc_id, set()).update(start + o for o in offsets if 0 < o < end - start)
                edges.setdefault(doc_id, set()).add(end)
            for d in docs:
                gaps = set(d.candidate_offsets)
                rows.append({"doc_id": d.doc_id, "system": system, "chunk_offset": offset,
                             "offsets": sorted(found.get(d.doc_id, set()) & gaps),
                             "edge_gaps": sorted(edges[d.doc_id] & gaps)})
        print(json.dumps({"status": "A6_STABILITY", "corpus": corpus.name, "system": system,
                          "seconds": round(time.time() - started, 1)}), flush=True)
    write_jsonl(rows, corpus.path("stability.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["sat", "finetune", "stability"])
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="gum")
    parser.add_argument("--models", nargs="*", default=list(FT_MODELS))
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="amendment 7.4: fold seed = SEED + 100*k + g; LOGO only, saved as <model>-seed<k>")
    args = parser.parse_args()
    {"sat": run_sat, "finetune": run_finetune, "stability": run_stability}[args.stage](args)


if __name__ == "__main__":
    main()
