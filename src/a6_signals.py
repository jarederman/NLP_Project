"""CPU signals for amendment 6: rule features (regex / finite-state), an interpolated Kneser-Ney
trigram LM, the hidden semi-Markov posterior with a sentence-length prior, Dawid-Skene EM over
splitter votes, and the small fitting helpers (Platt, L2 logistic, ridge).

Everything here is gold-free at use time. Gold enters only as training labels from GUM train
(the length prior, the n-gram sentences, the logistic/ridge targets), always within the
training folds.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from .a6_data import safe_lower, word_spans
from .analyze_evidence_gaps import left_context_category
from .gap_features import CATEGORIES, document_features


# ------------------------------------------------------------------ rule features (course L2)
CLOSERS = "\"'”’»)]}"
LIST_MARKER = re.compile(r"^\(?(\d{1,3}|[a-z]|[ivxlc]{1,5})[.)]$")
NUMBER = re.compile(r"^[-+]?\$?\d+([.,:/]\d+)*[.%]?$")
TIME = re.compile(r"^(\d{1,2}[:.]\d{2}|[ap]\.m\.?)$")
BULLET = re.compile(r"^[-–—*•·]+$")
RULE_NAMES = (
    [f"cat_{c}" for c in CATEGORIES]
    + ["abbr", "abbr_single_letter", "abbr_short", "abbr_internal_period", "abbr_type_stat"]
    + ["log_dist_prev_terminal", "log_dist_next_terminal"]
    + ["left_list_marker", "left_single_letter_period", "left_number", "left_time",
       "right_closer_or_comma", "right_digit", "right_bullet", "right_opener", "right_list_marker",
       "left_ends_closer"]
)


def rule_features(text: str, offsets: list[int]) -> np.ndarray:
    """(n_gaps, len(RULE_NAMES)) casing-free rule features of one document."""
    n = len(offsets)
    lower = safe_lower(text)
    dummy = [0.5] * n
    groups = document_features(lower, offsets, dummy, ["insert_mask"] * n)
    spans = word_spans(lower)
    words = [lower[s:e] for s, e in spans]
    cat = np.zeros((n, len(CATEGORIES)))
    for i, o in enumerate(offsets):
        cat[i, CATEGORIES.index(left_context_category(lower, o))] = 1.0
    extra = np.zeros((n, 10))
    for i in range(n):
        left, right = words[i], words[i + 1] if i + 1 < len(words) else ""
        core = left.rstrip(CLOSERS)
        extra[i] = [
            float(bool(LIST_MARKER.match(left))),
            float(bool(re.match(r"^\(?[a-z]\.$", core))),
            float(bool(NUMBER.match(core))),
            float(bool(TIME.match(core))),
            float(right[:1] in ",;:)]}" or right[:1] in ".!?"),
            float(right[:1].isdigit()),
            float(bool(BULLET.match(right))),
            float(right[:1] in "\"'“‘«([{"),
            float(bool(LIST_MARKER.match(right))),
            float(left != core),
        ]
    return np.column_stack([cat, groups["A"], groups["L"][:, :2], extra])


# ------------------------------------------------------------------ n-gram LM (course L5)
TOKEN = re.compile(r"\w+|[^\w\s]")
BOS, EOS = "<s>", "</s>"


def tokens_of(chunk: str) -> list[str]:
    return TOKEN.findall(safe_lower(chunk))


class KneserNeyTrigram:
    """Interpolated Kneser-Ney trigram (absolute discount 0.75) over lower-cased tokens."""

    def __init__(self, sentences: list[list[str]], discount: float = 0.75):
        self.d = discount
        self.c3, self.c2h = Counter(), Counter()          # trigram counts, bigram-history totals
        self.f3 = defaultdict(set)                         # history -> distinct followers (trigram)
        self.cont2, self.cont2h, self.f2 = Counter(), Counter(), defaultdict(set)
        self.cont1 = Counter()
        bigram_types = set()
        for sentence in sentences:
            seq = [BOS, BOS] + sentence + [EOS]
            for a, b, c in zip(seq, seq[1:], seq[2:]):
                self.c3[(a, b, c)] += 1
                self.c2h[(a, b)] += 1
                self.f3[(a, b)].add(c)
                bigram_types.add((b, c))
        for (a, b, c) in self.c3:              # continuation counts for the bigram level
            self.cont2[(b, c)] += 1
        for (b, c), k in self.cont2.items():
            self.cont2h[b] += k
            self.f2[b].add(c)
        for (b, c) in bigram_types:
            self.cont1[c] += 1
        self.n_bigram_types = max(len(bigram_types), 1)
        self.vocab = len(self.cont1) + 1

    def _p1(self, w: str) -> float:
        return (self.cont1[w] + 0.5) / (self.n_bigram_types + 0.5 * self.vocab)

    def _p2(self, b: str, w: str) -> float:
        total = self.cont2h[b]
        if not total:
            return self._p1(w)
        lam = self.d * len(self.f2[b]) / total
        return max(self.cont2[(b, w)] - self.d, 0) / total + lam * self._p1(w)

    def logprob(self, a: str, b: str, w: str) -> float:
        total = self.c2h[(a, b)]
        if not total:
            return math.log(self._p2(b, w))
        lam = self.d * len(self.f3[(a, b)]) / total
        return math.log(max(self.c3[(a, b, w)] - self.d, 0) / total + lam * self._p2(b, w))


def gold_sentences(docs, gold: dict[str, set[int]]) -> list[list[str]]:
    out = []
    for d in docs:
        cuts = [0] + sorted(gold[d.doc_id]) + [len(d.text)]
        for a, b in zip(cuts, cuts[1:]):
            toks = tokens_of(d.text[a:b])
            if toks:
                out.append(toks)
    return out


NGRAM_NAMES = ("lp_end", "lp_start", "lp_continue", "ngram_lr")


def ngram_features(model: KneserNeyTrigram, text: str, offsets: list[int]) -> np.ndarray:
    spans = word_spans(text)
    chunk_tokens = [tokens_of(text[s:e]) for s, e in spans]
    flat, ends = [], []
    for toks in chunk_tokens:
        flat.extend(toks)
        ends.append(len(flat))
    out = np.zeros((len(offsets), 4))
    for i in range(len(offsets)):
        k = ends[i]
        a = flat[k - 2] if k >= 2 else BOS
        b = flat[k - 1] if k >= 1 else BOS
        nxt = flat[k] if k < len(flat) else EOS
        end = model.logprob(a, b, EOS)
        start = model.logprob(BOS, BOS, nxt)
        cont = model.logprob(a, b, nxt)
        out[i] = [end, start, cont, end + start - cont]
    return out


# ------------------------------------------------------------------ hidden semi-Markov posterior (course L3)
def length_prior(docs, gold: dict[str, set[int]], max_len: int = 200) -> np.ndarray:
    """P(L) for L = 1..max_len words, from Gold sentences, add-0.5 smoothing (index 0 unused)."""
    counts = np.full(max_len + 1, 0.5)
    counts[0] = 0.0
    for d in docs:
        g = [int(o in gold[d.doc_id]) for o in d.candidate_offsets]
        last = -1
        for i, b in enumerate(g + [1]):        # the document end closes the last sentence
            if b:
                counts[min(i - last, max_len)] += 1
                last = i
    return counts / counts.sum()


def _logsumexp(x: np.ndarray) -> float:
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum())) if np.isfinite(m) else -np.inf


def hsmm_posterior(p: np.ndarray, prior_len: np.ndarray, base_rate: float) -> np.ndarray:
    """Posterior boundary marginals of one document. Words 0..n (n = #gaps); gap j follows word j.
    Emission of a boundary at gap j = evidence odds / prior odds; the document end is a boundary
    with emission 1. Exact forward-backward in log space over segment lengths 1..Lmax."""
    n = len(p)
    if n == 0:
        return np.zeros(0)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    log_e = np.log(q / (1 - q)) - math.log(base_rate / (1 - base_rate))
    log_e = np.append(log_e, 0.0)                      # index n = document end
    lmax = len(prior_len) - 1
    log_pl = np.log(np.maximum(prior_len[1:], 1e-300))  # index L-1
    # alpha[j+1] = log prob of a segmentation of words 0..j with a boundary after word j
    alpha = np.full(n + 2, -np.inf)
    alpha[0] = 0.0
    for j in range(n + 1):
        lo = max(0, j + 1 - lmax)
        prev = alpha[lo:j + 1]                         # segments starting at word lo..j
        lengths = j + 1 - np.arange(lo, j + 1)
        alpha[j + 1] = log_e[j] + _logsumexp(prev + log_pl[lengths - 1])
    z = alpha[n + 1]
    beta = np.full(n + 2, -np.inf)                     # beta[j+1]: words j+1..n given a boundary after j
    beta[n + 1] = 0.0
    for j in range(n - 1, -2, -1):
        hi = min(n, j + lmax)
        ends = np.arange(j + 1, hi + 1)                # segment = words j+1..end
        beta[j + 1] = _logsumexp(log_pl[ends - j - 1] + log_e[ends] + beta[ends + 1])
    return np.exp(np.clip(alpha[1:n + 1] + beta[1:n + 1] - z, -700, 0))


# ------------------------------------------------------------------ Dawid-Skene EM (course L3: EM)
def dawid_skene(votes: np.ndarray, strata: np.ndarray, prior: np.ndarray | None, iterations: int = 200,
                per_stratum_rate: bool = False) -> dict:
    """votes (S, G) in {0,1}; strata (G,) in {0,1}; prior (G,) fixed per-gap P(z=1), or None to
    learn a base rate (single, or one per stratum with ``per_stratum_rate``; amendment 6a).
    Returns per-system/stratum sensitivity and specificity, the full posterior and the
    leave-one-system-out posteriors."""
    s_n, _ = votes.shape
    q = votes.mean(0)                                  # majority-style initialisation
    learn_prior = prior is None
    def learned_base(q):
        if per_stratum_rate:
            rates = np.array([np.clip(q[strata == t].mean() if (strata == t).any() else 0.5, 1e-4, 1 - 1e-4)
                              for t in (0, 1)])
            return float(q.mean()), np.log(rates / (1 - rates))[strata]
        rate = float(np.clip(q.mean(), 1e-4, 1 - 1e-4))
        return rate, np.log(rate / (1 - rate))

    for _ in range(iterations):
        rate, learned = learned_base(q)
        sens = np.zeros((s_n, 2))
        spec = np.zeros((s_n, 2))
        for t in (0, 1):
            m = strata == t
            w1, w0 = q[m], 1 - q[m]
            for s in range(s_n):
                sens[s, t] = (w1 @ votes[s, m] + 0.5) / (w1.sum() + 1.0)
                spec[s, t] = (w0 @ (1 - votes[s, m]) + 0.5) / (w0.sum() + 1.0)
        log_odds = _vote_log_odds(votes, strata, sens, spec)
        base = learned if learn_prior else _logit(prior)
        q = _sigmoid(base + log_odds.sum(0))
    rate, learned = learned_base(q)
    base = learned if learn_prior else _logit(prior)
    log_odds = _vote_log_odds(votes, strata, sens, spec)
    total = log_odds.sum(0)
    loo = np.array([_sigmoid(base + total - log_odds[s]) for s in range(s_n)])
    return {"sensitivity": sens, "specificity": spec, "base_rate": rate, "posterior": _sigmoid(base + total),
            "leave_one_out": loo}


def _vote_log_odds(votes, strata, sens, spec):
    se, sp = sens[:, strata], spec[:, strata]          # (S, G)
    return np.where(votes == 1, np.log(se / (1 - sp)), np.log((1 - se) / sp))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ------------------------------------------------------------------ fitting helpers
def fit_logistic_np(x: np.ndarray, y: np.ndarray, lam: float, iterations: int = 50) -> dict:
    """L2 logistic regression on standardised features by Newton steps (bias unpenalised)."""
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = np.column_stack([(x - mu) / sd, np.ones(len(x))])
    reg = np.full(z.shape[1], lam)
    reg[-1] = 0.0
    theta = np.zeros(z.shape[1])
    rate = np.clip(y.mean(), 1e-4, 1 - 1e-4)
    theta[-1] = math.log(rate / (1 - rate))
    n = len(y)
    for _ in range(iterations):
        p = _sigmoid(z @ theta)
        grad = z.T @ (p - y) / n + reg * theta
        if np.abs(grad).max() < 1e-9:
            break
        h = (z * (p * (1 - p) / n)[:, None]).T @ z + np.diag(reg + 1e-12)
        step = np.linalg.solve(h, grad)
        obj = lambda t: np.mean(np.logaddexp(0, z @ t) - y * (z @ t)) + 0.5 * np.sum(reg * t * t)
        f0, t = obj(theta), 1.0
        while obj(theta - t * step) > f0 - 1e-4 * t * float(grad @ step) and t > 1e-8:
            t *= 0.5
        theta = theta - t * step
    return {"w": theta[:-1], "b": theta[-1], "mu": mu, "sd": sd}


def predict_logistic_np(model: dict, x: np.ndarray) -> np.ndarray:
    return _sigmoid(((x - model["mu"]) / model["sd"]) @ model["w"] + model["b"])


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> dict:
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-9)
    z = (x - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - ym))
    return {"w": w, "b": ym, "mu": mu, "sd": sd}


def predict_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - model["mu"]) / model["sd"]) @ model["w"] + model["b"]
