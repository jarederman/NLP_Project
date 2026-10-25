from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass
class Candidate:
    condition: str
    rate: float
    boundaries: list[int]


def _rng(seed: int, doc_id: str, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{doc_id}|{label}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _count(rate: float, population: int) -> int:
    return max(1, round(rate * population)) if population else 0


def generate_candidates(gold: list[int], doc_id: str, seed: int, rates: list[float]) -> list[Candidate]:
    n = len(gold)
    positives = [i for i, y in enumerate(gold) if y]
    negatives = [i for i, y in enumerate(gold) if not y]
    output = [Candidate("gold", 0.0, gold.copy())]
    for rate in rates:
        label = f"{rate:g}"
        # Over: add a rate of the available false positions, capped by number of gold boundaries
        over = gold.copy()
        r = _rng(seed, doc_id, "over" + label)
        k = min(_count(rate, max(1, len(positives))), len(negatives))
        for i in r.sample(negatives, k):
            over[i] = 1
        output.append(Candidate("over", rate, over))

        under = gold.copy()
        r = _rng(seed, doc_id, "under" + label)
        k = min(_count(rate, len(positives)), len(positives))
        for i in r.sample(positives, k):
            under[i] = 0
        output.append(Candidate("under", rate, under))

        shifted = gold.copy()
        r = _rng(seed, doc_id, "shift" + label)
        k = min(_count(rate, len(positives)), len(positives))
        movable = positives.copy()
        r.shuffle(movable)
        moved = 0
        for old in movable:
            choices = [j for d in (1, -1, 2, -2, 3, -3)
                       if 0 <= (j := old + d) < n and shifted[j] == 0]
            if choices:
                shifted[old] = 0
                shifted[r.choice(choices)] = 1
                moved += 1
                if moved >= k:
                    break
        output.append(Candidate("shift", rate, shifted))

        mixed = gold.copy()
        r = _rng(seed, doc_id, "mixed" + label)
        k_del = min(_count(rate / 2, len(positives)), len(positives))
        k_add = min(_count(rate / 2, max(1, len(positives))), len(negatives))
        for i in r.sample(positives, k_del):
            mixed[i] = 0
        valid_neg = [i for i in negatives if mixed[i] == 0]
        for i in r.sample(valid_neg, min(k_add, len(valid_neg))):
            mixed[i] = 1
        output.append(Candidate("mixed", rate, mixed))
    return output
