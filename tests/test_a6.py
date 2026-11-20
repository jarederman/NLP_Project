import itertools
import math
import random
import unittest

import numpy as np

from src.a6_combine import _bcubed, _pk, _windowdiff, pick
from src.a6_data import safe_lower, word_spans
from src.a6_gpu import STRIDE, WINDOW, chunk_plan, corrupt, windows
from src.a6_signals import (KneserNeyTrigram, RULE_NAMES, dawid_skene, fit_ridge, hsmm_posterior, ngram_features,
                            predict_ridge, rule_features)


def brute_force_hsmm(p, prior_len, base_rate):
    n = len(p)
    odds = (p / (1 - p)) / (base_rate / (1 - base_rate))
    total, marg = 0.0, np.zeros(n)
    for bits in itertools.product((0, 1), repeat=n):
        cuts = [i for i, b in enumerate(bits) if b] + [n]
        lengths = np.diff([-1] + cuts)
        if lengths.max() >= len(prior_len):
            continue
        w = np.prod([prior_len[L] for L in lengths]) * np.prod([odds[i] for i, b in enumerate(bits) if b])
        total += w
        marg += w * np.array(bits)
    return marg / total


class HsmmTests(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        prior = np.array([0.0, 0.1, 0.3, 0.4, 0.2])
        for _ in range(5):
            p = rng.uniform(0.05, 0.95, size=6)
            np.testing.assert_allclose(hsmm_posterior(p, prior, 0.3), brute_force_hsmm(p, prior, 0.3), atol=1e-9)

    def test_empty_document(self):
        self.assertEqual(len(hsmm_posterior(np.zeros(0), np.array([0, 1.0]), 0.5)), 0)


class DawidSkeneTests(unittest.TestCase):
    def test_recovers_reliable_annotator(self):
        rng = np.random.default_rng(1)
        z = rng.random(3000) < 0.2
        good = np.where(rng.random(3000) < 0.98, z, ~z)
        noisy = [np.where(rng.random(3000) < 0.75, z, ~z) for _ in range(3)]
        votes = np.array([good] + noisy, dtype=float)
        fit = dawid_skene(votes, np.zeros(3000, dtype=int), None)
        self.assertGreater(fit["sensitivity"][0, 0], fit["sensitivity"][1, 0])
        self.assertGreater(fit["specificity"][0, 0], fit["specificity"][1, 0])
        self.assertGreater(np.mean((fit["posterior"] > 0.5) == z), 0.95)
        self.assertEqual(fit["leave_one_out"].shape, votes.shape)

    def test_leave_one_out_never_sees_the_left_out_system(self):
        rng = np.random.default_rng(2)
        z = rng.random(500) < 0.3
        votes = np.array([np.where(rng.random(500) < a, z, ~z) for a in (0.9, 0.8, 0.85, 0.7)], dtype=float)
        for prior in (None, np.full(500, 0.3)):
            base = dawid_skene(votes, np.zeros(500, dtype=int), prior)["leave_one_out"]
            for s in range(len(votes)):
                flipped = votes.copy()
                flipped[s] = rng.random(500) < 0.5          # (a full flip is uninformative-invariant in DS)
                other = dawid_skene(flipped, np.zeros(500, dtype=int), prior)["leave_one_out"]
                np.testing.assert_allclose(other[s], base[s])
                self.assertGreater(np.abs(np.delete(other, s, 0) - np.delete(base, s, 0)).max(), 1e-6)

    def test_fixed_prior_moves_posterior(self):
        votes = np.array([[1, 0], [1, 0], [0, 0]], dtype=float)
        low = dawid_skene(votes, np.zeros(2, dtype=int), np.array([0.01, 0.01]))["posterior"]
        high = dawid_skene(votes, np.zeros(2, dtype=int), np.array([0.99, 0.99]))["posterior"]
        self.assertTrue((high >= low).all())


class NgramTests(unittest.TestCase):
    def test_distribution_sums_to_one(self):
        lm = KneserNeyTrigram([["the", "cat", "sat"], ["the", "dog", "sat", "."], ["a", "cat", "ran"]])
        vocab = list(lm.cont1) + ["unseen"]
        for history in (("<s>", "<s>"), ("the", "cat"), ("x", "y")):
            total = sum(math.exp(lm.logprob(*history, w)) for w in vocab)
            self.assertLess(total, 1.0 + 1e-9)
            self.assertGreater(total, 0.5)

    def test_end_of_sentence_feature(self):
        lm = KneserNeyTrigram([["it", "ended", "."], ["it", "ended", "."], ["then", "more"]])
        text = "it ended. then more"
        f = ngram_features(lm, text, [e for _, e in word_spans(text)[:-1]])
        self.assertEqual(f.shape, (3, 4))
        self.assertGreater(f[1, 0], f[0, 0])         # "</s>" after "ended ." beats after "it"


class RuleTests(unittest.TestCase):
    def test_shape_and_casing_free(self):
        text = "See Dr. Smith at 3.15 p.m. today. 1. First item"
        offsets = [e for _, e in word_spans(text)[:-1]]
        a = rule_features(text, offsets)
        self.assertEqual(a.shape, (len(offsets), len(RULE_NAMES)))
        np.testing.assert_allclose(a, rule_features(text.swapcase(), offsets))


class TaggerHelperTests(unittest.TestCase):
    def test_windows_cover_every_token(self):
        for n in (1, 100, WINDOW, WINDOW + 1, 1000, 1537):
            for shift in (0, 37, STRIDE - 1):
                covered = np.zeros(n, dtype=bool)
                for a, b in windows(n, shift):
                    self.assertLessEqual(b - a, WINDOW)
                    covered[a:b] = True
                self.assertTrue(covered.all())

    def test_corrupt_keeps_word_alignment(self):
        text = 'He left. "Then she came." Done'
        spans = word_spans(text)
        new, new_spans = corrupt(text, spans, [0, 1, 0, 0, 1], random.Random(0))
        self.assertEqual(len(new_spans), len(spans))
        self.assertEqual([new[a:b] for a, b in new_spans], new.split(" "))

    def test_chunk_plan(self):
        class Doc:
            candidate_offsets = list(range(99))
        self.assertEqual(chunk_plan(Doc, 0), [(0, 64), (64, 100)])
        self.assertEqual(chunk_plan(Doc, 32), [(0, 32), (32, 96), (96, 100)])


class GoldQualityTests(unittest.TestCase):
    def test_perfect_segmentation(self):
        g = np.array([0, 1, 0, 0, 1, 0])
        self.assertEqual(_pk(g, g, 2), 0.0)
        self.assertEqual(_windowdiff(g, g, 2), 0.0)
        self.assertAlmostEqual(_bcubed(g, g), 1.0)

    def test_errors_lower_quality(self):
        g = np.array([0, 1, 0, 0, 1, 0])
        c = np.array([0, 0, 0, 0, 0, 0])
        self.assertGreater(_pk(g, c, 2), 0.0)
        self.assertLess(_bcubed(g, c), 1.0)


class SelectionTests(unittest.TestCase):
    def test_pick_prefers_simpler_within_tolerance(self):
        point = {"a": {"global_spearman": 0.50}, "b": {"global_spearman": 0.505}, "c": {"global_spearman": 0.60}}
        self.assertEqual(pick(point, ["a", "b", "c"]), "c")
        self.assertEqual(pick(point, ["a", "b"]), "a")
        self.assertEqual(pick(point, ["a", "b", "c"], ["b"]), "b")

    def test_ridge_recovers_linear_target(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(200, 3))
        y = x @ np.array([1.0, -2.0, 0.5]) + 3
        np.testing.assert_allclose(predict_ridge(fit_ridge(x, y, 1e-8), x), y, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
