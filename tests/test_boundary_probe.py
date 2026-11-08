import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.boundary_probe import (
    LEFT_TARGETS, PROBE_LAYERS, RIGHT_TARGETS, FlatCorpus, fit_logistic, gather_windows, insert_separators,
    lr_features, lr_specs, mask_specs, predict_logistic, run_lr_rows, run_mask_rows,
)
from src.boundary_probe_eval import select
from src.layout_data import iter_paragraph_flags

MASK, PERIOD, CLS, SEP, PAD = 103, 1012, 101, 102, 0
SPECIAL = dict(mask_id=MASK, period_id=PERIOD, cls_id=CLS, sep_id=SEP, pad_id=PAD)
# doc A tokens: dr . smith came . heading he left ; doc B: one two . three
DOC_A = ([10, PERIOD, 11, 12, PERIOD, 13, 14, 15], [(1, 2), (3, 3), (4, 5), (6, 6), (7, 7)])
DOC_B = ([20, 21, PERIOD, 22], [(1, 1), (2, 3)])


def reference_window(ids, slot, content, left_budget, right_budget):
    """The original Python construction of masked_punctuation_probabilities, generalised to any content."""
    s, e = slot
    left = ids[max(0, s - left_budget):s]
    right = ids[e:e + right_budget]
    return [CLS, *left, *content, *right, SEP]


def batch_windows(corpus, spec):
    src = torch.as_tensor(corpus.src)
    tensors = {k: torch.as_tensor(v) for k, v in spec.items()}
    ids, attention = gather_windows(torch, src, tensors, corpus.at)
    rows = []
    for row, mask in zip(ids.tolist(), attention.tolist()):
        rows.append([t for t, m in zip(row, mask) if m])
    return rows, ids, tensors


class GatherTests(unittest.TestCase):
    def setUp(self):
        self.corpus = FlatCorpus([DOC_A, DOC_B], **SPECIAL)

    def test_mask_rows_match_reference_construction(self):
        for budgets in ((2, 3), (126, 127), (1, 1)):
            spec = mask_specs(self.corpus, *budgets)
            rows, ids, tensors = batch_windows(self.corpus, spec)
            expected = [reference_window(doc[0], slot, [MASK], *budgets) for doc in (DOC_A, DOC_B) for slot in doc[1]]
            self.assertEqual(rows, expected)
            positions = tensors["tpos"].tolist()
            self.assertTrue(all(ids[i, p] == MASK for i, p in enumerate(positions)))

    def test_likelihood_ratio_rows(self):
        budgets = (2, 3)
        spec = lr_specs(self.corpus, *budgets)
        rows, ids, tensors = batch_windows(self.corpus, spec)
        slots = [(doc, slot) for doc in (DOC_A, DOC_B) for slot in doc[1]]
        for i, row in enumerate(rows):
            doc, slot = slots[spec["gap"][i]]
            ids_doc = doc[0]
            original = ids_doc[slot[0]:slot[1]]
            content = (original or [PERIOD]) if spec["cond"][i] == 0 else []
            self.assertEqual(row, reference_window(ids_doc, slot, content, *budgets))
            # the masked target holds exactly the source token
            self.assertEqual(int(ids[i, spec["tpos"][i]]), int(self.corpus.src[spec["source"][i]]))
        # gap 0 of doc A ("dr|.|smith") has one left token only -> only left target 0 exists
        targets = sorted(spec["target"][(spec["gap"] == 0) & (spec["cond"] == 0)].tolist())
        self.assertEqual(targets, [0, 2, 3, 4])

    def test_special_tokens_are_never_targets(self):
        ids, slots = insert_separators(*DOC_A, right_indices=[2, 3, 5, 6, 7],
                                       breaks=[False, False, True, False, False], sep_id=SEP)
        corpus = FlatCorpus([(ids, slots)], **SPECIAL)
        spec = lr_specs(corpus, 3, 3)
        self.assertFalse(np.isin(corpus.src[spec["source"]], [CLS, SEP, PAD, MASK]).any())

    def test_lr_features(self):
        spec = {"cond": np.array([0, 1, 0]), "gap": np.array([0, 0, 1]), "target": np.array([2, 2, 0])}
        features = lr_features(spec, np.array([-1.0, -3.0, -0.5]), 2)
        self.assertEqual(features.shape, (2, 2 * (LEFT_TARGETS + RIGHT_TARGETS) + 2))
        self.assertAlmostEqual(features[0, 2], 2.0)        # delta for target 2 of gap 0
        self.assertEqual(features[0, 5 + 2], 1.0)          # valid flag
        self.assertEqual(features[1, 0], 0.0)              # gap 1 lacks condition B -> missing
        self.assertAlmostEqual(features[0, -1], 2.0)       # right sum


class TinyModelTests(unittest.TestCase):
    """The extraction loops on a tiny random BERT (CPU): shapes, layer indexing, and agreement
    with an unbatched forward pass."""

    def test_extraction_loops(self):
        from transformers import BertConfig, BertForMaskedLM
        torch.manual_seed(0)
        model = BertForMaskedLM(BertConfig(vocab_size=1100, hidden_size=16, num_hidden_layers=12,
                                           num_attention_heads=2, intermediate_size=32)).eval()
        corpus = FlatCorpus([DOC_A, DOC_B], **SPECIAL)
        src = torch.as_tensor(corpus.src)
        spec = mask_specs(corpus, 2, 3)
        p, hidden = run_mask_rows(model, torch, src, spec, corpus.at, [PERIOD], batch_size=3)
        self.assertEqual(tuple(hidden.shape), (7, 16 * len(PROBE_LAYERS)))
        window = torch.tensor([reference_window(DOC_A[0], DOC_A[1][2], [MASK], 2, 3)])
        with torch.inference_mode():
            out = model.bert(input_ids=window, output_hidden_states=True)
            position = int(spec["tpos"][2])
            logits = model.cls(out.last_hidden_state[0, position]).float()
        self.assertAlmostEqual(float(p[2]), float(torch.softmax(logits, -1)[PERIOD]), places=4)
        self.assertTrue(torch.allclose(hidden[2, 16:].float(), out.hidden_states[12][0, position], atol=1e-2))
        l_spec = lr_specs(corpus, 2, 3)
        logp = run_lr_rows(model, torch, src, l_spec, corpus.at, MASK, batch_size=4)
        self.assertEqual(len(logp), len(l_spec["nl"]))
        self.assertTrue(bool((logp <= 0).all()))


class SeparatorTests(unittest.TestCase):
    def test_sep_goes_after_slot_and_before_right_context(self):
        # paragraph break after "came." (slot 4:5, right token "heading"=5) and after "heading" (insert at 6)
        ids, slots = insert_separators(*DOC_A, right_indices=[2, 3, 5, 6, 7],
                                       breaks=[False, False, True, True, False], sep_id=SEP)
        self.assertEqual(ids, [10, PERIOD, 11, 12, PERIOD, SEP, 13, SEP, 14, 15])
        self.assertEqual(slots, [(1, 2), (3, 3), (4, 5), (7, 7), (9, 9)])
        self.assertEqual(ids[slots[3][1]], SEP)            # "heading [MASK] [SEP] he"


class LogisticTests(unittest.TestCase):
    def test_recovers_probabilities(self):
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(20000, 3, generator=gen)
        true = torch.sigmoid(2.0 * x[:, 0] - 1.0)
        y = (torch.rand(20000, generator=gen) < true).float()
        model = fit_logistic(torch, x, y, 1e-6)
        p = predict_logistic(torch, model, x)
        self.assertLess(float((p - true).abs().mean()), 0.02)
        self.assertLess(model["max_grad"], 1e-4)


class LayoutTests(unittest.TestCase):
    def test_paragraph_flags(self):
        text = "\n".join([
            "# newdoc id = GUM_x_a", "# newpar", "# newpar_block = head (1 s)", "# sent_id = a-1",
            "1\tHello\t_\t_\t_\t_\t_\t_\t_\t_", "",
            "# sent_id = a-2", "1\tWorld\t_\t_\t_\t_\t_\t_\t_\t_", "2\tagain\t_\t_\t_\t_\t_\t_\t_\t_", "",
            "# newpar", "# sent_id = a-3", "1\tBye\t_\t_\t_\t_\t_\t_\t_\t_", "", ""])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.conllu"
            path.write_text(text)
            self.assertEqual(list(iter_paragraph_flags(path)), [("GUM_x_a", [(1, True), (2, False), (1, True)])])


class SelectionRuleTests(unittest.TestCase):
    def result(self, points, lows):
        return {"point": {f"raw/{c}": {"global_spearman": v} for c, v in points.items()},
                "paired_vs_baseline": {f"raw/{c}": {"global_spearman": [low, 1.0]} for c, low in lows.items()}}

    def test_simplest_within_tolerance(self):
        r = self.result({"platt": 0.3, "lr": 0.395, "probe": 0.40, "probe+lr": 0.45},
                        {"lr": 0.01, "probe": 0.02, "probe+lr": 0.05})
        self.assertEqual(select(r)["selected"], "probe+lr")
        r = self.result({"platt": 0.3, "lr": 0.445, "probe": 0.40, "probe+lr": 0.45},
                        {"lr": 0.01, "probe": 0.02, "probe+lr": 0.05})
        self.assertEqual(select(r)["selected"], "lr")

    def test_nothing_effective(self):
        r = self.result({"platt": 0.3, "lr": 0.5, "probe": 0.5, "probe+lr": 0.5},
                        {"lr": -0.01, "probe": -0.02, "probe+lr": -0.001})
        self.assertEqual(select(r)["selected"], "platt")


if __name__ == "__main__":
    unittest.main()
