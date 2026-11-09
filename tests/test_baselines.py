import unittest

from src.baselines import boundary_f1


class BoundaryF1Tests(unittest.TestCase):
    def test_boundary_f1(self):
        self.assertEqual(boundary_f1({1, 2}, {1, 2}), 1.0)
        self.assertEqual(boundary_f1(set(), set()), 1.0)          # single-sentence doc, nobody splits
        self.assertEqual(boundary_f1({1}, set()), 0.0)
        self.assertAlmostEqual(boundary_f1({1, 2, 3}, {1}), 0.5)


if __name__ == "__main__":
    unittest.main()


class DecompositionTests(unittest.TestCase):
    def test_decompose_separates_document_and_system_effects(self):
        import numpy as np
        from src.baseline_decomposition import decompose, gold_variance_share
        keys = np.array(["a", "a", "b", "b", "c", "c"])
        gold = np.array([1.0, 0.5, 0.75, 0.25, 0.5, 0.0])      # documents differ; first system better (exact in binary)
        difficulty_only = np.array([1.0, 1.0, 0.5, 0.5, 0.1, 0.1])
        system_only = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(decompose(difficulty_only, gold, keys)["between_documents"], 1.0)
        self.assertAlmostEqual(decompose(system_only, gold, keys)["within_documents"], 1.0)
        self.assertAlmostEqual(gold_variance_share(gold, keys), 0.4)   # between-doc SS 0.25 / total SS 0.625
