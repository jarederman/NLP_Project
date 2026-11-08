import unittest

from src.conditioned_bert import render_candidate

P = 9  # period token id
# tokens: dr(0) .(1) smith(2) came(3) .(4) heading(5) he(6) left(7)
FULL = [10, P, 11, 12, P, 13, 14, 15]
# gaps: after "dr."  (replace slot 1:2), after "smith" (insert at 3), after "came." (replace 4:5),
#       after "heading" (insert at 6), after "he" (insert at 7)
PLANS = [("replace_terminal_punctuation", 0, 1, 2), ("insert_at_gap", 0, 3, 3),
         ("replace_terminal_punctuation", 0, 4, 5), ("insert_at_gap", 0, 6, 6), ("insert_at_gap", 0, 7, 7)]
VECTOR = (0, 0, 1, 1, 0)   # candidate splits after "came." and after "heading"


class RenderCandidateTests(unittest.TestCase):
    def test_raw_reproduces_original_slots(self):
        ids, slots = render_candidate(FULL, PLANS, None, "raw", P)
        self.assertEqual(ids, FULL)
        self.assertEqual(slots, [(p[2], p[3]) for p in PLANS])

    def test_raw_plus_candidate_adds_missing_marks_only(self):
        ids, slots = render_candidate(FULL, PLANS, VECTOR, "raw_plus_candidate", P)
        self.assertEqual(ids, [10, P, 11, 12, P, 13, P, 14, 15])   # "." added after "heading"
        self.assertEqual(ids[slots[3][0]:slots[3][1]], [P])
        self.assertEqual(ids[slots[0][0]:slots[0][1]], [P])       # abbreviation period kept

    def test_candidate_only_follows_candidate(self):
        ids, slots = render_candidate(FULL, PLANS, VECTOR, "candidate_only", P)
        self.assertEqual(ids, [10, 11, 12, P, 13, P, 14, 15])      # "dr." -> "dr", heading gets "."
        self.assertEqual(slots[0], (1, 1))
        left, right = ids[:slots[2][0]], ids[slots[2][1]:]
        self.assertEqual(left + right, [10, 11, 12, 13, P, 14, 15])  # query gap itself is masked out

    def test_query_is_identical_across_modes(self):
        """Masking a gap removes whatever its slot holds, so the query never sees its own mark."""
        for mode in ("raw_plus_candidate", "candidate_only"):
            ids, slots = render_candidate(FULL, PLANS, (1, 1, 1, 1, 1), mode, P)
            for g, (a, b) in enumerate(slots):
                self.assertTrue(all(t == P for t in ids[a:b]))


if __name__ == "__main__":
    unittest.main()
