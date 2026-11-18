from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.surface_data import extract_gold_boundaries, load_surface_documents


FIXTURE = """# newdoc id = GUM_test_one
# sent_id = leak-1
# text = Mr. Smith can't pay 3.14.
1\tMr.\tMr.\tPROPN\t_\t_\t2\tnmod\t_\t_
2\tSmith\tSmith\tPROPN\t_\t_\t4\tnsubj\t_\t_
3-4\tcan't\t_\t_\t_\t_\t_\t_\t_\t_
3\tca\tcan\tAUX\t_\t_\t4\taux\t_\t_
4\tn't\tnot\tPART\t_\t_\t0\troot\t_\t_
5\tpay\tpay\tVERB\t_\t_\t4\txcomp\t_\t_
6\t3.14\t3.14\tNUM\t_\t_\t5\tobj\t_\tSpaceAfter=No
7\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_

# sent_id = leak-2
# text = He moved to the U.S. He works there.
1\tHe\the\tPRON\t_\t_\t2\tnsubj\t_\t_
2\tmoved\tmove\tVERB\t_\t_\t0\troot\t_\t_
3\tto\tto\tADP\t_\t_\t5\tcase\t_\t_
4\tthe\tthe\tDET\t_\t_\t5\tdet\t_\t_
5\tU.S.\tU.S.\tPROPN\t_\t_\t2\tobl\t_\t_
6\tHe\the\tPRON\t_\t_\t7\tnsubj\t_\t_
7\tworks\twork\tVERB\t_\t_\t0\troot\t_\t_
8\tthere\tthere\tADV\t_\t_\t7\tadvmod\t_\tSpaceAfter=No
9\t.\t.\tPUNCT\t_\t_\t7\tpunct\t_\t_

"""


class SurfaceDataTest(unittest.TestCase):
    def test_surface_preserves_punctuation_and_mwt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.conllu"
            path.write_text(FIXTURE, encoding="utf-8")
            docs = load_surface_documents(path, "dev")
            self.assertEqual(len(docs), 1)
            doc = docs[0]
            self.assertEqual(
                doc.text,
                "Mr. Smith can't pay 3.14. He moved to the U.S. He works there.",
            )
            self.assertEqual(doc.tokens.count("can't"), 1)
            gold = extract_gold_boundaries(path, "dev", docs)[0]
            self.assertEqual(gold.sentence_count, 2)
            self.assertEqual(gold.boundary_offsets, [len("Mr. Smith can't pay 3.14.")])
            self.assertEqual(sum(gold.labels), 1)

    def test_sentence_comments_do_not_change_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.conllu"
            b = Path(tmp) / "b.conllu"
            a.write_text(FIXTURE, encoding="utf-8")
            b.write_text(
                FIXTURE.replace("# text = Mr. Smith can't pay 3.14.", "# text = deliberately wrong")
                .replace("# text = He moved to the U.S. He works there.", "# text = also wrong"),
                encoding="utf-8",
            )
            doc_a = load_surface_documents(a, "dev")[0]
            doc_b = load_surface_documents(b, "dev")[0]
            self.assertEqual(doc_a.text, doc_b.text)
            self.assertEqual(doc_a.text_sha256, doc_b.text_sha256)

    def test_sentence_blank_lines_do_not_change_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.conllu"
            b = Path(tmp) / "b.conllu"
            a.write_text(FIXTURE, encoding="utf-8")
            b.write_text(FIXTURE.replace("\n\n# sent_id", "\n# sent_id"), encoding="utf-8")
            self.assertEqual(
                load_surface_documents(a, "dev")[0].text,
                load_surface_documents(b, "dev")[0].text,
            )


if __name__ == "__main__":
    unittest.main()
