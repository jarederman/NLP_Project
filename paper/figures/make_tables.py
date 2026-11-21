"""Appendix tables of the paper, generated from the result files so that no number is typed by hand.

    python3 paper/figures/make_tables.py        # writes paper/tables/*.tex

Sources: results/a13/classic.json (all nine baselines + ours: agreement, ranking, paired differences, sensitivity),
results/a13/report.json (documents per perturbation type).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/tables"
CLASSIC = json.loads((ROOT / "results/a13/classic.json").read_text())["corpora"]
REPORT = json.loads((ROOT / "results/a13/report.json").read_text())["corpora"]

CORPORA = {"gum": "GUM (D)", "gentle": "GENTLE (R)", "lines": "LinES test (R)", "ewt": "EWT test (R)",
           "ewtdev": "EWT dev (T)", "linesx": "LinES tr+dev (T)"}
METHODS = [("Punctuation rule", "B1 punctuation rule"), ("Length prior", "B2 length prior"),
           ("Masked LM (PuncEv)", "B3 masked LM (PuncEv)"),
           ("SaT-12l-sm output as reference", "B4 SaT-12l-sm segm.\\ as ref."),
           ("SaT-12l-sm probabilities", "B5 SaT-12l-sm prob., our scoring"),
           ("Consensus M0", "B6 consensus"), ("DocAgree", "B7 DocAgree"), ("GPT-2 perplexity", "B8 GPT-2 perplexity"),
           ("Dawid-Skene", "B9 Dawid--Skene"), ("Ours", "\\textbf{Ours}")]
SENS_METHODS = [("Punctuation rule", "B1 rule"), ("Length prior", "B2 length prior"), ("Masked LM (PuncEv)", "B3 masked LM"),
                ("SaT-12l-sm output as reference", "B4 SaT segm."), ("SaT-12l-sm probabilities", "B5 SaT prob."),
                ("Consensus", "B6 consensus"), ("DocAgree", "B7 DocAgree"), ("GPT-2 perplexity", "B8 GPT-2"),
                ("Dawid-Skene", "B9 Dawid--Skene"), ("Ours (flat)", "\\textbf{Ours (flat)}"),
                ("Ours (layout)", "\\textbf{Ours (layout)}")]
TYPES = [("merge_terminal", "M-term"), ("merge_other_par", "M-par"), ("merge_other_inpar", "M-inpar"),
         ("split_terminal", "S-abbr"), ("split_punct", "S-punct"), ("split_nopunct", "S-none"), ("shift", "Shift"),
         ("control", "Control")]


def num(x: float, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("-", "$-$") if sign or x < 0 else s


def ci(v, sign: bool = False) -> str:
    """value with a 95% CI underneath (two-line cell via \\shortstack)."""
    p, lo, hi = v
    short = lambda x: num(x, sign)
    return f"\\shortstack{{{short(p)}\\\\{{\\tiny[{short(lo)}, {short(hi)}]}}}}"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)


def agreement(scen: str) -> str:
    cols = [c for c in CORPORA if scen in CLASSIC[c]["scenarios"]]
    head = " & ".join(CORPORA[c] for c in cols)
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", f"Evaluator & {head} \\\\", "\\midrule"]
    for key, label in METHODS:
        cells = []
        for c in cols:
            row = CLASSIC[c]["scenarios"][scen]["table"].get(key)
            cells.append(ci(row["global_spearman"]) if row else "--")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def within(scen: str) -> str:
    cols = [c for c in CORPORA if scen in CLASSIC[c]["scenarios"]]
    head = " & ".join(CORPORA[c] for c in cols)
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", f"Evaluator & {head} \\\\", "\\midrule"]
    for key, label in METHODS:
        cells = []
        for c in cols:
            sc = CLASSIC[c]["scenarios"][scen]
            row = sc["table"].get(key)
            rk = sc["ranking"].get(key) or REPORT[c]["scenarios"][scen]["ranking"].get(key)
            if not row:
                cells.append("--")
                continue
            tau = rk["genre_tau"][0] if rk and rk.get("genre_tau") else float("nan")
            cells.append(f"{row['within_doc_pairwise']:.2f} / {num(tau) if not math.isnan(tau) else '--'}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def diffs(scen: str) -> str:
    cols = [c for c in CORPORA if scen in CLASSIC[c]["scenarios"]]
    head = " & ".join(CORPORA[c] for c in cols)
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", f"Ours $-$ & {head} \\\\", "\\midrule"]
    for key, label in METHODS[:-1]:
        cells = []
        for c in cols:
            om = CLASSIC[c]["scenarios"][scen]["ours_minus"].get(f"Ours - {key}")
            cells.append(ci(om, sign=True) if om else "--")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def sensitivity(corpus: str) -> str:
    sens = CLASSIC[corpus]["sensitivity_k1"]
    docs = REPORT[corpus]["sensitivity"]["k=1"]["documents_with_type"]
    head = " & ".join(f"{lab}" for _, lab in TYPES)
    ndoc = " & ".join(f"{{\\tiny $n$={docs.get(t, 0)}}}" for t, _ in TYPES)
    lines = ["\\begin{tabular}{l" + "c" * len(TYPES) + "}", "\\toprule", f"Evaluator & {head} \\\\",
             f" & {ndoc} \\\\", "\\midrule"]
    for key, label in SENS_METHODS:
        if key not in sens:
            continue
        cells = []
        for t, _ in TYPES:
            v = sens[key].get(t)
            if v is None or (isinstance(v[0], float) and math.isnan(v[0])):
                cells.append("--")
                continue
            star = "$^*$" if sens[key]["_detected"].get(t) and t != "control" else ""
            cells.append(f"{v[0]:.2f}{star}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    write("c_agreement_flat.tex", agreement("flat"))
    write("c_agreement_layout.tex", agreement("nl"))
    write("c_within_flat.tex", within("flat"))
    write("c_within_layout.tex", within("nl"))
    write("c_diffs_flat.tex", diffs("flat"))
    write("c_diffs_layout.tex", diffs("nl"))
    for c in ("gum", "gentle", "ewtdev", "linesx"):
        write(f"d_sensitivity_{c}.tex", sensitivity(c))
    print("\n".join(sorted(p.name for p in OUT.glob("*.tex"))))
