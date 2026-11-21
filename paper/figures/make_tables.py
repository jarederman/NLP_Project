"""Appendix tables of the paper, generated from the result files so that no number is typed by hand.

    python3 paper/figures/make_tables.py        # writes paper/tables/*.tex

Sources: results/a13/classic.json (all nine baselines + ours: agreement, ranking, paired differences, sensitivity),
results/a13/report.json (rankings of the amendment-13 baselines, documents per perturbation type).

Bold marks the best value of a column and every value close to it: within RHO_TOL for correlations and pairwise
accuracies, within RATE_TOL for detection rates. Detection rates are compared among the flat-text evaluators only
(the paragraphs-given row is a different input condition), and the control column is never bold (0.50 is the target
for every evaluator). Paired differences are not ranked, so they are never bold.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/tables"
CLASSIC = json.loads((ROOT / "results/a13/classic.json").read_text())["corpora"]
REPORT = json.loads((ROOT / "results/a13/report.json").read_text())["corpora"]
RHO_TOL, RATE_TOL = 0.01, 0.02

CORPORA = {"gum": "GUM (D)", "gentle": "GENTLE (R)", "lines": "LinES test (R)", "ewt": "EWT test (R)",
           "ewtdev": "EWT dev (T)", "linesx": "LinES tr+dev (T)"}
METHODS = [("Punctuation rule", "B1 punctuation rule"), ("Length prior", "B2 length prior"),
           ("Masked LM (PuncEv)", "B3 masked LM (PuncEv)"),
           ("SaT-12l-sm output as reference", "B4 SaT-12l-sm as reference"),
           ("SaT-12l-sm probabilities", "B5 SaT-12l-sm probabilities"),
           ("Consensus M0", "B6 consensus"), ("DocAgree", "B7 DocAgree"), ("GPT-2 perplexity", "B8 GPT-2 perplexity"),
           ("Dawid-Skene", "B9 Dawid--Skene"), ("Ours", "Ours")]
SENS_METHODS = [("Punctuation rule", "B1 rule"), ("Length prior", "B2 length prior"), ("Masked LM (PuncEv)", "B3 masked LM"),
                ("SaT-12l-sm output as reference", "B4 SaT as ref."), ("SaT-12l-sm probabilities", "B5 SaT prob."),
                ("Consensus", "B6 consensus"), ("DocAgree", "B7 DocAgree"), ("GPT-2 perplexity", "B8 GPT-2"),
                ("Dawid-Skene", "B9 Dawid--Skene"), ("Ours (flat)", "Ours (flat)"),
                ("Ours (layout)", "Ours (paragraphs given)")]
NOT_COMPARED = {"Ours (layout)"}
TYPES = [("merge_terminal", "M-term"), ("merge_other_par", "M-par"), ("merge_other_inpar", "M-inpar"),
         ("split_terminal", "S-abbr"), ("split_punct", "S-punct"), ("split_nopunct", "S-none"), ("shift", "Shift"),
         ("control", "Control")]


def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def num(x: float, sign: bool = False, digits: int = 3) -> str:
    if isnan(x):
        return "--"
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace("-", "$-$")


def bold_if(text: str, on: bool) -> str:
    return f"\\textbf{{{text}}}" if on else text


def near_best(values: list, tol: float) -> list[bool]:
    """True for the best (largest) value and every value within ``tol`` of it; NaN/None never."""
    finite = [v for v in values if not isnan(v)]
    if not finite:
        return [False] * len(values)
    top = max(finite)
    return [not isnan(v) and v >= top - tol - 1e-12 for v in values]


def ci_cell(v, sign: bool = False, bold: bool = False) -> str:
    """Point estimate (optionally bold) with its 95% CI underneath."""
    p, lo, hi = v
    return f"\\shortstack{{{bold_if(num(p, sign), bold)}\\\\{{\\tiny[{num(lo, sign)}, {num(hi, sign)}]}}}}"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)


def header(cols: list[str], first: str = "Evaluator") -> list[str]:
    return ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
            f"{first} & " + " & ".join(CORPORA[c] for c in cols) + " \\\\", "\\midrule"]


def scenario_cols(scen: str) -> list[str]:
    return [c for c in CORPORA if scen in CLASSIC[c]["scenarios"]]


def agreement(scen: str) -> str:
    cols = scenario_cols(scen)
    rows = {key: [CLASSIC[c]["scenarios"][scen]["table"].get(key) for c in cols] for key, _ in METHODS}
    best = {c: near_best([(rows[k][j] or {}).get("global_spearman", [float("nan")])[0] for k, _ in METHODS], RHO_TOL)
            for j, c in enumerate(cols)}
    lines = header(cols)
    for i, (key, label) in enumerate(METHODS):
        cells = [ci_cell(r["global_spearman"], bold=best[c][i]) if r else "--" for c, r in zip(cols, rows[key])]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def within(scen: str) -> str:
    """Within-document pairwise accuracy / genre-mean Kendall tau of the segmenter ranking."""
    cols = scenario_cols(scen)
    pw, tau = {}, {}
    for key, _ in METHODS:
        for c in cols:
            sc = CLASSIC[c]["scenarios"][scen]
            row = sc["table"].get(key)
            rk = sc["ranking"].get(key) or REPORT[c]["scenarios"][scen]["ranking"].get(key)
            pw[key, c] = row["within_doc_pairwise"] if row else float("nan")
            tau[key, c] = rk["genre_tau"][0] if rk and rk.get("genre_tau") else float("nan")
    bpw = {c: near_best([pw[k, c] for k, _ in METHODS], RHO_TOL) for c in cols}
    btau = {c: near_best([tau[k, c] for k, _ in METHODS], RHO_TOL) for c in cols}
    lines = header(cols)
    for i, (key, label) in enumerate(METHODS):
        cells = [f"{bold_if(num(pw[key, c], digits=2), bpw[c][i])} / {bold_if(num(tau[key, c], digits=2), btau[c][i])}"
                 for c in cols]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def diffs(scen: str) -> str:
    cols = scenario_cols(scen)
    lines = header(cols, first="Ours $-$")
    for key, label in METHODS[:-1]:
        cells = []
        for c in cols:
            om = CLASSIC[c]["scenarios"][scen]["ours_minus"].get(f"Ours - {key}")
            cells.append(ci_cell(om, sign=True) if om else "--")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def sensitivity(corpus: str) -> str:
    sens = CLASSIC[corpus]["sensitivity_k1"]
    docs = REPORT[corpus]["sensitivity"]["k=1"]["documents_with_type"]
    present = [(k, lab) for k, lab in SENS_METHODS if k in sens]
    value = {(k, t): sens[k].get(t, [float("nan")])[0] for k, _ in present for t, _ in TYPES}
    compared = [k for k, _ in present if k not in NOT_COMPARED]
    best = {}
    for t, _ in TYPES:
        if t == "control":
            continue
        flags = near_best([value[k, t] for k in compared], RATE_TOL)
        best.update({(k, t): f for k, f in zip(compared, flags)})
    lines = ["\\begin{tabular}{l" + "c" * len(TYPES) + "}", "\\toprule",
             "Evaluator & " + " & ".join(lab for _, lab in TYPES) + " \\\\",
             " & " + " & ".join(f"{{\\tiny $n$={docs.get(t, 0)}}}" for t, _ in TYPES) + " \\\\", "\\midrule"]
    for key, label in present:
        if key in NOT_COMPARED:
            lines.append("\\midrule")
        cells = []
        for t, _ in TYPES:
            v = value[key, t]
            if isnan(v):
                cells.append("--")
                continue
            star = "$^*$" if sens[key]["_detected"].get(t) and t != "control" else ""
            cells.append(bold_if(f"{v:.2f}", best.get((key, t), False)) + star)
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
