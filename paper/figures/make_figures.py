"""Figures of the paper (F1 overview, F2 worked example, F3 score vs. true F1). Run from the project root:

    .venv/bin/python paper/figures/make_figures.py

F2 reads the frozen EWT-dev evidence of the final flat evaluator (amendment 12) and the gold of that split, which was
materialized after the one-shot test; F3 recomputes the EWT-dev item scores of B5, consensus and ours with the
amendment-13 code and checks them against results/a13. Nothing here feeds back into any method or score.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.a11_analysis import soft_f1_end  # noqa: E402

OUT = ROOT / "paper/figures"
EWTDEV = ROOT / "data/real_system_ewtdev"
EXAMPLE_DOC = "reviews-030875"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
INK, MUTED, ACCENT, GOLD, WARN = "#222222", "#8a8a8a", "#1f5fa8", "#2e7d32", "#c0392b"


# ------------------------------------------------------------------ F1: overview
def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, lw=0.8, size=7.5, style="round,pad=0.02,rounding_size=0.06",
        weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight,
            linespacing=1.15)


def arrow(ax, a, b, color=INK, lw=0.8, style="-|>", rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                                    connectionstyle=f"arc3,rad={rad}", mutation_scale=7))


def figure_overview() -> None:
    fig = plt.figure(figsize=(6.3, 2.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 33)
    ax.axis("off")

    # gold-isolation frame: everything inside never reads target-domain gold
    ax.add_patch(FancyBboxPatch((0.6, 0.6), 76.0, 31.8, boxstyle="round,pad=0,rounding_size=0.8", fc="none",
                                ec=MUTED, lw=0.8, ls=(0, (4, 2))))
    ax.text(1.6, 1.4, "reference-free: never reads target-domain gold", fontsize=6.6, color=MUTED, va="bottom",
            style="italic")

    box(ax, 2.0, 13.4, 13.6, 11.0, "raw document $X$\n(plain text, or\n+ paragraph breaks)\n$n$ whitespace gaps",
        fc="#f4f4f4", size=7)

    sources = ["regex rules (abbrev., numbering, quotes)",
               "Kneser–Ney trigram: $P(\\langle/s\\rangle\\mid$ left context$)$",
               "MaxEnt probe on frozen BERT [MASK]",
               "SaT-12l boundary prob. (zero-shot)",
               "ModernBERT boundary tagger (BCE)",
               "paragraph-break flag (layout only)"]
    y0, hh, gap = 26.4, 3.1, 0.62
    join = (48.2, 18.9)
    for k, t in enumerate(sources):
        y = y0 - k * (hh + gap)
        box(ax, 20, y, 25.5, hh, t, fc="#f4f4f4" if k == 5 else "#ffffff", size=6.6, ls="--" if k == 5 else "-")
        arrow(ax, (15.8, 18.9), (19.8, y + hh / 2), lw=0.55, color=MUTED)
        arrow(ax, (45.7, y + hh / 2), join, lw=0.55, color=MUTED, style="-")
    arrow(ax, join, (49.6, 18.9), lw=0.8)
    ax.text(32.75, 30.4, "evidence features $x_i$ at every gap", ha="center", fontsize=7, weight="bold")

    box(ax, 49.8, 14.35, 11.7, 9.1, "fusion\n$p_i = \\sigma(\\theta^\\top x_i)$\n$\\theta$ fitted on GUM\nfor quality + BCE",
        fc="#eaf1fa", ec=ACCENT, size=6.8)
    arrow(ax, (61.7, 18.9), (64.6, 18.9))
    box(ax, 64.8, 14.35, 10.6, 9.1, "boundary\nevidence\n$p_1,\\dots,p_n$", fc="#eaf1fa", ec=ACCENT, size=7)

    box(ax, 49.8, 1.5, 11.7, 6.4, "segmenters:\nPunkt, Stanza,\nWtP, SaT-3l", fc="#f4f4f4", size=6.3)
    arrow(ax, (8.8, 13.2), (8.8, 4.7), lw=0.55, color=MUTED, style="-")
    arrow(ax, (8.8, 4.7), (49.6, 4.7), lw=0.55, color=MUTED)
    arrow(ax, (61.7, 4.7), (64.6, 4.7))
    box(ax, 64.8, 1.5, 10.6, 6.4, "candidate\n$y\\in\\{0,1\\}^n$", fc="#ffffff", size=7)

    box(ax, 80.0, 9.0, 19.0, 10.6, "score $\\tilde Q(y,p)$\nnormalized Soft F1,\ndocument end counted",
        fc="#eaf1fa", ec=ACCENT, size=7, weight="bold")
    arrow(ax, (75.6, 18.9), (79.8, 16.8))
    arrow(ax, (75.6, 4.7), (79.8, 11.6))

    box(ax, 80.0, 22.8, 19.0, 8.2, "meta-evaluation (§4):\nscores frozen, then gold;\n$\\rho$, $\\tau$, error injection",
        fc="#ffffff", ec=MUTED, size=6.6, ls=(0, (4, 2)))
    arrow(ax, (89.5, 19.8), (89.5, 22.6), color=MUTED, lw=0.6)
    fig.savefig(OUT / "f1_overview.pdf")
    fig.savefig(OUT / "f1_overview.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ F2: worked example
def example_data():
    raw = {d["doc_id"]: d for d in map(json.loads, open(EWTDEV / "raw_test.jsonl"))}
    gold = {d["doc_id"]: set(d["offsets"]) for d in map(json.loads, open(EWTDEV / "gold_test.jsonl"))}
    cands = {}
    for c in map(json.loads, open(EWTDEV / "a12/candidates.jsonl")):
        if c["doc_id"] == EXAMPLE_DOC:
            cands[c["system"]] = set(c["candidate_boundary_offsets"])
    ev = np.load(EWTDEV / "a12/evidence_frozen.npz")
    pos = 0
    for did in ev["doc_ids"]:
        text = raw[did]["text"]
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if did == EXAMPLE_DOC:
            break
        pos += len(spans) - 1
    offsets = [e for _, e in spans[:-1]]
    p = ev["final (flat)"][pos:pos + len(offsets)]
    words = [text[s:e] for s, e in spans]
    return words, offsets, p, gold[EXAMPLE_DOC], cands


def f1(a: set, b: set) -> float:
    return 1.0 if not a and not b else 2 * len(a & b) / (len(a) + len(b))


def figure_example() -> dict:
    words, offsets, p, g, cands = example_data()
    n = len(offsets)
    shown = [("sat", "SaT"), ("punkt", "Punkt"), ("wtp", "WtP")]
    rows = {k: np.array([int(o in cands[k]) for o in offsets]) for k, _ in shown}
    gold_row = np.array([int(o in g) for o in offsets])
    q = {k: float(soft_f1_end(rows[k][None], p)[0]) for k, _ in shown}
    tf = {k: f1(cands[k], g) for k, _ in shown}

    fig = plt.figure(figsize=(3.25, 2.45))
    ax = fig.add_axes([0.13, 0.57, 0.62, 0.40])
    x = np.arange(n)
    colors = [GOLD if b else "#b9c6d6" for b in gold_row]
    ax.bar(x, p, width=0.72, color=colors, lw=0)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("$p_i$ (ours, flat)", fontsize=7, labelpad=1)
    ax.tick_params(axis="both", labelsize=6.5, length=2, pad=1)
    ax.set_xticks(x)
    ax.set_xticklabels(words[:n], rotation=90, fontsize=5.6)
    ax.tick_params(axis="x", length=0, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(n - 0.5, 1.02, "green bar = gold boundary", ha="right", va="top", fontsize=6, color=GOLD)

    rx = fig.add_axes([0.13, 0.0, 0.62, 0.36], sharex=ax)
    rx.axis("off")
    for r, (k, name) in enumerate(shown):
        yy = 2 - r
        for i in range(n):
            cut = rows[k][i]
            wrong = cut != gold_row[i]
            if cut or wrong:
                rx.plot(i, yy, marker="|" if cut else "x", ms=7 if cut else 4.5, mew=1.4 if cut else 1.0,
                        color=WARN if wrong else INK)
        rx.plot([-0.5, n - 0.5], [yy, yy], color="#dddddd", lw=0.5, zorder=0)
        rx.text(-0.9, yy, name, ha="right", va="center", fontsize=7)
        rx.text(n + 0.4, yy, f"{q[k]:.2f}    {tf[k]:.2f}", ha="left", va="center", fontsize=7)
    rx.text(n + 0.4, 2.55, "$\\tilde Q$       F1", ha="left", va="bottom", fontsize=7)
    rx.text(n / 2 - 0.5, -0.75, "| = cut;  red | = false split;  red × = missed boundary", ha="center",
            va="center", fontsize=6, color=INK)
    rx.set_ylim(-1.0, 2.9)
    fig.savefig(OUT / "f2_example.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "f2_example.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {"doc": EXAMPLE_DOC, "text": " ".join(words), "scores": q, "true_f1": tf,
            "gold_score": float(soft_f1_end(gold_row[None], p)[0]),
            "p": {words[i]: round(float(p[i]), 3) for i in np.flatnonzero(p > 0.1)}}


# ------------------------------------------------------------------ F3: score vs. true F1 on EWT dev (flat)
def figure_scatter() -> dict:
    from scipy.stats import spearmanr

    from src import a13_baselines as A
    from src.a6_data import SYSTEMS, logo_folds
    from src.quality_agreement import tie_exact

    c = A.load_corpus("ewtdev")
    gum = A.load_corpus("gum")
    satp = A.sat_platt(gum, c, logo_folds(gum.docs))
    cm = c.scen["flat"]["cmat"]
    f1 = c.scen["flat"]["f1"]
    panels = [("B6 consensus", c.scen["flat"]["M0"]),
              ("B5 SaT-12l-sm prob., our scoring", np.concatenate([soft_f1_end(cm[:, sl], satp[sl]) for sl in c.sl])),
              ("Ours (flat)", np.concatenate([soft_f1_end(cm[:, sl], c.ev["final_flat"][sl]) for sl in c.sl]))]
    rho = {name: float(spearmanr(tie_exact(v), tie_exact(f1))[0]) for name, v in panels}
    expected = json.loads((ROOT / "results/a13/report.json").read_text())["corpora"]["ewtdev"]["scenarios"]["flat"]["table"]
    for (name, _), key in zip(panels, ("Consensus M0", "SaT-12l-sm probabilities", "Ours")):
        if abs(rho[name] - expected[key]["global_spearman"][0]) > 1e-6:
            raise RuntimeError(f"F3 scores do not reproduce results/a13 for {name}")
    colors = {"punkt": "#c0392b", "sat": "#1f5fa8", "stanza": "#2e7d32", "wtp": "#b8860b"}
    names = {"punkt": "Punkt", "sat": "SaT", "stanza": "Stanza", "wtp": "WtP"}
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.95), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (name, v) in zip(axes, panels):
        for k, sysname in enumerate(SYSTEMS):
            idx = np.arange(k, len(v), len(SYSTEMS))
            jitter = rng.uniform(-0.006, 0.006, len(idx))
            ax.scatter(f1[idx] + jitter, v[idx], s=2.2, lw=0, alpha=0.55, color=colors[sysname],
                       label=names[sysname], rasterized=True)
        ax.set_title(f"{name}\n$\\rho$ = {rho[name]:.3f}", fontsize=7.5, pad=3)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6.5, length=2, pad=1)
        ax.set_xlabel("true F1", fontsize=7, labelpad=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("score", fontsize=7, labelpad=1)
    leg = axes[0].legend(loc="lower right", fontsize=6, markerscale=3, frameon=False, handletextpad=0.1,
                         borderaxespad=0.1, labelspacing=0.2)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.17, top=0.83, wspace=0.12)
    fig.savefig(OUT / "f3_scatter.pdf", dpi=300)
    fig.savefig(OUT / "f3_scatter.png", dpi=220)
    plt.close(fig)
    return rho


if __name__ == "__main__":
    figure_overview()
    info = figure_example()
    (OUT / "f2_example.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print("F3 rho:", figure_scatter())
