# Paper outline (v1, 2026-11-17)

Working title: **How Far Can Reference-Free Evaluation of Sentence Segmentation Go? Boundary Evidence, Blind Spots, and Cross-Domain Agreement**
(alternative, matching the proposal: *Cross-Domain Reference-Free Quality Estimation for Sentence Segmentation*)

Authors: Weifeng Jiang, Yanhan Zhu (NYU, CSCI-UA.0469). Format: ACL long-paper template, main text ≈ 8 pages
(+ unlimited references and appendix; **page limit to be confirmed against the course page**).

Status labels used throughout: **D** = development (GUM train, leave-one-genre-out cross-fitting);
**T** = pre-registered one-shot test (scores frozen before Gold was materialized); **P** = post-test analysis
(computed after the test, analysis only). Every number below was checked against `results/` on 2026-11-17.

---

## 0. What the project was required to answer, and where the paper answers it

The proposal / 11-02 progress report fixed the project's question, baselines and protocol. The paper must answer
each item explicitly (Section 1 states the answers; Section 5.1 gives the evidence).

| Requirement (progress report, 2026-11-02) | Answer in the paper | Where |
|---|---|---|
| RQ0: can a frozen BERT + Platt + Soft F1 evaluator select a splitter with lower Macro Selection Regret than Always-SaT? | **No.** GUM train CV: selected − Always-SaT = −0.026 [−0.037, −0.021]; Gold-win rate 0.16. With the final methods (A8.4) no single-system scorer beats Always-SaT on any corpus; only the multi-system QE regressor M3 does, on GENTLE per-document routing (−0.043 [−0.079, −0.015], P). | §5.1, Table 6 |
| Required baselines: Direct Internal Splitter (DIS) and Always-SaT | DIS (evidence thresholded at 0.5) has F1 0.831, below all four systems. Always-SaT is near-oracle: its corpus-level Macro regret is 0 on GUM, GENTLE and LinES. This is *why* RQ0 fails and why the goal was redefined. | §5.1 |
| Held-out setting: develop on GUM, freeze, evaluate once on EWT test | Done (amendment 4). Frozen Probe vs PuncEv: 0.411 vs 0.338 global Spearman, +0.074 [+0.023, +0.126]. | §5.3, Table 2 |
| Supporting outcomes: domain-level ranking, within-document pairwise accuracy | Reported for every method (cell Spearman, within-document pairwise). | Table 2 (appendix full) |
| Extension: GENTLE | Done, plus LinES (amendment 6, one-shot). | §5.4 |
| Course: related work + bibliography, honest limitations, presentation | §2 (all citations verified), §8, slides after the paper. | — |
| Course: team size / contribution statement | **To confirm with the instructor** (course normally expects 3–4 people; we are two). Add a contribution statement if required. | end matter |

---

## Abstract (draft, ~200 words)

Sentence segmentation is usually evaluated against gold boundaries, which do not exist for a new domain. We ask whether
the output of a segmenter can be scored *without* a reference, using language-model evidence at each candidate
boundary. Our original goal was operational: pick, per domain, a splitter (Punkt, Stanza, WtP, SaT) that beats always
using the strongest one (SaT). This fails: SaT is near-oracle, and our evidence systematically misses true boundaries
that carry no terminal punctuation. We therefore redefine the target as cross-document quality agreement (global
Spearman between score and gold F1) and study what limits it under a pre-registered, gold-isolated protocol with
one-shot tests on three held-out corpora. Three findings stand out. (1) The scoring formula matters little once the
evidence is strong; a hard pseudo-reference is statistically equivalent to soft scoring, and validity tracks evidence
accuracy almost linearly. (2) Every evidence-based scorer has the same blind spots — merged boundaries without
punctuation and false splits after abbreviations — which we quantify with ~110k controlled perturbations and which
explain both a systematic overrating of Punkt and the failure on LinES. (3) Combining evidence sources (a frozen
probe, zero-shot SaT, a fine-tuned ModernBERT, rules and a Kneser–Ney model) improves robustly over our earlier
evaluator on GENTLE (+0.21 global Spearman), with mixed evidence against multi-system consensus and no gain on LinES.

---

## 1 Introduction (≈1 page)

1. Problem: segmentation errors propagate downstream; gold boundaries are unavailable in a new domain; existing
   evaluation (boundary F1, Pk, WindowDiff) needs a reference. Reference-free quality estimation exists for MT but not
   for segmentation.
2. Original question (RQ0, from the proposal): reference-free *selection* of a splitter, measured by Macro Selection
   Regret against Always-SaT.
3. What happened, stated up front: RQ0 fails, and *why* (Always-SaT near-oracle; evidence blind to unpunctuated
   boundaries). This motivates the reframing, decided and pre-registered before computing the new metric (amendment 2).
4. Redefined questions:
   - **RQ1** Can a reference-free score rank segmentation quality *across documents and domains*?
   - **RQ2** What limits its validity — the scoring formula or the boundary evidence?
   - **RQ3** How does single-system evidence compare with multi-system consensus, and what does each actually measure?
5. Contributions (see box below). Short answer per RQ in one sentence each.

**Contributions (final wording to be tightened):**
- **C1 Protocol.** A gold-isolated, pre-registered evaluation of reference-free segmentation scoring: artifacts are
  split so that splitters and scorers never read gold; all decisions are made by cross-fitting on GUM train; scores
  are frozen before gold is materialized; one-shot tests on EWT, GENTLE and LinES; 8 dated amendments document every
  change of plan.
- **C2 Method.** A combined boundary-evidence model (MaxEnt stack over a frozen `[MASK]`-state probe, zero-shot SaT,
  a fine-tuned ModernBERT tagger, regular-expression rules and a Kneser–Ney trigram model) scored with *normalized*
  Soft F1. On GENTLE it improves over the frozen method by +0.212 [+0.101, +0.326]; robust to unstratified bootstrap,
  leaving out any genre, and four fine-tuning seeds.
- **C3 Diagnosis.** Evidence, not scoring, is the bottleneck: (a) normalization removes a document-level confound;
  (b) with strong evidence soft scoring ≈ hard pseudo-reference; (c) an oracle-substitution ceiling shows that fixing
  only unpunctuated boundaries suffices to beat Always-SaT; (d) controlled perturbations and error pricing locate two
  blind spots shared by all evidence-based scorers.
- **C4 Negative and complementary results, reported honestly.** No method reliably beats Always-SaT at selection;
  a large share of cross-document agreement is achievable from document difficulty alone (system agreement), which
  consensus captures; evidence adds within-document ranking.

---

## 2 Related work (≈0.6 page) — every citation to be verified before submission

- Sentence boundary detection: Punkt (Kiss & Strunk 2006); Stanza (Qi et al. 2020); WtP (Minixhofer et al. 2023);
  SaT (Frohmann et al. 2024).
- Segmentation evaluation: boundary F1; Pk (Beeferman et al. 1999); WindowDiff (Pevzner & Hearst 2002); B-cubed.
- Reference-free quality estimation: MT QE (Specia et al.; COMET-QE / Rei et al. 2020); pseudo-references and
  consensus (Albrecht & Hwa 2007; Louis & Nenkova 2013).
- Aggregating noisy annotators: Dawid & Skene (1979).
- Probing frozen representations: Tenney et al. (2019); Hewitt & Manning (2019). Calibration: Platt (1999); Guo et al. (2017).
- Models/data: BERT (Devlin et al. 2019); ModernBERT (Warner et al. 2024); Kneser–Ney (Kneser & Ney 1995; Chen & Goodman 1999);
  UD (de Marneffe et al. 2021); GUM (Zeldes 2017); EWT (Silveira et al. 2014); GENTLE (Aoyama et al. 2023); LinES (Ahrenberg 2007).
- Positioning sentence: prior QE work scores *content*; we score *segmentation structure*, where the reference-free
  signal is a boundary probability and the main failure mode is boundaries the signal cannot see.

---

## 3 Task, data and protocol (≈1.2 pages)

**3.1 Task.** Document text (reconstructed from CoNLL-U surface tokens, ignoring `# text` and sentence breaks);
candidate positions = whitespace gaps; a segmentation = a set of gaps. Quality of an output = boundary F1 against gold
(non-whitespace gold boundaries counted exactly, amendment 4a). A scorer sees only the text and the candidate output(s).

**3.2 Data and systems (Table 1).**

| Corpus | Role | Docs | Items (doc × system) | Genres | Status |
|---|---|---:|---:|---:|---|
| GUM train (UD r2.18) | development, LOGO cross-fitting | 197 | 788 | 15 | D |
| EWT test | one-shot test of Probe (amendment 4) | 316 | 1,264 | 5 | T |
| GENTLE test | one-shot test of Stack, primary (amendment 6) | 26 | 104 | 8 | T |
| LinES test | one-shot test of Stack, secondary | 9 | 36 | 1 | T |

Systems: Punkt, Stanza, WtP (`wtp-bert-mini`), SaT (`sat-3l-sm`). GUM dev only diagnosed early (not used for decisions);
GUM test never read. Contamination: Stanza `combined` and SaT `-sm` saw GUM/EWT train (so GUM numbers favor them);
GENTLE and LinES are in no system's training data.

**3.3 Gold isolation.** Separate JSONL artifacts (raw text / gold / candidates, joined by doc id + SHA256); scores are
written to disk before gold is materialized; the analysis stage is the only one that loads gold.

**3.4 Metrics.** Primary (since amendment 2): global Spearman between score and gold F1 over all items; decomposed
into *between-document* (document means) and *within-document* (residuals). Secondary: within-document pairwise
accuracy; (domain × system) cell Spearman; Macro Selection Regret and per-document routing regret vs Always-SaT.
CIs: documents resampled within genre, 1000 draws, paired differences.

**3.5 Pre-registration.** One paragraph + Appendix A timeline of the 8 amendments (what was decided, before which result).

---

## 4 Methods (≈1.3 pages)

**4.1 Scoring a candidate.** Soft F1 = 2 Σ c·p / (Σ c + Σ p) against per-gap evidence p; **normalized Soft F1** divides
by the maximum Soft F1 attainable in that document (exact top-k prefix). One-paragraph argument for normalization
(the per-document maximum varies 0.60–0.94 and is a confound: ρ = 0.79 with raw Soft F1, −0.17 with gold F1).

**4.2 Three evidence models (named consistently across the paper).**

| Name | Evidence p at each gap | Fitted on | Frozen | Tested once on |
|---|---|---|---|---|
| **PuncEv** | `bert-base-uncased` `[MASK]` probability of terminal punctuation, Platt-calibrated | GUM train | proposal / v1.1 | (EWT, as baseline) |
| **Probe** | L2 logistic probe on frozen BERT `[MASK]` hidden states (layers 8/12) + insertion likelihood ratios | GUM train | amendment 4 | EWT |
| **Stack** | MaxEnt over logit(Probe), zero-shot SaT-12l, fine-tuned ModernBERT-large tagger, regex rules, KN trigram features | GUM train (LOGO) | amendment 6 | GENTLE, LinES |

All evidence is computed on lower-cased raw text (no casing features, by design).

**4.3 Baselines and comparison scorers.**
- Always-SaT; Direct Internal Splitter (evidence ≥ 0.5 used as a segmenter).
- Hard pseudo-reference H(E): boundary F1 against evidence ≥ 0.5.
- Multi-system: leave-one-out consensus M0 (majority of the other three); Dawid–Skene with evidence prior M2;
  QE ridge regression M3 (Stack features + M2 + M0).
- Trivial: punctuation-rule reference; sentence-length prior; DocAgree (mean pairwise agreement of the four systems,
  constant within a document — measures document difficulty only).

---

## 5 Results (≈3 pages)

### 5.1 RQ0: reference-free selection does not beat Always-SaT
- PuncEv on GUM train CV: selected − Always-SaT −0.026 [−0.037, −0.021]; Gold-win 0.16; DIS F1 0.831 < every system.
- Final methods (A8.4, Table 6): Macro regret 0 for S0/S1/M3 on all corpora (they pick SaT), but per-document routing is
  worse than Always-SaT on GUM for every method; only M3 on GENTLE is better (P). Conclusion: with a near-oracle default,
  selection is the wrong target → amendment 2.

### 5.2 Diagnosis: the decisive errors are boundaries without terminal punctuation
- Evidence by left context (Table in appendix): 9.8% of gold boundaries have no terminal punctuation; 74–100% of them get p < 0.5;
  610 post-abbreviation non-boundaries have mean p 0.64.
- Oracle-substitution ceiling: fixing only unpunctuated gold boundaries makes selection beat Always-SaT (+0.0011, 14/15 genres)
  even though the evidence F1 (0.879) stays below SaT (0.971). *The evaluator does not need to segment better than the
  systems; it needs to be right on the gaps where they disagree.*

### 5.3 RQ1: cross-document agreement on four corpora (main table)

**Table 2 — global Spearman (normalized Soft F1 unless stated); [95% CI] for test cells.**

| Scorer | GUM (D) | EWT | GENTLE | LinES |
|---|---:|---:|---:|---:|
| PuncEv | 0.355 | 0.338 (T, baseline) | — | — |
| Probe = S0 | 0.574 | **0.411** (T) | 0.672 [0.544, 0.775] (T, baseline) | 0.868 [0.556, 0.974] (T, baseline) |
| Stack = S1 | 0.818 | — | **0.883** [0.826, 0.920] (T) | **0.815** [0.533, 0.921] (T) |
| H(Stack) | 0.810 | — | 0.896 (P) | 0.824 (P) |
| Consensus M0 | 0.733 | 0.657 (P) | 0.830 [0.757, 0.873] (T) | 0.861 [0.583, 0.932] (T) |
| QE ridge M3 | 0.861 | — | 0.900 [0.851, 0.939] (T) | 0.793 [0.425, 0.961] (T) |
| DocAgree | 0.637 | — | 0.773 (P) | 0.741 (P) |
| Punctuation rule | 0.340 | 0.392 (P) | 0.383 (P) | 0.437 (P) |
| Length prior | 0.048 | — | 0.037 (P) | 0.079 (P) |

"—" = not computed by design (Stack was frozen after the EWT test; computing it on EWT would be a second use of that test set).

**Table 3 — key paired differences (global Spearman, [95% CI]).**

| Comparison | GUM (D) | EWT (T) | GENTLE | LinES | Pooled GENTLE+LinES |
|---|---:|---:|---:|---:|---:|
| Probe − PuncEv | +0.219 | +0.074 [+0.023, +0.126] | — | — | — |
| Stack − Probe (S1 − S0) | +0.244 [+0.186, +0.309] | — | **+0.212 [+0.101, +0.326]** (T) | −0.053 [−0.177, +0.065] (T) | +0.155 [+0.078, +0.243] |
| M3 − M0 | +0.127 [+0.090, +0.165] | — | +0.070 [+0.010, +0.137] (T) | −0.068 (T) | +0.011 n.s. |
| Stack − M0 | +0.085 [+0.042, +0.128] | — | +0.053 [−0.016, +0.124] (P) | — | +0.001 (P) |
| Stack − H(Stack) | +0.008 [−0.008, +0.026] | — | −0.013 [−0.037, +0.010] (P) | −0.008 (P) | — |
| Stack − DocAgree | +0.181 [+0.125, +0.246] | — | +0.110 [+0.036, +0.193] (P) | +0.075 [−0.180, +0.424] (P) | — |

Text: the pre-registered test succeeded on GENTLE for both tracks and failed on LinES. Robustness (7.3/7.4) in §5.6.

### 5.4 RQ2: is it the scoring or the evidence?
1. Normalization: +0.235 [+0.181, +0.286] over raw Soft F1 (same evidence, same candidates).
2. Soft ≈ hard once evidence is strong (Stack − H(Stack) n.s. on all corpora); soft helps only with weak evidence (S0 − H(E0) +0.066 on GUM).
3. Ceiling curve (Figure 2): global Spearman rises almost linearly with evidence F1.
4. **Controlled perturbations (A8.2, Table 4, Figure 3).** GUM, 197 docs, 7 error types × 4 rates × 20 replicates (~110k candidates).

**Table 4 — GUM (D): detection rate at 10% perturbation / within-document slope of score on true F1 (ideal 1).**

| Scorer | merge, terminal | merge, no terminal | split after terminal (abbrev.) | split after comma etc. | split, no punct. | shift | mixed |
|---|---|---|---|---|---|---|---|
| PuncEv | 0.997 / 1.04 | **0.288 / −0.23** | **0.176 / −0.84** | 1.000 / 0.85 | 1.000 / 0.84 | 0.999 / 0.86 | 0.928 / 0.69 |
| Probe | 1.000 / 1.03 | 0.758 / 0.32 | 0.790 / 0.48 | 1.000 / 0.92 | 1.000 / 0.94 | 1.000 / 0.94 | 0.991 / 0.87 |
| Stack | 1.000 / 1.07 | 0.695 / 0.24 | 0.737 / 0.39 | 1.000 / 0.95 | 1.000 / 0.97 | 1.000 / 0.98 | 0.989 / 0.89 |
| Consensus | 1.000 / 1.07 | 0.493 / −0.07 | 0.637 / 0.33 | 1.000 / 0.96 | 1.000 / 0.96 | 1.000 / 0.95 | 0.975 / 0.84 |
| Dawid–Skene | 1.000 / 1.00 | 0.957 / 0.72 | 0.822 / 0.60 | 1.000 / 0.88 | 1.000 / 0.93 | 1.000 / 0.95 | 0.994 / 0.91 |

Message: all evidence-based scorers are near-perfect on four of six error types and share two blind spots; the
blind spots are worse on GENTLE and inverted on LinES (Stack slopes −0.14 / −0.48), which explains LinES (§6).
On synthetic mixed candidates, cross-document Spearman: Stack 0.919, DS 0.876, Probe 0.840, consensus 0.807, PuncEv 0.635.
5. Error pricing on real outputs (A8.3): fixing a system's missed unpunctuated boundaries is rewarded at 16% of its true F1
   value under Stack (missed punctuated boundaries: 89%); ratio 0.18 [0.11, 0.25]. Punkt's errors are dominated by this type
   (its absolute cost +0.059 F1 on GUM, +0.238 on GENTLE) → systematic overrating of Punkt (rank residual +0.067 / +0.046).

### 5.5 RQ3: evidence vs consensus measure different things

**Table 5 — between- / within-document decomposition.**

| Scorer | GUM between / within | GENTLE between / within | LinES between / within |
|---|---|---|---|
| Probe (S0) | 0.544 / 0.817 | 0.724 / 0.441 | 0.900 / 0.680 |
| Stack (S1) | 0.896 / 0.776 | 0.925 / 0.692 | 0.833 / 0.654 |
| Consensus (M0) | 0.893 / 0.579 | 0.900 / 0.370 | 0.917 / 0.526 |
| M3 | 0.931 / 0.849 | 0.944 / 0.865 | 0.833 / 0.838 |
| DocAgree | 0.913 / — | 0.883 / — | 0.867 / — |

Message: recognizing *which documents are hard* is easy (DocAgree alone gets between-document ρ ≈ 0.87–0.91);
evidence's added value is *within-document* ranking, where consensus is weak (it cannot see errors all systems share:
12.7% of system errors on GUM). Combining both (M3) is best on GUM and GENTLE.
Gold as a candidate (A8.1): no scorer ranks gold first in ≥ 80% of documents (Stack 0.40 / 0.43 / 0.15) — systems are
within a median of 2–13 gaps of gold, and those gaps are exactly the blind spots.

### 5.6 Robustness (short; details in appendix)
- GENTLE S1 − S0 robust: unstratified bootstrap +0.212 [+0.069, +0.411]; positive when leaving out each of 8 genres.
  M3 − M0 not robust (unstratified CI crosses 0).
- Fine-tuning seeds (4): selection always Stack; E2 alone unstable (0.594–0.781), Stack stable (0.804 ± 0.010).
- Ablation (A8.6): removing zero-shot SaT evidence costs −0.091 (GUM) / −0.070 (GENTLE); it does not favor SaT —
  it reduces SaT's underrating (rank residual −0.104 → −0.048). GENTLE without it ranks Punkt first (wrong).
- Document length: short documents slightly worse (S1 0.809 vs 0.870 for long).

### Table 6 — selection (original question), per-document routing regret minus Always-SaT [95% CI]

| Scorer | GUM (D) | GENTLE (P) | LinES (P) |
|---|---:|---:|---:|
| PuncEv | +0.027 [+0.022, +0.032] | — | — |
| Probe S0 | +0.005 [+0.002, +0.009] | −0.003 [−0.041, +0.028] | +0.002 [−0.004, +0.009] |
| Stack S1 | +0.006 [+0.003, +0.010] | −0.009 [−0.056, +0.041] | +0.004 [−0.003, +0.012] |
| Consensus M0 | +0.019 [+0.015, +0.023] | +0.011 [−0.026, +0.042] | +0.005 [−0.004, +0.014] |
| M3 | +0.003 [+0.001, +0.006] | **−0.043 [−0.079, −0.015]** | −0.003 [−0.007, +0.001] |

(Main text keeps this table only if space allows; otherwise move to appendix and quote two numbers in §5.1.)

---

## 6 Error analysis (≈0.5 page) — linguistic explanation of the failures
- Unpunctuated boundaries: headings, list items, captions, dialogue turns, boundaries before a dash/colon. Show 3 short examples.
- Post-abbreviation periods (“Dr.”, “e.g.”, initials) and sentence-internal quotes: evidence fires, gold does not split.
- LinES: annotation conventions differ (boundaries without terminal punctuation; terminal punctuation inside non-final
  units); under Stack both blind spots invert (negative slopes) → why the test failed.
- Punkt overrated / SaT underrated: Punkt's errors are concentrated in the blind spot; SaT's correct unpunctuated splits
  are not rewarded.
- **TODO (team, manual):** sample the 10 most discordant document pairs on GENTLE and LinES, categorize, pick examples.

## 7 Discussion (≈0.4 page)
- A reference-free score is only as good as its evidence on the gaps where systems disagree; formula engineering has
  diminishing returns. Practical recipe: use evidence for within-document ranking, system agreement for document difficulty.
- Evaluating vs predicting: a scorer that is itself a strong segmenter is, functionally, a pseudo-reference (soft ≈ hard).
  This is useful but it inherits the pseudo-reference's blind spots.

## 8 Limitations (≈0.4 page, honest)
- Small test sets (GENTLE 26 docs / 104 items; LinES 9 / 36); LinES result is a failure, not noise we can explain away.
- Adaptive design: eight amendments; all test numbers are labeled T vs P; post-test analyses (A7, A8) are exploratory.
- Only English, only four systems, only UD-style annotation; gold conventions differ across treebanks (LinES).
- Training-data contamination of Stanza/SaT on GUM/EWT; GUM development numbers favor them.
- Uncased by design: casing is a strong cue we chose not to use (it helped in P2), so absolute numbers are conservative.
- A single fine-tuned evidence model is seed-sensitive; Stack mitigates it but four seeds is still few.
- Selection: no method reliably beats Always-SaT; the operational use case of the proposal is not achieved.
- PuncEv evidence on GUM is 5-fold rather than LOGO cross-fitted.

## 9 Conclusion (≈0.2 page)
Answer RQ0–RQ3 in four sentences; one sentence of future work (targeted evidence for unpunctuated boundaries, e.g.
layout-aware or discourse cues; annotation-convention adaptation).

## End matter
- Contribution statement (who did what) — **if required by the course**.
- Code/data availability: GitHub repository, branch, commit; all results regenerable from `scripts/*.slurm`.

---

## Appendix plan
A. Pre-registration timeline (amendments 1–8: date, decision, which results had been seen).
B. Preliminary study: Qwen vs BERT on synthetic candidates (BERT 0.239 vs Qwen 0.211 global Spearman; Soft F1 ≫ BCE).
C. P2 calibration search (casing features help; excluded by design).
D. Candidate-conditioned context (raw text wins) and paragraph-layout control (`para`, supplementary).
E. Evidence by left context + oracle-substitution ceiling (full table).
F. Full per-corpus tables (fairness residuals, Pk / WindowDiff / B-cubed robustness, Dawid–Skene label flip M1).
G. Full perturbation and error-pricing tables for GENTLE and LinES (A8).
H. Reproduction: scripts, job ids, data release and checksums.

## Figures (generated by a script from `results/*.json`, to be added as `paper/figures/make_figures.py`)
1. Pipeline and gold-isolation diagram.
2. Ceiling curve: evidence F1 → global Spearman (GUM), with PuncEv / Probe / Stack marked.
3. Perturbation sensitivity by error type (slopes, 5 scorers, GUM) — the blind-spot figure.
4. Between- vs within-document bars for S0 / S1 / M0 / M3 / DocAgree on the three corpora.

## Open decisions (for the team)
1. Title (two options above).
2. Page limit and deadline (course page).
3. Format: LaTeX ACL template. No TeX on the cluster → write `.tex` in the repo, compile on Overleaf.
4. Team-size / contribution statement.
5. Who writes what: suggested split — Weifeng: §3–5 (protocol, methods, results); Yanhan: §1–2, §6 error-analysis examples, slides; both: §7–9.
