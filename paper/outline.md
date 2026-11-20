# Paper outline (v4, 2026-11-19)

Framing: **"we built a working reference-free evaluator, made it substantially more reliable, and analyzed what
determines its reliability and where its limits are."** v4 adds the amendment-12 results: the scoring function now
counts the document-final boundary (Q̃_end), and the final evaluators were tested once on two fresh corpora (EWT dev,
LinES train+dev) with pre-registered hypotheses; all numbers below use Q̃_end unless marked. v1–v3.1 are in git history.

Format: ACL template, **8 pages main text**; references and appendix do not count.
Markers: **[M#]** math block to write (planned, not written); **[F#]** figure slot; **[T#]** table slot;
**D** = GUM-train development (leave-one-genre-out, LOGO); **T** = pre-registered one-shot test (scores frozen before
gold was loaded); **R** = re-used test set (reported only, never used for a decision); **P** = post-hoc analysis.
All numbers are tie-exact (erratum 2026-11-19) and checked against `results/` on 2026-11-19.

---

## 0 For the team (not part of the paper)

### 0.1 Central question and the five-sentence spine
**Central question:** *Can we estimate the quality of a sentence segmentation without gold boundaries in the target
domain, and what determines how reliable such an evaluator is?*

1. **Problem.** On a new domain there are no gold boundaries, so we cannot tell how good a segmenter's output is.
2. **Method.** Our evaluator predicts a boundary probability at every whitespace gap and scores a candidate segmentation
   by its normalized Soft F1 against these probabilities (counting the document end) — no target-domain gold is needed.
3. **Experiments.** We score the outputs of four real segmenters (Punkt, Stanza, WtP, SaT) on English corpora under a
   gold-isolated, pre-registered protocol: develop on GUM, test each frozen version once on held-out corpora, and test the
   final evaluator once more on two fresh corpora; we compare against hard pseudo-references and system consensus.
4. **Results.** The evaluator works and can be made much more reliable: heterogeneous boundary evidence raises agreement
   with true quality from 0.67 to 0.88 on the GENTLE test (T); on fresh web text (EWT dev, T) the final evaluator reaches
   0.851, against 0.656 for the initial probe and 0.625 for system consensus, and it ranks the four systems correctly.
5. **What determines reliability.** The boundary evidence, not the scoring formula (apart from one fixable failure on
   single-sentence documents): controlled perturbations show which errors the evaluator notices — all evidence misses
   unpunctuated boundaries and false splits after abbreviations, and casing/layout move rather than remove these blind
   spots; evidence ranks systems far better than consensus; on text with different annotation conventions (LinES, twice)
   the improvements do not transfer, and selection cannot beat always using SaT.

### 0.2 Requirement map (proposal / progress report / course → where the paper answers)

| Requirement | Answer | Where |
|---|---|---|
| Proposal question: can a reference-free evaluator select a better splitter than Always-SaT (Macro Selection Regret)? | Not shown: SaT is the best system on every corpus (near-oracle), so there is almost no headroom. The evaluator *does* identify SaT as best (system-level ranking), which is the useful part of the selection question. | §5.1, §5.3-D, App. F |
| Required baselines: Always-SaT, Direct Internal Splitter (DIS) | Both reported. The first DIS (F1 0.831) is below every system; the case + layout evidence used as DIS later beats SaT-3l-sm on all four corpora (P/R). | §5.1, §6.1, App. G |
| Held-out evaluation: develop on GUM, freeze, test once on EWT | Done (T); extended to GENTLE and LinES (T), and a final one-shot test of the final evaluator on fresh EWT dev and LinES train+dev (T, amendment 12). | §4.3, T2 |
| Supporting outcomes: within-document pairwise accuracy, domain-level ranking | Reported (T2 columns; per-genre best-system identification). | §5.1, §5.3-D |
| Course: apply course methods | Regex rules (L2), Kneser–Ney n-gram (L5), logistic regression / MaxEnt (L6–L7), BIO tagging (L7), P/R/F (L6); B-cubed and HMM length prior in the appendix. | §3.3 |
| Course: related work with ≥ 5 published NLP papers | Yes (≈15 references). | §2 |
| Course: honest limitations, presentation | §6.2; slides from F1, T2, F4. | — |
| Team-size / contribution statement | **To confirm with the instructor** (we are two); contribution statement in the end matter. | end matter |

### 0.3 Main text vs appendix: where every experiment goes
Rule: an experiment is in the main text only if a results subsection cannot make its claim without it; otherwise it
gets **at most one sentence** in the main text and its full content in the appendix.

| Experiment (repo label) | Main text | Appendix |
|---|---|---|
| Synthetic-perturbation pilot, Qwen vs BERT, BCE vs Soft F1 (preliminary) | 1 sentence in §5.1 (why we moved to real splitters) | A |
| Real-splitter selection v1, train CV, oracle headroom (proposal question) | §5.1, one paragraph | F |
| P2 calibration search; candidate-conditioned context; paragraph-[SEP] control | 1 sentence each at most (§3.3, §5.2) | A |
| Normalization vs raw Soft F1 | §5.3-A, one number | B |
| Probe (amendment 3) + EWT one-shot test (amendment 4) | §5.1 (initial evaluator), T2 | B |
| Evidence stack + GENTLE/LinES one-shot tests (amendment 6) | §5.2, T2 | B |
| Dawid–Skene M1/M2, HSMM length prior | none | C |
| Multi-system QE regression M3 / S2 | one row of T2; §5.2 (post-hoc QE vs quality-oriented training) | C |
| Robustness: unstratified bootstrap, leave-one-genre-out, 4 seeds | 1 sentence in §5.2 | D |
| Hard pseudo-reference H(E) | §5.3-A | C |
| Controlled perturbations (8.2) | **§5.3-B, F4** | C |
| Error pricing on real outputs (8.3) | §5.3-B, 1–2 sentences | C |
| Gold as a 5th candidate (8.1), document length (8.7) | 1 sentence / none | C |
| Selection regret with final methods (8.4) | §5.1, numbers in one sentence | F |
| Trivial baselines: DocAgree, punctuation rule, length prior (8.5) | rows of T2; DocAgree in §5.3-C | C |
| Leave-one-evidence-out ablation, same-family check (8.6) | 1 sentence in §5.2 | C |
| Casing trial, SaT design analysis (9.1) | casing rows of T2 | D |
| Case × layout factorial (9.2+9.3) | layout block of T2 + 2 sentences in §5.2 | D |
| Evidence as a segmenter vs SaT-3l / 12l / 12l-sm | §6.1, 1 sentence | G |
| Quality-oriented fusion vs post-hoc QE (amendment 10) | **§5.2, T3** | E |
| System-level ranking / best-system identification (new, P) | **§5.3-D (or a column of T2)** | C |
| Tie-exact erratum | footnote in §4.2 | H |
| Document-final boundary in the scoring (Q̃_end) | §3.2 (definition), §5.3-A (why) | B |
| Final one-shot tests on EWT dev and LinES train+dev (amendment 12) | **T2 columns, §5.2, §5.3** | H (hypotheses table) |

### 0.4 Page budget (8 pages)
§1 0.9 · §2 0.6 · §3 1.5 · §4 1.0 · §5 3.0 · §6 0.8 · §7 0.2. Floats: F1 (full width, top of p. 2–3), F3–F5 (one column
each), T1 (small), T2 (full width), T3 (one column). F2 optional (merge into F1 as panel b if space is short).

### 0.5 Status of the additional experiments
- Amendment 11 (post-hoc, `results/a11/`): system-ranking agreement (E1), scoring formula (E2), real-error pricing (E3),
  discordant examples (E4, `results/a11/examples.md`), perturbation test v2 (F4) — all in §5.3.
- **Amendment 12 (one-shot tests, `results/a12/summary.md`)**, pre-registered hypotheses:

| Hypothesis | Estimate [95% CI] | Outcome |
|---|---:|---|
| H1 (primary) EWT dev, layout: final − consensus | +0.227 [+0.158, +0.298] | confirmed |
| H2 EWT dev: final − Probe (layout / flat) | +0.195 [+0.133, +0.256] / +0.202 [+0.146, +0.261] | confirmed |
| H3 EWT dev: Q̃_end − Q̃ for the final evaluator | +0.348 [+0.251, +0.449] | confirmed |
| H4 EWT dev: genre-mean system-ranking τ, final − consensus | +0.533 [+0.067, +0.600] | confirmed |
| H5 LinES train+dev: final − Probe | −0.088 [−0.245, +0.075] | no improvement (LinES failure reproduced) |
| H6 LinES: layout evaluator − flat evaluator (text without layout) | −0.095 [−0.299, +0.050] | expected direction, not significant |

---

## Title and abstract

**Title (proposal title):** *Cross-Domain Reference-Free Quality Estimation for Sentence Segmentation*

**Abstract (draft, ~170 words; five sentences of §0.1, no list of model versions).** Sentence segmentation quality is
hard to measure in a new domain because gold boundaries are missing. We develop a reference-free evaluator that predicts
a boundary probability at every candidate position and scores a segmentation by its normalized Soft F1 against these
probabilities. We evaluate the outputs of Punkt, Stanza, WtP and SaT under a gold-isolated, pre-registered protocol with
one-shot tests on held-out corpora, and compare with hard pseudo-references and system consensus. Combining heterogeneous
boundary evidence improves agreement with true quality from 0.67 to 0.88 (Spearman, held-out GENTLE test), and on fresh
web text the final evaluator reaches 0.85, well above the initial evaluator (0.66) and system consensus (0.63), while
ranking the segmenters correctly. Analysis shows that reliability is determined by the boundary evidence rather than the
scoring formula, that a shared blind spot — boundaries without terminal punctuation — and different annotation
conventions remain the main limits, and that evidence and consensus capture complementary information.

---

## 1 Introduction (0.9 page)
*This section answers: why this project, and what problem does it solve?*

### 1.1 Motivation and problem
- Sentence segmentation is a basic NLP step whose errors propagate (parsing, MT, retrieval). Segmenters (Punkt, Stanza,
  WtP, SaT) differ across domains, but a new domain has no gold boundaries, so their outputs cannot be scored.
- Goal: given raw text and one segmenter's output, score the output's quality **without** the target domain's gold.
- Distinguish two tasks explicitly: **sentence splitting** (predict where boundaries are) vs **quality estimation**
  (judge how good an existing segmentation is). We do the second; gold from a *source* corpus (GUM) may be used for
  training, target gold only for validation.

### 1.2 Our approach (points to F1)
Predict a boundary probability per gap (boundary evidence), compare the candidate's boundaries with it (normalized Soft
F1). Start from a BERT-based evaluator; guided by its errors on real systems, build an improved evaluator that combines
several models and linguistic features, and train the combination for quality estimation.

### 1.3 Research questions and contributions
- **RQ1** Can boundary evidence provide reliable reference-free quality estimates for real sentence splitters across
  domains, and how much can stronger evidence improve them? (§5.1–§5.2)
- **RQ2** What determines the reliability of such an evaluator — the boundary evidence, the scoring formula, or the
  information available from other systems — and where are its limits? (§5.3)

**Contributions.**
1. **A working reference-free evaluator, evaluated on real systems.** An end-to-end pipeline that scores the outputs of
   four real segmenters without target gold, built from course methods (regular expressions, n-gram LM, MaxEnt, BIO
   tagging), with a gold-isolated, pre-registered protocol: every frozen version tested once on held-out corpora, and the
   final evaluator tested once more on two fresh corpora (§3–§4, §5.1).
2. **A substantially more reliable evaluator.** Heterogeneous boundary evidence (+0.212 [+0.101, +0.326] Spearman over the
   initial probe on the GENTLE test), casing, paragraph layout, a quality-oriented training objective that holds for
   segmenters unseen in training (unlike post-hoc quality regression), and a corrected scoring function; on fresh EWT dev
   the final evaluator beats the initial probe by +0.195 and system consensus by +0.227 (§3.2–§3.4, §5.2).
3. **An analysis of what makes it reliable.** Normalization (and counting the document end) is all the formula needs;
   evidence quality drives reliability; ~110k controlled perturbations show which error types the evaluator notices and
   how casing/layout shift its blind spots; evidence and consensus measure different things (which system vs which
   document); and we report where reliability ends (different annotation conventions, splitter selection) (§5.3, §6).
→ *Link to §2: what exists for evaluating segmentation and for reference-free evaluation.*

## 2 Related work (0.6 page)
*This section answers: what have others done, and how is our project different?* Two themes only.
### 2.1 Sentence segmentation
Punkt (Kiss & Strunk 2006, unsupervised, abbreviation-aware); Stanza (Qi et al. 2020, neural tokenizer); WtP (Minixhofer
et al. 2023) and SaT (Frohmann et al. 2024, newline-pretrained, punctuation-robust). Different mechanisms → good test of an evaluator.
### 2.2 Evaluating without a reference
Reference-based segmentation metrics (boundary F1; Pk, Beeferman et al. 1999; WindowDiff, Pevzner & Hearst 2002);
reference-free QE for MT (Specia et al. 2018; COMET-QE, Rei et al. 2020); pseudo-references and consensus (Albrecht &
Hwa 2007; Louis & Nenkova 2013); limits of reference-free metrics, incl. bias toward their own underlying models
(Deutsch et al. 2022); reference-free evaluation of *topic* segmentation (SegReFree — **verify citation**).
Difference (one sentence): we evaluate **sentence-boundary quality of existing splitter outputs**, not generated text or topic segments.
→ *Link to §3.*

## 3 Methodology (1.5 pages)
*This section answers: what exactly does the evaluator do? (F1 + two formulas should suffice to understand it.)*

**[F1] Evaluator overview.** Raw document → gaps → boundary evidence sources → fusion → p_i; candidate segmentations of
four segmenters → normalized Soft F1 → scores/ranking; a dashed box marks everything that never sees target gold.

### 3.1 Task formulation
Input: raw document X and a candidate segmentation S from a splitter; output: a score Q(X, S) computed without target
gold; validity = how well Q ranks candidates by their true boundary F1. **[M1]** gaps i = 1..n, candidate y ∈ {0,1}^n,
gold g, true quality F1(y, g). Training may use GUM gold; target gold only for validation.

### 3.2 Scoring a candidate against boundary evidence
**[M2]** Soft F1 with the document-final boundary counted, Q(y, p) = 2(Σ y_i p_i + 1) / (Σ y_i + Σ p_i + 2), and the
**normalized** score Q̃ = Q / max_y' Q(y', p) (closed form: the best y' takes the k highest p_i, k = 0..n). Why: Q is a
soft boundary F1 with p as a probabilistic reference; the per-document maximum varies (0.60–0.94) and would otherwise
confound comparisons across documents; Q̃ = 1 means "as good as the evidence allows". The final boundary (always right)
makes the correct "no split" of a one-sentence document score 1 instead of 0; for long documents it changes nothing (§5.3-A).

### 3.3 Boundary evidence: from a single BERT signal to a heterogeneous stack
- **Initial evaluator.** Frozen `bert-base-uncased`: probability of terminal punctuation at a `[MASK]` (PuncEv), then a
  MaxEnt probe on the `[MASK]` hidden states (Probe). **[M3a]** p_i = σ(w·φ(h_i) + b).
- **Improved evaluator (evidence stack).** Logistic regression over: Probe; zero-shot SaT-12l probability; fine-tuned
  ModernBERT-large boundary tagger (BIO-style, punctuation dropout); regular-expression rules (abbreviations, initials,
  numbering); Kneser–Ney trigram features (P(`</s>` | context)); and, when the input has paragraphs, a paragraph-break
  flag. **[M3b]** p_i = σ(θ·x_i). One column "course topic" in a small inline table or the text.
- Casing: evidence models read original casing in the final version (lower-cased in the frozen tests). Details,
  hyperparameters and failed variants → App. B.
### 3.4 Training the fusion for quality
Standard: fit θ by boundary log loss **[M4a]**. Quality-oriented: add the squared error between Q̃(y, p_θ) and the true
F1 of the training candidates, **[M4b]** L = L_Q/L_Q⁰ + L_BCE/L_BCE⁰ + λ‖θ‖² (terms scaled by their value at the BCE
solution; gradients flow through the top-k sort). Intuition: the boundary loss treats all gaps alike; the quality loss
focuses on the gaps that decide which candidate is better.
### 3.5 Comparison methods
Hard pseudo-reference H(E) (threshold p at 0.5, boundary F1); system consensus M0 (agreement with the other three
segmenters); DocAgree (mean pairwise agreement of the four systems — document difficulty only); post-hoc QE regression
(S2/M3, ridge on score and document features, ± consensus; supplementary, not our main algorithm); punctuation-rule
reference and sentence-length prior; for selection, Always-SaT and the Direct Internal Splitter.
→ *Link to §4: how we test whether the scores are trustworthy.*

## 4 Experimental setup (1.0 page)
*This section answers: how do we show that the method works, and is the comparison fair?*
### 4.1 Datasets and systems
**[T1]** GUM train (D: training and development, 197 docs, 15 genres); EWT test (T: initial held-out evaluation, 316
docs); GENTLE test (T: held-out evaluation of the evidence stack, 26 docs, 8 genres); LinES test (T: secondary, 9 docs; no
paragraph layout); **EWT dev** (T: final one-shot test of the final evaluator, 318 docs, 5 genres, with layout) and
**LinES train+dev** (T: final one-shot test, 18 docs, no layout). Candidates: Punkt, Stanza, WtP (wtp-bert-mini), SaT
(sat-3l-sm) → items = document × system. Two input conditions: flat text, and text with paragraph layout (UD `# newpar` as
`\n`, seen by systems and evaluator; idealized). Contamination: SaT-sm and Stanza saw GUM/EWT training data (not EWT dev
itself); GENTLE and LinES are unseen by all systems.
### 4.2 Evaluation metrics
Ground truth = human-annotated UD sentence boundaries; quality of a candidate = its boundary F1. **[M5]** Primary: global
Spearman ρ between Q̃ and true F1 over all items (higher = better ranking across documents and systems). Supporting:
within-document pairwise accuracy; between/within-document decomposition; system-level ranking (best system per corpus
and per genre) and selection regret vs Always-SaT; boundary F1 of the evidence itself; error-type sensitivity (**[M6]**
perturbation slope, ideal 1). Footnote: identical candidates are tied exactly before ranking (erratum, App. H).
### 4.3 Protocol
LOGO development and all decisions on GUM, written down before results; freeze; score each test corpus before its gold is
loaded (gold isolation); one test per frozen evaluator (EWT test for the Probe; GENTLE and LinES for the stack). Later
improvements are decided on GUM only and reported on the used test sets as R; the **final evaluator** is then tested once
on two fresh corpora with pre-registered hypotheses (amendment 12). Genre-stratified document bootstrap (1000) for CIs and
paired differences. **[F2, optional]** timeline D → T → R → final T.
→ *Link to §5.*

## 5 Results and analysis (3.0 pages)
*Three consecutive questions: does it work → can we make it more reliable → what determines its reliability.*

### 5.1 Does the initial evaluator work on real splitters?
- One sentence: a synthetic-perturbation pilot showed BERT + Soft F1 is promising (App. A), but real systems are the test.
- Initial evaluator on real outputs: Probe beats PuncEv on the EWT one-shot test (0.411 vs 0.338, +0.074 [+0.023, +0.126],
  T, scored with the original Q̃; 0.647 vs 0.508 with Q̃_end, P); GUM 0.568; GENTLE 0.672 and LinES 0.868 (T).
- The original selection goal is not met: SaT is the best system on every corpus, Always-SaT is near-oracle (GUM regret
  0.0014), and the initial evaluator's selection is worse than Always-SaT (−0.026 [−0.037, −0.021]); details App. F.
- **Conclusion of 5.1:** reference-free evaluation of real splitters is feasible, but the initial evaluator's
  reliability is moderate → can stronger evidence make it more reliable?

### 5.2 Can stronger boundary evidence make the evaluator more reliable?
**[T2] Global Spearman (Q̃_end) by evaluator version × corpus.** D = GUM LOGO; T = one-shot test of that version
(T* = pre-registered test scored with the original Q̃ — values in the footnote); R = re-used test; **fresh** = the
final one-shot tests (amendment 12). Best evaluator per column bold.

*Flat text*

| Evaluator | GUM (D) | GENTLE | LinES test | EWT test | **EWT dev (fresh, T)** | **LinES tr+dev (fresh, T)** |
|---|---:|---:|---:|---:|---:|---:|
| Initial: Probe | 0.568 | 0.661 T* | 0.868 T* | 0.647 T* | 0.595 | **0.651** |
| Evidence stack | 0.817 | 0.881 T* | 0.818 T* | 0.772 R | 0.739 | 0.560 |
| + casing | 0.877 | 0.901 R | 0.840 R | **0.814** R | **0.800** | 0.541 |
| + quality-oriented training (final, flat) | **0.903** | **0.907** R | **0.873** R | 0.787 R | 0.797 | 0.563 |
| *System consensus M0* | 0.733 | 0.830 T | 0.861 T | 0.657 | 0.657 | 0.468 |
| *DocAgree (document difficulty only)* | 0.637 | 0.773 | 0.741 | 0.529 | 0.508 | 0.410 |

Footnote: with the original Q̃ the pre-registered cells are Probe 0.672 / 0.868 / 0.411 and stack 0.883 / 0.818; Q̃_end
changes GUM, GENTLE and LinES by ≤ 0.007 and EWT a lot (single-sentence documents, §5.3-A). Punctuation-rule and
length-prior baselines (0.34–0.44 and ≈ 0.05) and M3 in App. C.

*Text with paragraph layout (systems and evaluator see paragraph breaks)*

| Evaluator | GUM (D) | GENTLE (R) | EWT test (R) | **EWT dev (fresh, T)** |
|---|---:|---:|---:|---:|
| Initial: Probe | 0.690 | 0.643 | 0.692 | 0.656 |
| Evidence stack | 0.843 | 0.848 | 0.800 | 0.776 |
| + casing | 0.897 | 0.867 | 0.845 | 0.831 |
| + layout feature | 0.939 | 0.886 | 0.892 | 0.847 |
| + quality-oriented training (final, layout) | **0.948** | **0.903** | **0.894** | **0.851** |
| *System consensus M0 / DocAgree* | 0.675 / 0.475 | 0.795 / 0.747 | 0.590 / 0.482 | 0.625 / 0.509 |

Text:
- **Clean tests.** The evidence stack improves over the Probe on GENTLE, +0.212 [+0.101, +0.326] (T), robust to
  unstratified bootstrap and to leaving out any GENTLE genre; the stack is stable across four fine-tuning seeds on GUM
  (App. D). On fresh EWT dev every step of the ladder helps and the final evaluator beats the Probe by +0.195 [+0.133,
  +0.256] (layout) and +0.202 [+0.146, +0.261] (flat), and consensus by +0.227 [+0.158, +0.298] / +0.141 [+0.075, +0.213]
  (T, pre-registered H1–H2). On LinES the improvements do not transfer, on the test (−0.050, n.s.) nor on fresh LinES
  train+dev (−0.088 [−0.245, +0.075], H5): "no improvement shown" twice — but the evaluator still beats consensus there
  (+0.095 [+0.007, +0.210]).
- Removing any evidence source hurts on GUM and GENTLE, SaT evidence most (App. C).
- **Casing and layout** (decided on GUM): each gives a large gain; they are substitutes (interaction −0.059 [−0.078,
  −0.043]) because both reveal the same boundaries (two sentences; factorial in App. D).
- **[T3] Quality-oriented training vs post-hoc QE** (GUM LOGO, same features/folds/items; Δ vs the BCE stack; LOSO = scored segmenter unseen in training):

| Method | Flat | LOSO Δ | Layout | LOSO Δ |
|---|---:|---:|---:|---:|
| Stack (boundary loss) | 0.877 | — | 0.939 | — |
| Post-hoc ridge QE (S2) | 0.860 | −0.032 | 0.927 | −0.023 |
| Post-hoc F1 regressor / residual QE | 0.873 | −0.024 | 0.930 | −0.017 |
| Quality loss only | 0.849 | −0.066 | 0.933 | −0.007 |
| **Quality + boundary loss** | **0.903** (+0.026 [+0.014, +0.038]) | **+0.022** | **0.948** (+0.009 [+0.004, +0.015]) | **+0.010** |

  Post-hoc regressors learn the four seen systems and lose on an unseen one; quality loss alone breaks the
  probabilities (held-out log loss ×3–5); quality + boundary loss improves and transfers (log loss unchanged). Outside
  GUM the gain is small (GENTLE +0.007 / +0.020 R; EWT dev final vs +layout +0.004, T).
- **Conclusion of 5.2:** stronger evidence makes the evaluator substantially more reliable, with clean held-out gains on
  GENTLE and fresh EWT dev, but not on text with different annotation conventions (LinES) → what determines reliability?

### 5.3 What determines evaluation reliability?
**A. The scoring formula (normalized vs raw vs hard; one failure mode).** Normalization is essential (GUM: +0.235
[+0.181, +0.286] for PuncEv, +0.053 [+0.025, +0.084] for the final evaluator). With strong evidence, normalized Soft F1 ≈
hard pseudo-reference (GUM final +0.008 n.s. flat, +0.015 with layout; GENTLE n.s.; EWT dev +0.005 n.s. / +0.009). **One
failure mode of plain Soft F1**: in a document with no internal boundary (a single sentence) the correct "no split" has
Soft F1 = 0, so the best candidate is ranked last. Found post hoc on the EWT test (33/316 such documents; final evaluator
0.572 → 0.894 when the document end is counted, P), adopted before the final test, and **confirmed on fresh EWT dev**
(42/318 single-sentence documents; +0.348 [+0.251, +0.449], T, H3); GUM, GENTLE and LinES have no such documents.
**[F3] Evidence quality vs reliability:** one point per evaluator version (boundary F1 of the evidence used as a
segmenter, or held-out log loss, vs global Spearman; GUM, GENTLE, EWT dev): reliability rises with evidence quality.

**B. Which errors does the evaluator notice? (controlled perturbations)** **[F4] Detection heatmap** (main figure of
§5.3): inject exactly one error of a type into the gold segmentation; detection rate = share of cases where the score
drops (ties ½); rows = error types, columns = PuncEv, Probe, Stack, +casing, +layout, final, consensus; panel (b) = slope
of score on true F1 (does the penalty scale with the true loss; ideal 1). **[M6]** both definitions. GUM, ~197 docs ×
9 types × 5 levels × 20 replicates. Findings (GUM; replicated on GENTLE, EWT test and fresh EWT dev, App. C):
- Every version notices merges at terminal punctuation, splits at commas or unpunctuated positions, and shifted
  boundaries (≈ 1.0). Two error types are hard:
- **Missing a boundary without terminal punctuation.** At a paragraph break: Stack 0.67 → +casing 0.81 → +layout 0.98
  → final 0.98. *Inside a paragraph*: Stack 0.52, +casing 0.61, final (flat) 0.62, but **+layout 0.05 and final (layout)
  0.07 [0.03, 0.13]** — having learned "unpunctuated boundary ≈ newline", the layout evaluator now *rewards* removing
  such boundaries (slope −0.59); on fresh EWT dev the same: 0.84 (flat final) vs 0.35 (layout final). This trade-off
  explains the layout evaluator's drop on LinES (no paragraph marks; H6 −0.095, n.s.).
- **False split after an abbreviation or numbering**: PuncEv 0.22 (rewards it), Probe 0.77, Stack 0.73, final 0.81–0.84
  on GUM; only 0.19–0.36 on LinES, whose conventions differ.
- Consensus is blind to unpunctuated merges at paragraph breaks (0.46): all systems make them.
Real outputs (E3, one or two sentences): fixing a system's missed boundaries at paragraph breaks is rewarded at 0.34 of
their true value by the Stack and at 1.01 by the final evaluator; inside paragraphs −0.18 and −0.58 (fixing them is
penalized). Punkt's errors concentrate in these blind spots, so it is overrated (rank residual +0.067 on GUM). Two or
three real examples from E4 (e.g. GENTLE poetry: boundaries after semicolons and dashes get p = 0.02–0.4, so Punkt,
which never splits there, is overrated). Takeaway: a high overall Spearman can hide error types that are mis-priced.

**C. Evidence vs consensus: they measure different things.** **[F5] Between/within-document decomposition.** Consensus is
good at telling which *documents* are hard (between-document ρ 0.89–0.92, like DocAgree) but weak within a document
(0.37–0.58) and bad at ranking the *systems*. **[T4] System-ranking agreement** (Kendall τ_b between the evaluator's and the
true ranking of the four systems, averaged over genres / documents): GUM final 0.933 / 0.894 vs consensus 0.489 / 0.458;
GENTLE 0.773 / 0.758 vs 0.300 / 0.383; fresh EWT dev 1.000 / 0.783 vs 0.467 / 0.541 (T, H4); LinES train+dev 1.000 / 0.840
vs 0.667 / 0.628. Consensus rewards agreeing with the majority and cannot see errors all systems share; evidence carries
its own model biases (B). On fresh EWT dev the final evaluator wins within documents (0.802 vs 0.539) while consensus is
better between documents (0.919 vs 0.876) — the two are complementary.

**D. Domain shift.** Reliability is highest on GUM-like text (GUM, GENTLE) and holds on fresh web text (EWT dev), but the
GUM-derived improvements do not transfer to LinES (test and fresh train+dev): its conventions invert the blind spots
(unpunctuated in-paragraph boundaries; splits after abbreviations), and the layout-trained evaluator must not be used on
text without layout. The evaluator ranks SaT first on every corpus, so dynamic selection has no headroom (§5.1), but the
system ranking is useful as such (C).
- **Conclusion of 5.3:** reliability depends less on the formula than on whether the evidence sees the errors that
  actually separate the candidates — and every added cue (casing, layout) moves the blind spots rather than removing them.

## 6 Discussion and limitations (0.8 page)
### 6.1 Main findings
1. Reference-free quality estimation for sentence segmentation is feasible on real systems, and its reliability varies by domain.
2. Better boundary evidence — heterogeneous sources, casing, layout, quality-oriented training — substantially improves it;
   systematic boundary errors remain the main bottleneck.
3. Evidence-based scoring and system consensus carry complementary information: consensus knows which documents are
   hard, evidence knows which system is better; each new cue moves the evidence's blind spots rather than removing them.
One sentence: the improved evidence used directly as a segmenter matches or beats SaT (App. G) — evaluating and
segmenting meet when the evidence is strong.
### 6.2 Limitations
- Dynamic selection is not shown to beat Always-SaT (no headroom on our corpora).
- The improvements do not transfer to LinES (test and fresh train+dev); the layout evaluator is unreliable on text
  without layout and on unpunctuated boundaries inside paragraphs.
- Soft scoring does not beat a hard pseudo-reference once evidence is strong; the document-end correction was found post
  hoc (then confirmed on a fresh test).
- Small test sets (GENTLE 26 docs, LinES 9 + 18), four English systems, UD conventions; idealized layout; EWT dev is a
  fresh set of documents but the same domain as the EWT test.
- Training-data overlap (SaT-sm, Stanza with GUM/EWT; shared GUM supervision of our evidence and SaT-sm; GENTLE follows GUM conventions).
- The design was adaptive (twelve dated amendments, pre-registered); numbers on re-used tests are exploratory (R) —
  the fresh one-shot tests (EWT dev, LinES train+dev) are the clean evidence for the final evaluator.
### 6.3 Future work
Evidence for hard cases (unpunctuated boundaries without layout cues, abbreviations); calibration across annotation
conventions; domains where the best splitter actually changes, so that selection can be tested.

## 7 Conclusion (0.2 page)
We developed and evaluated a reference-free quality estimator for sentence segmentation. Stronger, heterogeneous
boundary evidence substantially improved quality ranking on held-out corpora — on fresh web text the final evaluator
reaches a Spearman of 0.85 and ranks the segmenters correctly, well above system consensus — while systematic error
patterns and different annotation conventions limit its reliability. Comparisons with hard pseudo-references and
consensus show that effective evaluation depends less on the scoring function than on the accuracy, coverage and blind
spots of the boundary evidence.

## End matter
Team contributions (who did what); code and data availability (repository, commit, `scripts/*.slurm`).

---

## Appendix plan (not counted in the 8 pages)
- **A. Preliminary exploration.** Synthetic perturbations on GUM dev; Qwen vs BERT; BCE vs Soft F1; early calibration
  (P2 search), candidate-conditioned context, paragraph-[SEP] control.
- **B. Implementation details.** Probe and evidence stack; ModernBERT tagger training; rules and n-gram features; casing and
  layout variants; normalization derivation (top-k closed form) and the gradient of the quality loss.
- **C. Additional experiments.** Full perturbation results (7 types × 4 rates, three corpora); error pricing; gold as a
  candidate; evidence ablations and same-family check; Dawid–Skene and HSMM variants; trivial baselines; Pk / WindowDiff /
  B-cubed robustness; document length.
- **D. Robustness and exploratory analyses.** Bootstrap sensitivity, leave-one-genre-out, seed variance; case × layout
  factorial (both scenarios, four corpora); SaT design analysis.
- **E. Quality-oriented training.** All variants (incl. ranking loss), nested cross-fitting, LOSO, boundary metrics, R-corpus tables.
- **F. Selection.** Macro / per-document routing regret vs Always-SaT, DIS, oracle headroom.
- **G. Evidence as a segmenter.** Comparisons with SaT-3l-sm, SaT-12l, SaT-12l-sm.
- **H. Reproducibility and protocol.** Data releases and checksums; experimental chronology (amendments 1–12, test-reuse
  rules, tie-exact erratum); the amendment-12 hypotheses table; code and configuration; job ids.

## Figures and tables (summary)
| Slot | Content | Source | Priority |
|---|---|---|---|
| F1 | Evaluator overview (+ gold-isolation box) | drawn | 1 |
| F4 | Perturbation test: detection heatmap (one injected error) + slope panel, error type × evaluator version | `results/a11/report.json` (perturbation) | 2 |
| F3 | Evidence quality vs reliability | `results/a9/layout_report.json`, `results/a10`, `results/a12` | 3 |
| F5 | Between vs within decomposition (final evaluator, M0, DocAgree; three corpora) | `results/a6`, `results/a9`, `results/a10` | 4 |
| F2 | Protocol timeline D → T → R (optional / panel of F1) | drawn | 5 |
| T1 | Data and systems | manifests | — |
| T2 | Evaluator versions × corpora | above | — |
| T3 | Quality-oriented training vs post-hoc QE | `results/a10` | — |
| T4 | System-ranking agreement (genre-mean / document-mean Kendall τ): final vs Probe vs consensus, GUM / GENTLE / EWT dev / LinES train+dev | `results/a11/report.json`, `results/a12/report.json` | — |
All plots from one script `paper/figures/make_figures.py` reading the JSON results (to write).

## Notation (keep consistent in all math blocks)
X document; i gap index, n gaps; y candidate (0/1 per gap); g gold; p evidence; x_i features; θ fusion weights; Q Soft F1;
Q̃ normalized Soft F1 (document end counted); F1(y, g) true quality; ρ Spearman; LOSO = leave one segmenter out.

## Open decisions
1. F2 separate or panel of F1.
2. Team-size / contribution statement; division of writing (suggested: Weifeng §3–5 and appendix; Yanhan §1–2, §5.3-B
   examples, F1 drawing, slides; both §6–7).
3. Next step: `paper/main.tex` (ACL template, compiled on Overleaf), starting with §3–4 (notation and formulas).
