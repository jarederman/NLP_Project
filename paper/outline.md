# Paper outline (v3, 2026-11-19)

v3 adopts the framing **"we built a working reference-free evaluator, made it substantially more reliable, and analyzed
what determines its reliability and where its limits are"** (not "the method failed, then we looked for reasons"). It
adds an explicit research-questions-and-contributions block (§1.3), merges discussion and limitations (§6), and lists
candidate experiments to add to the main text (§0.5). v1/v2 are in git history.

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
   by its normalized Soft F1 against these probabilities — no target-domain gold is needed.
3. **Experiments.** We score the outputs of four real segmenters (Punkt, Stanza, WtP, SaT) on four English corpora under a
   gold-isolated protocol with one-shot held-out tests, and compare against hard pseudo-references and system consensus.
4. **Results.** The evaluator works and can be made much more reliable: stronger, heterogeneous boundary evidence raises
   agreement with true quality from 0.67 to 0.88 on the held-out GENTLE test (T), and casing, paragraph layout and
   quality-oriented training raise it further (GUM 0.574 → 0.903 on flat text, 0.948 with layout; GENTLE 0.91, R); it
   identifies the best segmenter on every corpus.
5. **What determines reliability.** The boundary evidence, not the scoring formula: all evidence shares a blind spot on
   boundaries without terminal punctuation; evidence and consensus capture complementary information; under domain
   shift (web text, different annotation conventions) reliability drops, and selection cannot beat always using SaT.

### 0.2 Requirement map (proposal / progress report / course → where the paper answers)

| Requirement | Answer | Where |
|---|---|---|
| Proposal question: can a reference-free evaluator select a better splitter than Always-SaT (Macro Selection Regret)? | Not shown: SaT is the best system on every corpus (near-oracle), so there is almost no headroom. The evaluator *does* identify SaT as best (system-level ranking), which is the useful part of the selection question. | §5.1, §5.3-D, App. F |
| Required baselines: Always-SaT, Direct Internal Splitter (DIS) | Both reported. The first DIS (F1 0.831) is below every system; the case + layout evidence used as DIS later beats SaT-3l-sm on all four corpora (P/R). | §5.1, §6.1, App. G |
| Held-out evaluation: develop on GUM, freeze, test once on EWT | Done (T); extended to GENTLE and LinES (T). | §4.3, T2 |
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

### 0.4 Page budget (8 pages)
§1 0.9 · §2 0.6 · §3 1.5 · §4 1.0 · §5 3.0 · §6 0.8 · §7 0.2. Floats: F1 (full width, top of p. 2–3), F3–F5 (one column
each), T1 (small), T2 (full width), T3 (one column). F2 optional (merge into F1 as panel b if space is short).

### 0.5 Candidate experiments to add to the main text (decision needed)
Each item names the main-text claim it supports. "Post-hoc" items reuse frozen scores and need no new test data.

| # | Experiment | Supports | Cost | Recommendation |
|---|---|---|---|---|
| E1 | **System-level ranking and best-system identification** per corpus and per genre, with CIs, for Probe, Stack, final evaluator, consensus (preliminary check: final evaluator ranks SaT first on all four corpora; per-genre best system found in GUM 13–14/15, GENTLE 6–7/8, EWT 3/5 flat, 5/5 with layout) | §5.1/§5.3-D: the evaluator is useful for "which splitter is best here?" even though dynamic selection cannot beat SaT | CPU, < 1 h, post-hoc | **Do** |
| E2 | **Soft vs hard pseudo-reference and normalized vs raw Soft F1 for the final evaluator** (currently only for the amendment-6 stack) | §5.3-A: does the scoring formula matter once evidence is strong, and does quality-oriented training change that? | CPU, ~1 h (a10 code + two scorers) | **Do** |
| E3 | **Error pricing on real outputs for each evaluator version** (missed unpunctuated boundaries: rewarded at 16% of their true value under the amendment-6 stack — how much under the final evaluator?) | §5.3-B: connects the synthetic blind-spot test (F4) to real system errors | CPU, < 1 h, post-hoc | **Do** |
| E4 | **Discordant-example mining** for the error analysis (items where the final evaluator and gold disagree most, with gap contexts, auto-tagged by error type) | §5.3-B/§6: 2–3 real examples instead of invented ones | CPU, < 1 h; team picks examples | **Do** |
| E5 | **Clean one-shot test of the final evaluator on a fresh corpus** (pre-registered; candidates: UD English PUD or ParTUT if they have document structure, or EWT dev; GUM test only if the team lifts the "never read GUM test" rule) | §5.2: today every number for the final evaluator outside GUM is R; one clean T result would make the main claim much stronger | CPU fetch + GPU ~1 h + CPU analysis; needs a dataset check first | **Strongly recommended** if time allows |
| E6 | Evidence + consensus combination (fixed rank average, decided on GUM LOGO, checked with leave-one-segmenter-out) | §5.3-C: turns "complementary" into a measured gain (target: EWT, where consensus wins) | CPU, ~1 h; pre-registration | Optional |
| — | More evidence models (LLM evidence, LoRA adaptation, larger SaT as evidence) | — | GPU, days | Not now (scope) |

---

## Title and abstract

**Title (proposal title):** *Cross-Domain Reference-Free Quality Estimation for Sentence Segmentation*

**Abstract (draft, ~170 words; five sentences of §0.1, no list of model versions).** Sentence segmentation quality is
hard to measure in a new domain because gold boundaries are missing. We develop a reference-free evaluator that predicts
a boundary probability at every candidate position and scores a segmentation by its normalized Soft F1 against these
probabilities. We evaluate the outputs of Punkt, Stanza, WtP and SaT on four English corpora under a gold-isolated
protocol with one-shot held-out tests, and compare with hard pseudo-references and system consensus. Combining
heterogeneous boundary evidence substantially improves agreement with true quality (Spearman 0.67 → 0.88 on the held-out
GENTLE test), and casing, paragraph layout and quality-oriented training improve it further; the evaluator also
identifies the best segmenter on every corpus. Our analysis shows that reliability is determined by the boundary
evidence rather than the scoring formula, that evidence and consensus are complementary, and that a shared blind spot —
boundaries without terminal punctuation — and domain shift remain the main limits.

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
   four real segmenters on four English corpora without target gold, built from course methods (regular expressions,
   n-gram LM, MaxEnt, BIO tagging), together with a gold-isolated, pre-registered protocol with one-shot held-out tests
   (§3–§4, §5.1).
2. **A substantially more reliable evaluator.** Heterogeneous boundary evidence (+0.212 [+0.101, +0.326] Spearman over the
   initial probe on the GENTLE one-shot test), casing and paragraph layout, and a quality-oriented training objective that
   improves agreement and holds for segmenters unseen in training, unlike post-hoc quality regression (GUM 0.574 → 0.903
   on flat text, 0.948 with paragraph layout) (§3.4, §5.2).
3. **An analysis of what makes it reliable.** The scoring formula matters only through normalization; evidence quality
   drives reliability; ~110k controlled perturbations locate a shared blind spot (unpunctuated boundaries, abbreviations)
   that explains rating biases; evidence and consensus measure different things (within- vs between-document); and we
   report where reliability ends (web text, different annotation conventions, splitter selection) (§5.3, §6).
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
**[M2]** Soft F1 Q(y, p) = 2 Σ y_i p_i / (Σ y_i + Σ p_i); **normalized** Q̃ = Q / max_y' Q(y', p) (closed form: top-k
prefix of the sorted p). Why: Q is a soft boundary F1 with p as a probabilistic reference; the per-document maximum
varies (0.60–0.94) and would otherwise confound comparisons across documents; Q̃ = 1 means "as good as the evidence allows".

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
docs); GENTLE test (T: main held-out evaluation of the improved evaluator, 26 docs, 8 genres); LinES test (T: secondary,
9 docs; no paragraph layout). Candidates: Punkt, Stanza, WtP (wtp-bert-mini), SaT (sat-3l-sm) → items = document × system.
Two input conditions: flat text, and text with paragraph layout (UD `# newpar` as `\n`, seen by systems and evaluator;
idealized). Contamination note: SaT-sm and Stanza saw GUM/EWT training data; GENTLE and LinES are unseen by all systems.
### 4.2 Evaluation metrics
Ground truth = human-annotated UD sentence boundaries; quality of a candidate = its boundary F1. **[M5]** Primary: global
Spearman ρ between Q̃ and true F1 over all items (higher = better ranking across documents and systems). Supporting:
within-document pairwise accuracy; between/within-document decomposition; system-level ranking (best system per corpus
and per genre) and selection regret vs Always-SaT; boundary F1 of the evidence itself; error-type sensitivity (**[M6]**
perturbation slope, ideal 1). Footnote: identical candidates are tied exactly before ranking (erratum, App. H).
### 4.3 Protocol
LOGO development and all decisions on GUM, written down before results; freeze; score the test corpus before its gold
is loaded (gold isolation); one test per frozen evaluator (EWT for the Probe; GENTLE and LinES for the stack); genre-
stratified document bootstrap (1000) for CIs and paired differences. Later improvements are decided on GUM only and
reported on the used test sets as R (exploratory). **[F2, optional]** timeline D → T → R.
→ *Link to §5.*

## 5 Results and analysis (3.0 pages)
*Three consecutive questions: does it work → can we make it more reliable → what determines its reliability.*

### 5.1 Does the initial evaluator work on real splitters?
- One sentence: a synthetic-perturbation pilot showed BERT + Soft F1 is promising (App. A), but real systems are the test.
- Initial evaluator on real outputs: Probe beats PuncEv on the EWT one-shot test (0.411 vs 0.338, +0.074 [+0.023,
  +0.126], T); on GUM 0.574; GENTLE 0.672 and LinES 0.868 (T).
- The original selection goal is not met: SaT is the best system on every corpus, Always-SaT is near-oracle (GUM regret
  0.0014), and the initial evaluator's selection is worse than Always-SaT (−0.026 [−0.037, −0.021]); details App. F.
- **Conclusion of 5.1:** reference-free evaluation of real splitters is feasible, but the initial evaluator's
  reliability is moderate → can stronger evidence make it more reliable?

### 5.2 Can stronger boundary evidence make the evaluator more reliable?
**[T2] Global Spearman by evaluator version × corpus** (best non-baseline per column bold).

*Flat text*

| Evaluator | GUM (D) | EWT | GENTLE | LinES |
|---|---:|---:|---:|---:|
| Initial: Probe | 0.574 | 0.411 T | 0.672 T | 0.868 T |
| Evidence stack | 0.818 | 0.516 R | 0.883 T | 0.818 T |
| + casing | 0.877 | **0.531** R | 0.903 R | 0.840 R |
| + quality-oriented training (final, flat) | **0.903** | 0.522 R | **0.910** R | **0.873** R |
| *System consensus M0* | 0.733 | 0.657 | 0.830 T | 0.861 T |
| *Multi-system QE M3* | 0.861 | — | 0.900 T | 0.793 T |
| *DocAgree / punctuation rule / length prior* | 0.637 / 0.340 / 0.048 | 0.529 / 0.392 / — | 0.773 / 0.383 / 0.037 | 0.741 / 0.437 / 0.079 |

*Text with paragraph layout (systems and evaluator see paragraph breaks; LinES has none)*

| Evaluator | GUM (D) | EWT (R) | GENTLE (R) |
|---|---:|---:|---:|
| Evidence stack | 0.844 | 0.526 | 0.853 |
| + casing | 0.898 | 0.545 | 0.876 |
| + layout feature | 0.939 | 0.570 | 0.890 |
| + quality-oriented training (final, layout) | **0.948** | **0.572** | **0.910** |
| *System consensus M0 / DocAgree* | 0.675 / 0.475 | 0.591 / 0.482 | 0.795 / 0.747 |

Text:
- **Clean test:** the evidence stack improves over the Probe on GENTLE, +0.212 [+0.101, +0.326] (T), robust to
  unstratified bootstrap and to leaving out any GENTLE genre; the stack is stable across four fine-tuning seeds on GUM (App. D). On LinES it does not improve
  (−0.050 [−0.178, +0.081], n.s.): "no improvement shown", not "significant degradation".
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
  probabilities (held-out log loss ×3–5); quality + boundary loss improves and transfers (log loss unchanged). On the
  re-used tests the gain is smaller (GENTLE +0.007 / +0.020, EWT ≈ 0).
- Against baselines: the evaluator beats consensus on GUM and GENTLE; consensus is better on EWT (→ §5.3-C).
- **Conclusion of 5.2:** stronger evidence makes the evaluator substantially more reliable, with a clean held-out gain on
  GENTLE, but the gain is not guaranteed on every new domain → what determines reliability?

### 5.3 What determines evaluation reliability?
**A. The scoring formula (soft vs hard, normalized vs raw).** Normalization is essential (+0.235 [+0.181, +0.286] over raw
Soft F1, same evidence). Beyond it, soft scoring ≈ hard pseudo-reference when evidence is strong (GENTLE 0.883 vs 0.896,
n.s.; GUM +0.008 n.s.) — the formula is not where reliability comes from. [E2 would add the final evaluator here.]
**[F3] Evidence quality vs reliability:** one point per evidence/evaluator version (boundary F1 of the evidence or held-out
log loss vs global Spearman, GUM and GENTLE): reliability rises with evidence quality.

**B. Evidence quality and error types.** **[F4] Blind-spot test** (~110k controlled perturbations on GUM): slope of the
score on true F1 by error type (ideal 1) for PuncEv, Probe, Stack, + casing, + layout, final. All evidence handles merges
at terminal punctuation (slope ≈ 1) but is insensitive to **missing boundaries without terminal punctuation** (Stack 0.24;
PuncEv −0.23 rewards the error) and to **false splits after abbreviations/numbering**; casing and layout repair the first
(0.24 → 0.46 → 0.72 → 0.75 final). On real outputs, Punkt's errors concentrate in this blind spot, so it is overrated
(rank residual +0.067 on GUM; its missed unpunctuated boundaries are rewarded at only 16% of their true value). Two or
three real examples (from E4). Takeaway: overall Spearman can be high while specific error types are mis-priced.

**C. Evidence vs consensus.** **[F5] Between/within-document decomposition.** Consensus is good at telling which
*documents* are hard (between-document ρ ≈ 0.89–0.92, like DocAgree) but weak within a document (0.37–0.58), because it
cannot see errors all systems share; evidence adds within-document discrimination but carries its own model biases. On
EWT (web text, many one-sentence documents, little within-document signal) consensus wins on flat text (0.657 vs 0.531;
significant for the flat evaluators); with layout the gap shrinks (0.591 vs 0.570–0.572; n.s. for lower-nl). The two are complementary.

**D. Domain shift and system-level usefulness.** The evaluator identifies the best segmenter (SaT) on every corpus and the
best segmenter in most genres (GUM 13–14/15, GENTLE 6–7/8, EWT 3/5 flat, 5/5 with layout; E1 for CIs) — useful for "which
splitter should I use here?", although dynamic selection cannot beat Always-SaT because SaT is always best. Reliability
drops under domain shift: LinES annotation conventions invert the blind spot (negative slopes), and the layout-trained
evaluator must not be used on text without layout (it picks the wrong system on LinES).
- **Conclusion of 5.3:** reliability depends less on overall boundary accuracy or the formula than on whether the
  evidence sees the errors that actually separate the candidates.

## 6 Discussion and limitations (0.8 page)
### 6.1 Main findings
1. Reference-free quality estimation for sentence segmentation is feasible on real systems, and its reliability varies by domain.
2. Better boundary evidence — heterogeneous sources, casing, layout, quality-oriented training — substantially improves it;
   systematic boundary errors remain the main bottleneck.
3. Evidence-based scoring and system consensus carry complementary information; neither is best everywhere.
One sentence: the improved evidence used directly as a segmenter matches or beats SaT (App. G) — evaluating and
segmenting meet when the evidence is strong.
### 6.2 Limitations
- Dynamic selection is not shown to beat Always-SaT (no headroom on our corpora).
- LinES does not reproduce the GENTLE improvement; EWT favors consensus.
- Soft F1 does not beat a hard pseudo-reference once evidence is strong.
- Small test sets (GENTLE 26 docs, LinES 9), four English systems, UD conventions; idealized layout.
- Training-data overlap (SaT-sm, Stanza with GUM/EWT; shared GUM supervision of our evidence and SaT-sm; GENTLE follows GUM conventions).
- Final-evaluator numbers on re-used tests are exploratory (R); the design was adaptive (ten dated amendments, pre-registered).
### 6.3 Future work
Evidence for hard cases (unpunctuated boundaries without layout cues, abbreviations); calibration across annotation
conventions; domains where the best splitter actually changes, so that selection can be tested.

## 7 Conclusion (0.2 page)
We developed and evaluated a reference-free quality estimator for sentence segmentation. Stronger, heterogeneous
boundary evidence substantially improved quality ranking on the held-out GENTLE corpus, and casing, layout and
quality-oriented training improved it further; systematic error patterns and domain differences limit its reliability.
Comparisons with hard pseudo-references and consensus show that effective evaluation depends less on the scoring
function than on the accuracy, coverage and blind spots of the boundary evidence.

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
- **H. Reproducibility.** Data releases and checksums; experimental chronology (amendments 1–10, test-reuse rules,
  tie-exact erratum); code and configuration; job ids.

## Figures and tables (summary)
| Slot | Content | Source | Priority |
|---|---|---|---|
| F1 | Evaluator overview (+ gold-isolation box) | drawn | 1 |
| F4 | Blind-spot slopes by error type × evaluator version | `results/a8`, `results/a9/layout_report.json`, `results/a10/report_T3.json` | 2 |
| F3 | Evidence quality vs reliability | `results/a9/layout_report.json`, `results/a10` | 3 |
| F5 | Between vs within decomposition (final evaluator, M0, DocAgree; three corpora) | `results/a6`, `results/a9`, `results/a10` | 4 |
| F2 | Protocol timeline D → T → R (optional / panel of F1) | drawn | 5 |
| T1 | Data and systems | manifests | — |
| T2 | Evaluator versions × corpora | above | — |
| T3 | Quality-oriented training vs post-hoc QE | `results/a10` | — |
All plots from one script `paper/figures/make_figures.py` reading the JSON results (to write).

## Notation (keep consistent in all math blocks)
X document; i gap index, n gaps; y candidate (0/1 per gap); g gold; p evidence; x_i features; θ fusion weights; Q Soft F1;
Q̃ normalized Soft F1; F1(y, g) true quality; ρ Spearman; LOSO = leave one segmenter out.

## Open decisions
1. Which of E1–E6 to run (§0.5).
2. F2 separate or panel of F1.
3. Team-size / contribution statement; division of writing (suggested: Weifeng §3–5 and appendix; Yanhan §1–2, §5.3-B
   examples, F1 drawing, slides; both §6–7).
4. Next step: `paper/main.tex` (ACL template, compiled on Overleaf), starting with §3–4 (notation and formulas).
