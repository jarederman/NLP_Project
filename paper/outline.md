# Paper outline (v5.4, 2026-11-20)

**Structure (since v5):** the final evaluator is the main character; the main text compares it with the evaluators a
reader would naturally try (heuristics, language models, an off-the-shelf segmenter turned into an evaluator, system
consensus) and reports, next to agreement with true quality, a **controlled sensitivity test** with a quality-neutral
control. Our own earlier versions are in the appendix.
**v5.2 (fact revision after an external review):** the training objective is stated as implemented; flat text is the
default reference-free setting and paragraph layout a controlled "structure given" condition; T/R/P status is stated per
corpus for the final evaluator; the SaT soft-vs-hard claim rests on a controlled comparison; claims are worded to what
the statistics support; T2/T3 in the main text are trimmed (full tables in the appendix).
**v5.3:** the document-end claim is stated exactly (non-degenerate, not "scores 1"); the GUM agreement of the two
scoring forms is given with measured numbers; the control is described as a procedural sanity check; §5.1 reports key
paired differences with CIs instead of blanket statements. The structure is final; next step is writing. v1–v5.2 are in
git history.
**v5.4 (code review, 2026-11-20):** numbers updated after the review fixes (Dawid–Skene leave-one-out refit, LOGO PuncEv on
GUM, Q_end in every perturbation table, tie-aware top-1); main-text numbers changed only for B9 (EWT dev +0.179 → +0.183).
GUM LOGO is not nested for the base features: stated in §4.4 with the measured size (App. H).

Format: ACL template, **8 pages main text**; references and appendix do not count. Markers: **[M#]** math to write
(planned, not written); **[F#]** figure; **[T#]** table. Status: **D** = GUM development (leave-one-genre-out);
**T** = pre-registered one-shot test; **R** = re-used test; **P** = post-hoc analysis. All numbers: tie-exact, scoring
Q̃_end, checked against `results/a12`, `results/a13` on 2026-11-19. `paper/draft.md` is the superseded Chinese draft of
2026-11-14 and is not the basis of the paper.

---

## 0 For the team (not part of the paper)

### 0.1 The story in five sentences
1. **Problem.** On a new domain there are no gold boundaries, so nobody can tell how good a segmenter's output is;
   reference-free quality estimation for sentence segmentation is underexplored, and so is how to evaluate it.
2. **Method.** The evaluator predicts a boundary probability at every gap (fusing course techniques with pretrained
   neural boundary models) and scores a segmentation by its normalized Soft F1 against these probabilities.
3. **How we test it.** Three views on held-out splits: agreement with true quality on real outputs of four segmenters,
   agreement on the ranking of the segmenters, and a controlled sensitivity test with a quality-neutral control; under
   gold isolation, with pre-registered one-shot tests of the final evaluator.
4. **Results.** On the fresh EWT dev split (plain text) it reaches Spearman 0.797 with true quality — far above
   heuristics, language-model perplexity and system consensus (0.657), and on par with the strongest supervised
   segmenter plugged into the same scoring (0.762, n.s.); given paragraph structure, 0.851. The two evaluators have
   different blind spots.
5. **What limits it.** After normalization and the document-end correction, mainly the evidence: every evaluator misses
   some boundaries without terminal punctuation and some false splits after abbreviations, paragraph cues move this blind
   spot, consensus cannot see errors that all systems share, and different annotation conventions (LinES) break the
   transfer; selection cannot beat always using SaT.

### 0.2 Requirement map (proposal / progress report / course → paper)

| Requirement | Answer | Where |
|---|---|---|
| Proposal question: select a better splitter than Always-SaT? | Not shown: SaT is the best system on every corpus (no headroom). The evaluator ranks the systems correctly, which is the useful part of selection. | §5.1, §5.3, App. F |
| Required baselines: Always-SaT, Direct Internal Splitter | Reported (selection); DIS = our evidence used as a segmenter (App. G). | §5.3, App. F–G |
| Held-out evaluation (develop on GUM, freeze, test once) | Earlier frozen versions tested once on EWT test, GENTLE, LinES test; the final evaluator tested once on new splits EWT dev and LinES train+dev (T). | §4.4, T2 |
| Supporting outcomes: within-document pairwise, domain-level ranking | Reported for all evaluators. | §5.1, App. C |
| Course methods | Regex rules (L2), Kneser–Ney n-gram (L5), logistic regression / MaxEnt (L6–L7), token-level sequence labeling (L7), P/R/F (L6). | §3.3 |
| Related work ≥ 5 published NLP papers; limitations; presentation | §2 (≈15); §6.2; slides from F1, F2, T2, T3. | — |
| Team-size / contribution statement | To confirm with the instructor (two members). | end matter |

### 0.3 Main text vs appendix
Rule: main text = the evaluator, the baselines, the three evaluation views, and the analysis needed to explain them.

| Material | Main text | Appendix |
|---|---|---|
| Final evaluator (evidence, fusion, scoring Q̃_end) | §3 (method, F1, F2) | B (details, hyperparameters) |
| Evaluators we tried on the way (PuncEv-only, Probe-only, HSMM length prior, QE regression, LLM pilot) | §3.5, one paragraph | A, B |
| Development ladder and controlled training-objective comparison (amendment 10) | one sentence each (§3.4, §5.3) | B |
| Baselines B1–B9 and the agreement table (amendment 13) | **§4.2, §5.1, T2 (six representative rows)** | C (all nine, all corpora, CIs) |
| Controlled sensitivity with control (amendments 11, 13) | **§5.2, T3 (six rows, six columns)** | D (all methods, types, severities, corpora; slopes; z) |
| One-shot tests and hypotheses (amendments 4, 6, 12) | status labels; §4.4, §5.1 | H |
| Scoring-formula analysis (normalization, document end, controlled soft vs hard) | §5.3, three sentences | C |
| Evidence vs consensus decomposition; same-family bias; unfitted ensemble | §5.1 / §5.3, one sentence each | C |
| Case × layout factorial, evidence by left context, oracle ceiling | one sentence (§5.2) | E |
| Selection regret | §5.3, one paragraph | F |
| SaT analysis, evidence as a segmenter | one sentence (§6) | G |
| Protocol history (13 amendments), tie-exact erratum, reproducibility | footnote §4.4 | H |

### 0.4 Page budget (8 pages)
§1 0.9 · §2 0.6 · §3 1.4 · §4 1.2 · §5 2.8 · §6 0.8 · §7 0.2. Floats: F1 (full width), F2 (one column, worked example —
priority), T1 (small), T2 (one column, 6 × 6), T3 (one column, 6 × 6). F3 (scatter) only if space remains.

---

## Title and abstract
**Title:** *Cross-Domain Reference-Free Quality Estimation for Sentence Segmentation*

**Abstract (draft, ~170 words).** Sentence segmentation quality is hard to measure in a new domain, where gold
boundaries are missing, and reference-free quality estimation for segmentation is underexplored. We build an evaluator
that predicts a boundary probability at every candidate position, combining regular-expression rules, an n-gram model, a
probe on a frozen BERT and pretrained neural boundary models, and scores a segmentation by its normalized Soft F1
against these probabilities. We evaluate it on outputs of Punkt, Stanza, WtP and SaT on held-out English splits, against
simple and adapted baselines, with three views: agreement with true quality, agreement on the ranking of segmenters,
and controlled error injection with a quality-neutral control. On a fresh split of web text the evaluator reaches a
Spearman correlation of 0.80 with true quality (0.85 when paragraph structure is given), well above heuristics,
language-model perplexity and system consensus (0.66), and comparable to a strong supervised segmenter used as an
evaluator. Error injection shows the remaining limits: boundaries without terminal punctuation, false splits after
abbreviations, and annotation conventions that differ from the training corpus.

---

## 1 Introduction (0.9 page)
*Why this project, what problem it solves, what we found.*
### 1.1 Motivation and problem
- Sentence segmentation is a basic step whose errors propagate (parsing, MT, retrieval); segmenters (Punkt, Stanza, WtP,
  SaT) behave differently across domains, but a new domain has no gold boundaries.
- Goal: given raw text and a segmenter's output, estimate the output's quality without the target domain's gold.
  **Splitting** predicts boundaries; **quality estimation** judges an existing segmentation. We do the second; source-
  corpus gold (GUM) may be used for training, target gold only to validate.
- Boundary F1, Pk and WindowDiff need a reference; reference-free QE is established for MT but underexplored for
  sentence segmentation, so we also have to say how to evaluate such an evaluator (§4).
### 1.2 Approach (F1)
Boundary evidence per gap → normalized Soft F1 per candidate. The evidence model fuses course techniques with
pretrained neural boundary models and is trained so that its scores track true quality.
### 1.3 Research questions and contributions
- **RQ1** Can a reference-free score rank real segmentation outputs by their true quality on held-out data, and how
  does it compare with simple and adapted evaluators? (§5.1)
- **RQ2** Which segmentation errors does it notice, and what limits its reliability? (§5.2–§5.3)

**Contributions.**
1. **Method** (§3): a reference-free quality estimator for sentence segmentation that fuses probabilistic boundary
   evidence (rules, Kneser–Ney n-gram, MaxEnt probe on frozen BERT, zero-shot SaT, a fine-tuned binary boundary tagger)
   and scores an existing segmentation by normalized Soft F1 with a document-end correction.
2. **Evaluation** (§4): a systematic evaluation that checks agreement with true quality, the ranking of segmenters, and
   sensitivity to specific error types (with a quality-neutral control), against rules, language models, a strong
   segmenter used as pseudo-reference or as evidence, and multi-system consensus; gold isolation and pre-registered
   one-shot tests of the final evaluator on new held-out splits.
3. **Findings** (§5): on the fresh EWT dev split the evaluator reaches Spearman 0.797 (+0.141 [+0.075, +0.213] over
   consensus; 0.851 given paragraph structure), comparable to a strong supervised segmenter used as an evaluator
   (+0.035, n.s.); controlled error injection shows complementary blind spots and the limits of transfer across
   annotation conventions.
→ *§2: related work.*

## 2 Related work (0.6 page)
### 2.1 Sentence segmentation
Punkt (Kiss & Strunk 2006); Stanza (Qi et al. 2020); WtP (Minixhofer et al. 2023); SaT (Frohmann et al. 2024).
### 2.2 Evaluating without a reference, and evaluating evaluators
Reference-based segmentation metrics (P/R/F; Pk, Beeferman et al. 1999; WindowDiff, Pevzner & Hearst 2002); MT quality
estimation and metric meta-evaluation by correlation with human judgments (Specia et al. 2018; Rei et al. 2020;
WMT metrics tasks); pseudo-references and consensus (Albrecht & Hwa 2007; Louis & Nenkova 2013); aggregation of noisy
annotators (Dawid & Skene 1979); behavioral / perturbation tests of metrics (Ribeiro et al. 2020, CheckList; Sai et al.
2021 — **verify**); reference-free metrics favor their own underlying models (Deutsch et al. 2022); reference-free
evaluation of topic segmentation (**verify** citation). Difference: we evaluate sentence-boundary quality of existing
splitter outputs. (Run a proper literature search before claiming novelty; the text says "underexplored", not "first".)
→ *§3.*

## 3 Method (1.4 pages)
*What the evaluator does; F1 + two formulas + F2 should be enough to understand it.*

**[F1] Evaluator overview.** Raw document → gaps → evidence sources → fusion → p_i; candidate segmentations of four
segmenters → normalized Soft F1 → score; dashed box = never sees target gold.

**[F2] Worked example (one column; priority over any statistics figure).** A 3–4 sentence snippet (e.g. from EWT dev)
with its gaps marked, a bar for p_i at each gap, two candidates (one missing an unpunctuated boundary, one with a false
split after "Dr."), their Q̃ scores and their true F1. A reader sees in five seconds how a score is computed.

### 3.1 Task
Input: document X with gaps i = 1..n; a candidate y ∈ {0,1}^n from a segmenter. Output: a score s(X, y) computed without
target gold. Validity: how well s orders candidates by true quality F1(y, g). **[M1]**
### 3.2 Scoring a candidate
**[M2]** Q(y, p) = 2(Σ y_i p_i + 1) / (Σ y_i + Σ p_i + 2) — a soft boundary F1 in which the evidence p plays the
reference and the document end (always a boundary) is counted; **Q̃ = Q / max_y' Q(y', p)** (closed form: the best y'
cuts at the k most probable gaps, k = 0..n). Why: comparable across documents (the attainable maximum varies); and
counting the document-final boundary prevents a degenerate score for candidates without internal boundaries — without
it the correct "no split" of a one-sentence document always scores 0; with it, it scores 2/(Σp + 2) relative to the
best candidate (e.g. 0.95 when two internal gaps have p = 0.4; 1 when the evidence puts little mass inside). Q̃ = 1: as
good as the evidence allows.
### 3.3 Boundary evidence
Small inline table: source → what it captures → course topic.
- Regular-expression rules (abbreviations, initials, numbering, quotes) — L2.
- Kneser–Ney trigram features (P(`</s>` | left context)) — L5.
- MaxEnt probe on frozen `bert-base-uncased` `[MASK]` states + insertion likelihood ratio — L6/L7. **[M3]** p_i = σ(w·φ(h_i) + b).
- Zero-shot SaT-12l boundary probability (pretrained segmenter, not trained by us).
- Fine-tuned ModernBERT-large **binary boundary tagger** (one logit per word, BCE; token-level sequence labeling — L7).
- Paragraph-break flag in the layout condition (§4.1).
- Casing: all sources read lower-cased text except zero-shot SaT-12l, which reads the original casing in the flat
  evaluator (the layout evaluator is lower-cased throughout).
### 3.4 Fusion trained for quality
**[M4]** p_i = σ(θ·x_i); θ minimizes L = L_Q/L_Q⁰ + L_BCE/L_BCE⁰ + λ‖θ‖², with L_Q = mean over training candidates of
(Q̃⁰(y, p_θ) − F1(y, g))² and L_BCE the boundary log loss (each term scaled by its value at the BCE solution; gradients
pass through the top-k sort). **As implemented, the training objective uses the normalized Soft F1 without the
document-end term (Q̃⁰); evaluation uses Q̃ with it (§3.2).** On GUM, where the fusion is trained, no document is a
single sentence; the two scores differ per item by 0.001–0.002 on average (at most 0.04) and give the same global
Spearman to three decimals (0.903 flat, 0.948 with paragraphs). We keep the frozen models rather than retrain. Intuition:
the boundary loss treats all gaps alike; the quality loss focuses on the gaps that decide which candidate is better.
One sentence: in a controlled comparison on GUM this beats boundary-only training and post-hoc QE regression, which
overfits to the four seen systems (App. B).
### 3.5 What we tried on the way (one paragraph)
PuncEv (masked-LM punctuation probability) and the frozen probe alone (our first evaluators, tested once on EWT test),
an HSMM length prior, post-hoc QE regression and an LLM pilot; one clause each on why kept or dropped (App. A–B).
→ *§4: how to judge such scores.*

## 4 Evaluating an evaluator (1.2 pages)
### 4.1 Data, segmenters and input conditions
**[T1]** GUM train (D: training and development, 197 docs, 15 genres). Held-out splits: EWT test (316 docs), GENTLE (26,
8 genres), LinES test (9) — used to test earlier frozen versions, hence re-used (R) for the final evaluator; **EWT dev**
(318 docs, 5 genres) and **LinES train+dev** (18) — new splits, never used before the final evaluator's one-shot test (T).
They are new documents, not new domains (same sources as EWT test and LinES test). Candidates: Punkt, Stanza, WtP,
SaT-3l-sm → items = document × segmenter. Contamination: SaT-sm and Stanza saw GUM/EWT training data; GENTLE and LinES
are unseen by all.
**Input conditions.** *Flat text* — the default, fully reference-free setting. *Paragraph layout* — a controlled
condition in which paragraph breaks from UD `# newpar` are given (as newlines) to segmenters and evaluators; in UD every
paragraph break is a sentence boundary, so this condition supplies part of the gold structure and is reported as
"structure given", not as raw text.
### 4.2 Baselines (what a reader would try)
B1 punctuation rule (pseudo-reference: cut after every terminal punctuation); B2 sentence-length prior; B3 masked LM
(BERT probability of terminal punctuation, Platt-calibrated) with our scoring; B4 the strongest available segmenter,
SaT-12l-sm, **its own segmentation as pseudo-reference**; B5 **its calibrated probabilities with our scoring**;
B6 multi-system consensus; B7 DocAgree (document difficulty only); B8 **GPT-2 perplexity** (mean token log-likelihood of
the text written one sentence per line); B9 **Dawid–Skene** aggregation of the four segmenters (leave-one-out posterior,
our scoring). In the layout condition baselines also see the paragraph breaks. SaT-12l-sm was trained on GUM train.
### 4.3 Three views
- **V1 Agreement with true quality** **[M5]**: global Spearman ρ between score and true F1 over all items; within-
  document pairwise accuracy.
- **V2 System ranking** **[M6]**: Kendall τ between the evaluator's and the true ranking of the four segmenters, averaged
  over genres (and documents).
- **V3 Controlled sensitivity** **[M7]**: inject exactly one error of a type into the gold segmentation; **detection
  rate** = probability that the score drops (ties ½). **Control**: move one false boundary to another random non-boundary
  position — true quality unchanged. Because the two positions are exchangeable, every evaluator sits at 0.5 in
  expectation, so the control is a sanity check of the procedure (no directional bias, correct tie handling), not
  evidence that the responses to injected errors are driven by quality; how large a response is compared with the
  reaction to such neutral moves is reported as a standardized z in App. D. "Detected" = 95% CI above 0.5. Why V3: V1 can
  be high without any sensitivity (B7 never reacts to any error yet has ρ = 0.51–0.77).
### 4.4 Protocol and status of the numbers
Gold isolation (scores frozen before gold is loaded); all method decisions on GUM by leave-one-genre-out, written down
before results; genre-stratified document bootstrap (1000) for CIs. **Final evaluator:** GUM = D; GENTLE, LinES test,
EWT test = R; EWT dev, LinES train+dev = T (pre-registered hypotheses; primary: final evaluator > consensus in the layout
condition). **Baselines and all comparisons with them, and the sensitivity test, are post hoc (P)** — computed after the
one-shot test without changing the final evaluator; baselines are fixed or fitted on GUM only. Footnotes: tie-exact
ranking; GUM cross-fitting is leave-one-genre-out for the fused model but not nested for its learned base features (the
probe, tagger and n-gram features of the training genres were fitted with the held-out genre) — refitting the probe and
n-gram features nested changes the final evaluator's GUM ρ by +0.0005 / +0.0007 (n.s.), so the GUM numbers are not
inflated; test-time models never see test data either way (App. H). Protocol history in App. H.
→ *§5.*

## 5 Results (2.8 pages)
### 5.1 RQ1: does the score track true quality?
**[T2] Global Spearman with true F1** (flat text unless marked; representative baselines — all nine with CIs in App. C).

| Evaluator | EWT dev (T) | LinES tr+dev (T) | GENTLE (R) | EWT test (R) | LinES test (R) | EWT dev, paragraphs given (T) |
|---|---:|---:|---:|---:|---:|---:|
| B1 Punctuation rule | 0.378 | 0.296 | 0.383 | 0.369 | 0.437 | 0.796 |
| B8 GPT-2 perplexity | 0.062 | 0.049 | 0.064 | 0.043 | −0.322 | 0.039 |
| B6 Consensus | 0.657 | 0.468 | 0.830 | 0.657 | 0.861 | 0.625 |
| B4 SaT-12l-sm segmentation as reference | 0.738 | **0.630** | 0.880 | 0.754 | **0.888** | 0.797 |
| B5 SaT-12l-sm probabilities, our scoring | 0.762 | **0.630** | 0.891 | 0.778 | 0.873 | 0.835 |
| **Ours** | **0.797** | 0.563 | **0.907** | **0.787** | 0.873 | **0.851** |

(Status letters refer to Ours; baseline rows are P. GUM (D) and B2/B3/B7/B9 in App. C.)

Text:
- **Key paired differences (ours − baseline, 95% CI).** EWT dev, flat: consensus +0.141 [+0.075, +0.213], Dawid–Skene
  +0.183 [+0.117, +0.253], masked LM +0.333 [+0.256, +0.414], rule +0.419 [+0.334, +0.503], GPT-2 +0.735 [+0.640, +0.838].
  EWT dev with paragraphs given: consensus +0.227 [+0.158, +0.298] (the pre-registered primary test), rule +0.055
  [+0.033, +0.083] (the rule becomes strong, 0.796, once paragraph breaks are given). GENTLE: consensus +0.077 [+0.021,
  +0.141]. LinES train+dev: consensus +0.095 [+0.007, +0.210], masked LM +0.124 [−0.061, +0.369] (n.s.); LinES test:
  consensus +0.011 (n.s.). So ours is clearly better than simple, classic and consensus-based evaluators on EWT and
  GENTLE, and only partly on LinES.
- GPT-2 perplexity separates candidates within a document (pairwise 0.71) but not across documents: its likelihood
  mostly reflects what a document is about — the confound that normalization removes from Soft F1.
- **Against the strong segmenter**: no significant difference from SaT-12l-sm probabilities in our scoring on any
  held-out split (EWT dev +0.035 [−0.014, +0.088]; GENTLE +0.015 [−0.035, +0.064]; LinES train+dev −0.066 [−0.236, +0.095]
  — there SaT is ahead, 0.630 vs 0.563); better than SaT's own segmentation as reference on web text (EWT dev +0.060
  [+0.008, +0.112]), not elsewhere.
- One sentence each (App. C): system ranking — ours and B5 rank the four segmenters almost perfectly (EWT dev genre-mean
  τ 1.0 given paragraphs), consensus does not (0.47); SaT-based evaluators slightly overrate the SaT candidate (rank
  residual +0.02 to +0.03); an unfitted average of ours and B5 is best on web text (EWT dev 0.808) but not on LinES.
- LinES: no evaluator transfers well (≤ 0.63 on train+dev); our improvements over the initial probe do not hold there
  (pre-registered H5) — §5.3.
→ *Spearman says how well the score tracks quality overall; which errors does it actually notice?*

### 5.2 RQ2a: which errors does each evaluator notice?
**[T3] Detection rate of one injected error** (EWT dev; * = 95% CI above 0.5; the control should be 0.50; all methods,
all error types, severities and corpora in App. D).

| Evaluator | Merge at terminal punct. (easy) | Merge, no punct., at paragraph break | Merge, no punct., inside paragraph | False split after abbrev./numbering | False split at comma (easy) | Control |
|---|---:|---:|---:|---:|---:|---:|
| B1 Punctuation rule | 1.00* | 0.01 | 0.01 | 0.00 | 0.97* | 0.50 |
| B8 GPT-2 perplexity | 0.82* | 0.86* | **0.89*** | 0.53 | 0.87* | 0.49 |
| B6 Consensus | 1.00* | 0.52 | 0.51 | 0.55 | 0.97* | 0.50 |
| B5 SaT-12l-sm probabilities | 0.96* | 0.85* | 0.64* | **0.75*** | 0.99* | 0.49 |
| **Ours (flat)** | 0.99* | **0.93*** | 0.84* | 0.66* | 0.97* | 0.50 |
| Ours (paragraphs given) | 0.99* | 1.00* | 0.31 | 0.69* | 1.00* | 0.49 |

Text:
- Errors marked by punctuation are noticed by every evaluator that reads the text (the control sits at 0.50 for all, as
  the procedure requires).
- Two error types separate the evaluators: **missing boundaries without terminal punctuation** and **false splits after
  abbreviations/numbering**. The rule ignores the first entirely; consensus (and Dawid–Skene) is near chance on both —
  aggregating systems cannot reveal an error that most of them make. (One-sided heuristics in App. D: the length prior
  catches every merge and no split.)
- Ours (flat) and GPT-2 are the most sensitive to unpunctuated merges; SaT-12l-sm probabilities are the most sensitive
  to post-abbreviation splits (0.75 vs 0.66 ours, 0.53 GPT-2) — complementary blind spots.
- Given paragraph breaks, our evaluator catches every merge at a paragraph break but misses most inside paragraphs
  (0.31): it has learned "unpunctuated boundary ≈ newline". Same on GUM and GENTLE (App. D; factorial in App. E).
- DocAgree (App. D) shows why V1 alone is not enough: ρ = 0.51 on EWT dev, detection 0.50 everywhere.
→ *What determines how reliable an evaluator is?*

### 5.3 RQ2b: what limits reliability?
- **The formula matters in two places** (App. C): normalization across documents (+0.24 for PuncEv, +0.05 for ours on
  GUM; GPT-2 shows what happens without it) and the document-end term (single-sentence documents: +0.348 [+0.251, +0.449]
  on EWT dev, T). With these in place, soft vs hard scoring of the *same* probabilities matters little for our evidence
  (EWT dev +0.005 / +0.013) and helps SaT probabilities on web text (+0.025 [+0.002, +0.053] flat; n.s. on GENTLE and
  LinES) — a controlled comparison with the same calibration and document-end convention (App. C).
- **Then mostly the evidence** (one sentence + App. B): along our development each evidence component added agreement
  (GUM 0.57 → 0.90 flat); our evidence, which combines course techniques with pretrained neural models, is comparable to
  the best supervised segmenter's probabilities, and neither covers both hard error types (T3).
- **Consensus vs evidence** (one sentence + App. C): consensus is good at *which document is hard* (between-document ρ
  0.92) and poor at *which segmentation is better* (within-document 0.54 vs ours 0.80, EWT dev with paragraphs).
- **Domain shift**: annotation conventions set the ceiling — on LinES (both splits) all evaluators fall to ≤ 0.63 and
  post-abbreviation splits are detected at 0.2–0.4 (App. D).
- **Selection** (one paragraph, App. F): SaT is the best segmenter on every corpus; no evaluator selects better than
  always using it — ranking ability (§5.1) does not translate into a selection gain without headroom.
- **Conclusion of §5:** once scores are normalized and the document end is counted, an evaluator is only as reliable as
  its evidence on the gaps that separate the candidates; adding cues moves its blind spots rather than removing them.

## 6 Discussion and limitations (0.8 page)
### 6.1 Main findings
1. Reference-free quality estimation for sentence segmentation is feasible: with good evidence and a normalized,
   document-end-aware score, it tracks true quality on held-out splits and ranks segmenters correctly.
2. Our fused evidence (course techniques + pretrained neural models) is comparable to a strong off-the-shelf segmenter
   used as an evaluator; they fail on different error types, and their unfitted average helps on web text.
3. Controlled error injection is needed next to correlation: it shows what each evaluator cannot see (consensus: shared
   errors; perplexity: cross-document calibration and abbreviations; paragraph-trained evidence: in-paragraph boundaries).
One sentence: our evidence used directly as a segmenter matches or beats SaT-3l-sm (App. G).
### 6.2 Limitations
- English only; four segmenters; UD annotation. The layout condition takes paragraph breaks from UD, where they are
  sentence boundaries by construction — results with layout are "structure given" and optimistic.
- Small held-out sets (GENTLE 26 docs, LinES 9 + 18); the new splits are new documents, not new domains.
- Baseline comparisons and the sensitivity test are post hoc; only the final evaluator's scores on the new splits are
  one-shot tests, and the final evaluator's numbers on GENTLE / LinES test / EWT test are re-used tests.
- The quality-training objective omits the document-end term used at evaluation (harmless on GUM, but not unified).
- Training-data overlap: SaT-sm and Stanza with GUM/EWT; our evidence and SaT-sm both supervised on GUM; GENTLE follows
  GUM conventions.
- Adaptive design (13 dated amendments, pre-registered); the document-end correction was found post hoc and then
  confirmed on a new split. An independent code review found five issues affecting reported numbers (App. H); after the fixes the
  pre-registered selections and all one-shot hypotheses (amendment 12) are unchanged.
- Selection not shown to beat Always-SaT; transfer to different annotation conventions (LinES) fails.
### 6.3 Future work
Evidence for unpunctuated boundaries without layout; adaptation to annotation conventions; combining evidence sources
with different blind spots; a unified training objective; domains where the best segmenter changes.

## 7 Conclusion (0.2 page)
We built and evaluated a reference-free quality estimator for sentence segmentation. On held-out splits it tracks true
quality far better than heuristics, language-model perplexity or system consensus and is comparable to a strong
supervised segmenter used as an evaluator, while controlled error injection shows which errors it misses and why
annotation conventions limit its transfer.

## End matter
Team contributions; code and data availability (repository, commit, `scripts/*.slurm`).

---

## Appendix plan (not counted in the 8 pages)
- **A. Preliminary exploration.** Synthetic perturbations on GUM dev; Qwen vs BERT; BCE vs Soft F1; early calibration.
- **B. Method details and development.** Probe, rules, n-gram, taggers, hyperparameters; the development ladder (Probe →
  stack → +casing → +layout → +quality training, all corpora, both conditions); the controlled training-objective
  comparison (boundary loss vs quality loss vs post-hoc QE, leave-one-segmenter-out); the top-k maximum and the gradient
  of the quality loss; the objective-vs-evaluation scoring note.
- **C. Agreement, full.** All nine baselines × all corpora (incl. GUM) × both conditions with CIs; within-document
  pairwise; system-ranking τ; rank residuals (same-family bias); unfitted ensemble; normalized vs raw scoring; the
  controlled soft-vs-hard comparison (`results/a13/softhard.md`); between/within decomposition.
- **D. Sensitivity, full.** All methods and error types, all severities (1 error … 40% of boundaries), slopes, z-scores
  (with their caveats), real-error pricing; GUM / GENTLE / LinES train+dev.
- **E. Blind-spot diagnosis.** Evidence by left context; oracle ceiling; case × layout factorial.
- **F. Selection.** Regret vs Always-SaT, Direct Internal Splitter, oracle headroom.
- **G. SaT analysis and evidence as a segmenter.**
- **H. Protocol and reproducibility.** Amendment timeline 1–13, test-reuse rules, one-shot hypotheses (amendment 12),
  tie-exact erratum, code-review erratum (findings, fixes, before/after numbers, the nested-LOGO check:
  `results/review_fixes/`), data releases, scripts, job ids.

## Figures and tables
| Slot | Content | Source |
|---|---|---|
| F1 | Evaluator overview (+ gold-isolation box) | drawn |
| F2 | Worked example: snippet, p_i bars, two candidates, Q̃ vs true F1 | `results/a11/examples.md` (pick), evidence caches |
| T1 | Data, splits, status | manifests |
| T2 | Agreement (V1): six representative evaluators × six columns | `results/a13/report.json`, `classic.json` |
| T3 | Sensitivity (V3): six evaluators × six columns on EWT dev | `results/a13/report.json`, `classic.json` |
| F3 (if space) | Scatter of score vs true F1 on EWT dev: consensus / B5 / ours | `results/a12`, `results/a13` |
All plots from one script `paper/figures/make_figures.py` (to write).

## Notation
X document; i gap, n gaps; y candidate (0/1 per gap); g gold; p evidence; x_i features; θ fusion weights; Q Soft F1
(document end counted); Q̃ normalized; Q̃⁰ normalized without the end term (training objective only); F1(y, g) true
quality; ρ Spearman; τ Kendall.

## Open decisions
1. F3 in or out (page budget).
2. Team-size / contribution statement; division of writing (suggested: Weifeng §3–5 and appendix; Yanhan §1–2, F1/F2
   drawings, examples, slides; both §6–7).
3. Next step: `paper/main.tex` (ACL template, Overleaf), starting with §3–4.
