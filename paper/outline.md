# Paper outline (v2, 2026-11-19)

Replaces v1 (2026-11-17, in git history), which followed the amendment log. v2 is organized around **one evaluator and
three questions about it**; the amendment history moves to the appendix. Paper language: English. Format: ACL template,
**8 pages main text**; references and appendix do not count.

Markers used below: **[M#]** = a math block to write (planned, not written yet); **[F#]** = figure slot; **[T#]** = table
slot; **D** = GUM-train development (leave-one-genre-out, LOGO); **T** = pre-registered one-shot test (scores frozen before
gold was materialized); **R** = re-used test set (reported only, never used for a decision). All numbers are tie-exact
(erratum 2026-11-19) and were checked against `results/` on 2026-11-19.

---

## 0 For the team (not part of the paper)

### 0.1 The story in five sentences (the spine every section must serve)
1. **Problem.** On a new domain there is no gold segmentation, so we cannot tell which sentence segmenter output is good;
   we want a *reference-free* score that ranks segmentation outputs by their true quality.
2. **Method.** Our evaluator predicts, at every whitespace gap, the probability of a sentence boundary from several
   evidence sources (course methods: regex rules, n-gram LM, MaxEnt probe, BIO tagger), fuses them, and scores a
   candidate segmentation by its *normalized Soft F1* against these probabilities.
3. **How we test it.** Gold-isolated, pre-registered protocol: develop on GUM (LOGO), freeze, test once on held-out
   corpora (EWT, GENTLE, LinES); validity = rank agreement (global Spearman) between score and true F1 across documents.
4. **Findings.** It works well when the evidence is good (GUM 0.94, GENTLE 0.88 T / 0.91 R), and the evidence is the
   bottleneck: all evidence shares a blind spot on boundaries without terminal punctuation; casing and paragraph layout
   repair most of it, and training the fusion for quality (not only boundary accuracy) adds a gain that transfers to
   unseen systems.
5. **Limits.** It does not beat "always use SaT" for choosing a segmenter, it is weaker than multi-system consensus on
   web text (EWT), and it fails on LinES, whose annotation conventions differ.

### 0.2 Requirement map (proposal / progress report / course → where the paper answers)

| Requirement | Answer | Where |
|---|---|---|
| Proposal question: can a reference-free evaluator select a splitter better than Always-SaT (Macro Selection Regret)? | No; SaT is near-oracle (GUM Macro regret 0.0014), and no method beats it. This is why validity, not selection, is the main target. | §1 (one paragraph), §5.5, App. F |
| Required baselines: Always-SaT, Direct Internal Splitter (DIS) | Both reported; the first DIS (F1 0.831) is below every system, while the case + layout evidence used as DIS later beats SaT-3l-sm on all four corpora (P/R). | §5.5, §6, App. G |
| Held-out evaluation (develop on GUM, freeze, test once on EWT) | Done (T); extended to GENTLE and LinES (T). | §4.3, T2 |
| Within-document pairwise accuracy, domain-level ranking | Reported for all methods. | T2 (main), App. C (full) |
| Course: apply course methods | Regex rules, KN-smoothed n-gram, MaxEnt/logistic regression, BIO tagging, P/R/F, (B-cubed, HMM length prior in appendix). | §3.2 (one sentence + table column) |
| Course: related work, honest limitations, presentation | §2, §7, slides from F1/T2/F4. | — |
| Team-size / contribution statement | **To confirm with the instructor** (we are two). | end matter |

### 0.3 Main text vs appendix: where every experiment goes
Rule: an experiment is in the main text only if a section's question cannot be answered without it. Everything else
gets **at most one sentence** in the main text and its full content in the appendix.

| Experiment (repo label) | Main text | Appendix |
|---|---|---|
| Qwen vs BERT pilot on synthetic candidates (preliminary) | 1 sentence in §3.2 (why a BERT-family encoder) | B |
| Real-splitter selection v1, train CV, oracle headroom (proposal question) | §1 one paragraph, §5.5 one paragraph | F |
| Evidence by left context + oracle-substitution ceiling | 1 sentence in §5.3 ("the evaluator only needs to be right where systems disagree") | E |
| P2 calibration search (casing features help) | 1 sentence in §5.3 (first sign of the casing effect) | C |
| Normalization vs raw Soft F1 (amendment 2) | §5.2, one number | C |
| Candidate-conditioned context; paragraph-[SEP] control (amendments 2–3) | none | C |
| Probe (amendment 3), EWT one-shot test (amendment 4) | Probe = evidence source in §3.2; EWT = T column of T2 | A, C |
| Stack + GENTLE/LinES one-shot tests (amendment 6) | Stack row of T2 (T) | A, C |
| Dawid–Skene M1/M2, HSMM length prior, QE regressor M3 | M3 in T2 (one row); others none | C |
| Robustness: unstratified bootstrap, leave-one-genre-out, 4 seeds (amendment 7) | 1 sentence in §5.1 | D |
| Gold as a 5th candidate (8.1) | 1 sentence in §5.3 | D |
| Controlled perturbations (8.2) | **§5.3, F4** | D |
| Error pricing on real outputs (8.3) | §5.3, 1–2 sentences (Punkt overrating) | D |
| Selection regret with final methods (8.4) | §5.5 | F |
| Trivial baselines (8.5) | rows of T2 | C |
| Leave-one-evidence-out ablation, same-family check (8.6) | §5.2, 1 sentence | D |
| Document length (8.7) | none | D |
| SaT design analysis; casing trial (9.1) | casing rows of T2 | G |
| Case × layout factorial (9.2+9.3) | layout block of T2 + 2 sentences in §5.3 | E |
| Evidence as a segmenter vs SaT-3l / 12l / 12l-sm | §6, 1 sentence | G |
| Quality-oriented fusion vs post-hoc QE (amendment 10) | **§5.4, T3** | H |
| Tie-exact erratum | footnote in §4.2 | A |

### 0.4 Page budget (8 pages)
§1 1.0 · §2 0.5 · §3 1.6 · §4 1.0 · §5 2.9 · §6 0.5 · §7 0.35 · §8 0.15. Floats: F1 (½ column wide or full width),
F3–F5 (one column each), T1 (small), T2 (full width), T3 (one column). F2 is optional (merge into F1 as panel b if space is short).

---

## Title and abstract

**Title (keeps the proposal title):** *Cross-Domain Reference-Free Quality Estimation for Sentence Segmentation*
(alt.: *Scoring Sentence Segmentations Without a Reference: Learned Boundary Evidence and Its Blind Spots*).

**Abstract (draft, ~170 words).** Sentence segmenters are evaluated against gold boundaries, which a new domain does not
have. We build a reference-free evaluator: it estimates a boundary probability at every whitespace gap from several
evidence sources — regular-expression rules, a Kneser–Ney n-gram model, a MaxEnt probe on a frozen BERT, zero-shot SaT and
a fine-tuned BIO tagger — fuses them, and scores a candidate segmentation by its normalized Soft F1 against these
probabilities. We test whether the score ranks segmentation quality across documents and domains under a gold-isolated,
pre-registered protocol with one-shot tests on three held-out English corpora. The evaluator agrees strongly with true
quality when its evidence is good (Spearman 0.94 on GUM, 0.88 on the GENTLE test), and evidence is the bottleneck: all
sources miss boundaries without terminal punctuation. Casing and paragraph layout repair most of this blind spot, and
training the fusion for quality estimation rather than boundary accuracy alone adds a gain that holds for unseen
segmenters. The evaluator does not beat always using the strongest segmenter for selection, trails multi-system
consensus on web text, and fails where annotation conventions differ.

---

## 1 Introduction (1.0 page)
*Purpose: say what we do, why, and what we found, before any detail.*
1. **Why it matters.** Segmentation errors propagate to parsing, MT and retrieval; on a new domain there is no gold, so
   practitioners cannot measure which segmenter output is good. Existing measures (boundary F1, Pk, WindowDiff) all need
   a reference; reference-free quality estimation exists for MT but not for segmentation.
2. **What we build** (one paragraph, pointing to F1): boundary evidence → fused probability per gap → normalized Soft F1
   per candidate → a ranking of candidate segmentations, without gold.
3. **The question we started with and why we changed it** (one paragraph, no tables): the proposal asked whether such a
   score can *select* a better splitter than always using SaT. It cannot, because SaT is already near-oracle on our data
   (§5.5); so we evaluate the evaluator directly: does its score track true quality across documents and domains?
4. **Questions** (each answered by one results subsection):
   - **Q1** Does the reference-free score agree with true segmentation quality on held-out domains? (§5.1)
   - **Q2** What determines how well it agrees — the scoring formula or the evidence? (§5.2–§5.3)
   - **Q3** Should the evidence fusion be trained for boundary accuracy or for quality estimation? (§5.4)
5. **Contributions** (three bullets):
   - **An evaluator** for reference-free segmentation quality built from course methods, with a quality-oriented
     training objective (BCE + differentiable normalized-Soft-F1 loss).
   - **A test bed** for such evaluators: gold isolation, pre-registered decisions, one-shot held-out tests,
     between/within-document decomposition, and controlled perturbations that expose blind spots.
   - **Findings** with numbers: evidence quality is the bottleneck; one shared blind spot explains rating biases and the
     LinES failure; casing + layout + quality-oriented fusion raise GUM agreement from 0.818 to 0.948; post-hoc QE
     regressors overfit to the seen segmenters; selection does not beat Always-SaT.
→ *Link to §2: position against segmentation and QE literature.*

## 2 Related work (0.5 page)
- Sentence boundary detection: Punkt (Kiss & Strunk 2006); Stanza (Qi et al. 2020); WtP (Minixhofer et al. 2023); SaT (Frohmann et al. 2024).
- Segmentation evaluation with a reference: boundary P/R/F; Pk (Beeferman et al. 1999); WindowDiff (Pevzner & Hearst 2002).
- Reference-free QE: MT QE (Specia et al. 2018; COMET/COMET-QE, Rei et al. 2020); pseudo-references and consensus
  (Albrecht & Hwa 2007; Louis & Nenkova 2013); noisy-annotator aggregation (Dawid & Skene 1979, appendix only).
- Probing and calibration: Tenney et al. 2019; Hewitt & Manning 2019; Platt 1999; Guo et al. 2017.
- Models and data: BERT; ModernBERT (Warner et al. 2024); Kneser–Ney (Kneser & Ney 1995; Chen & Goodman 1999); UD, GUM, EWT, GENTLE, LinES.
- Positioning (one sentence): QE usually scores content; we score structure, where the signal is a boundary probability
  and the dominant failure is boundaries the signal cannot see.
→ *Link to §3: define the object we score and the evaluator.*

## 3 The evaluator (1.6 pages)
*Purpose: everything needed to reproduce the score; F1 shows it at a glance.*

**[F1] Evaluator overview (architecture).** Left: raw document → whitespace gaps. Middle: evidence sources per gap
(rules, n-gram, BERT probe, SaT, ModernBERT tagger, layout flag) → fusion layer → p_i. Right: candidate segmentations
from four segmenters → normalized Soft F1 → ranking. A dashed box marks what never sees gold. Optional panel (b) = F2.

**3.1 Task.** A document is a sequence of gaps; a segmentation is a 0/1 vector over gaps; true quality is boundary F1
against gold. **[M1]** notation: document d, gaps i = 1..n_d, candidate y ∈ {0,1}^{n_d}, gold g, F1(y, g).
One sentence: text is rebuilt from UD tokens so that every system and the evaluator see the same raw string.

**3.2 Boundary evidence.** One short paragraph per source, in course terms; one table column "course topic".
- Regex rules (abbreviations, initials, numbering, quotes) — L2 regular expressions/FSA.
- Kneser–Ney trigram features (probability of `</s>` after the left context) — L5 n-gram LM and smoothing.
- MaxEnt probe on frozen `bert-base-uncased` `[MASK]` states (layers 8/12) + insertion likelihood ratio — L6/L7
  logistic regression/MaxEnt. **[M2a]** p_i = σ(w·φ(h_i) + b).
- Zero-shot SaT-12l boundary probability (Platt-calibrated) — external model.
- Fine-tuned ModernBERT-large BIO-style boundary tagger with punctuation dropout — L7 sequence labeling.
- Layout flag: the gap is a paragraph break (when the input has paragraphs).
- One sentence each: a pilot chose a bidirectional encoder over a left-to-right LM (App. B); evidence reads raw text,
  not the candidate (App. C).

**3.3 Fusion.** A logistic stack over the evidence features (logits of the sources, rule and n-gram features, layout
flag), cross-fitted by genre. **[M2b]** p_i = σ(θ·x_i). Two training objectives, compared in §5.4:
- boundary accuracy (standard): **[M4a]** L_BCE = mean over gaps of the logistic loss + L2;
- quality-oriented: **[M4b]** L_Q = mean over (d, candidate) of (Q̃(y, p_θ) − F1(y, g))², combined as
  L = L_Q/L_Q⁰ + L_BCE/L_BCE⁰ (+ L2), each term scaled by its value at the BCE solution; gradient flows through the
  top-k sort of [M3] (piecewise differentiable).

**3.4 Scoring a candidate.** **[M3]** Soft F1 Q(y, p) = 2 y·p / (|y|₁ + |p|₁); normalized Q̃ = Q / max_y' Q(y', p),
where the maximum is a top-k prefix of the sorted p (closed form). One sentence of motivation: the per-document maximum
varies (0.60–0.94) and confounds cross-document comparison; normalization removes it (§5.2).
→ *Link to §4: the evaluator outputs a score per (document, segmenter); §4 defines how we judge such scores.*

## 4 Evaluating the evaluator (1.0 page)
*Purpose: data, metrics, and why the numbers are trustworthy.*

**[T1] Data and segmenters.** GUM train (D, 197 docs, 15 genres, 788 items); EWT test (316 docs, 5 genres); GENTLE test
(26 docs, 8 genres); LinES test (9 docs). Items = document × {Punkt, Stanza, WtP (wtp-bert-mini), SaT (sat-3l-sm)}.
Columns: role, docs, items, genres, paragraph layout available (GUM/GENTLE/EWT yes, LinES no), status (D/T/R).
One sentence on contamination: SaT-sm and Stanza saw GUM/EWT training data; GENTLE and LinES are unseen by all systems.

**4.1 Two input conditions.** *Flat text* (paragraph breaks removed — the original protocol) and *text with paragraph
layout* (UD `# newpar` restored as `\n`; both segmenters and evaluator see it). Layout from UD is idealized → optimistic.

**4.2 Agreement metrics.** **[M5]** global Spearman ρ between score and true F1 over all items (primary); decomposition
into between-document (document means) and within-document (residuals) Spearman; within-document pairwise accuracy;
genre-stratified document bootstrap (1000) for CIs and paired differences. Footnote: ties between identical candidates
are made exact before ranking (erratum, App. A).

**4.3 Protocol.** Gold isolation (separate text / gold / candidate artifacts; scores written before gold is loaded);
all decisions on GUM LOGO and pre-registered in writing before results; one-shot tests: EWT for the Probe, GENTLE and
LinES for the Stack; later improvements are decided on GUM only and reported on the used test sets as R.
**[F2, optional]** protocol timeline: develop (D) → freeze → test once (T) → re-use for reporting only (R).

**4.4 Baselines.** Multi-system consensus M0 (agreement with the other three segmenters); DocAgree (mean pairwise
agreement of the four systems — measures only document difficulty); QE regression (S2/M3: ridge on score and
document features, + consensus for M3); punctuation-rule reference; sentence-length prior; for selection: Always-SaT,
Direct Internal Splitter.
→ *Link to §5: Q1 first (does it work?), then why (Q2), then how to train it (Q3).*

## 5 Results (2.9 pages)

### 5.1 Q1 — The score tracks true quality on held-out domains, when the evidence is good
**[T2] Main table: global Spearman by evaluator version × corpus** (status per cell; best non-baseline per column bold).

*Flat text (systems and evaluator without paragraph breaks)*

| Evaluator | GUM (D) | EWT | GENTLE | LinES |
|---|---:|---:|---:|---:|
| PuncEv (BERT `[MASK]` punctuation prob.) | 0.355 | 0.338 T | — | — |
| Probe | 0.574 | 0.411 T | 0.672 T | 0.868 T |
| Stack | 0.818 | 0.516 R | 0.883 T | 0.818 T |
| + casing | 0.877 | 0.531 R | 0.903 R | 0.840 R |
| + quality-oriented fusion (final, flat) | 0.903 | 0.522 R | 0.910 R | 0.873 R |
| *Consensus M0* | 0.733 | 0.657 | 0.830 T | 0.861 T |
| *DocAgree* | 0.637 | 0.529 | 0.773 | 0.741 |
| *QE regression M3* | 0.861 | — | 0.900 T | 0.793 T |
| *Punctuation rule / length prior* | 0.340 / 0.048 | 0.392 / — | 0.383 / 0.037 | 0.437 / 0.079 |

*Text with paragraph layout (systems and evaluator see paragraph breaks; LinES has none)*

| Evaluator | GUM (D) | EWT (R) | GENTLE (R) |
|---|---:|---:|---:|
| Stack | 0.844 | 0.526 | 0.853 |
| + casing | 0.898 | 0.545 | 0.876 |
| + layout feature | 0.939 | 0.570 | 0.890 |
| + quality-oriented fusion (final, layout) | **0.948** | **0.572** | **0.910** |
| *Consensus M0 / DocAgree* | 0.675 / 0.475 | 0.591 / 0.482 | 0.795 / 0.747 |

Text (one paragraph each):
- Clean tests (T): Probe beats PuncEv on EWT (+0.074 [+0.023, +0.126]); Stack beats Probe on GENTLE (+0.212 [+0.101,
  +0.326]) and loses on LinES (−0.050, n.s.) — the pre-registered test succeeded on GENTLE and failed on LinES.
- Later versions (decided on GUM only) keep improving on GENTLE (R) and GUM; the final evaluator reaches 0.948 (D).
- Against baselines: better than consensus on GUM and GENTLE; **consensus is better on EWT** (web text, many short
  one-sentence documents); DocAgree alone gets 0.53–0.77, i.e. part of the agreement is document difficulty (§5.3 end).
- Robustness in one sentence (App. D): the GENTLE gain holds with unstratified bootstrap, leaving out any genre, and four seeds.
→ *Why does it work on some corpora and not others? §5.2–5.3.*

### 5.2 Q2a — The formula matters little; the evidence matters a lot
- Normalization is needed: +0.235 [+0.181, +0.286] over raw Soft F1 with the same evidence (one sentence, [M3]).
- With strong evidence, soft scoring ≈ a hard pseudo-reference (Stack − H(Stack) +0.008, n.s.): beyond normalization,
  formula engineering has little left to give.
- **[F3] Evidence quality vs agreement.** x = evidence quality (boundary F1 of the evidence used as a segmenter, or
  held-out log loss), y = global Spearman; one point per evidence source / evaluator version on GUM (and GENTLE);
  optional curve from the oracle-shift experiment (App. E). Message: agreement rises almost monotonically with evidence quality.
- Ablation in one sentence (App. D): on GUM and GENTLE removing any evidence source hurts, SaT evidence most (−0.09 / −0.07); not on LinES.
→ *If evidence is the bottleneck, which gaps does it get wrong?*

### 5.3 Q2b — One blind spot, and how casing and layout repair it
- **[F4] Blind-spot test** (controlled perturbations, ~110k candidates on GUM): within-document slope of score on true F1
  (ideal 1) per error type — merge at terminal punctuation, merge without terminal punctuation, false split after an
  abbreviation, mixed — for PuncEv, Probe, Stack, +casing, +layout, final. **[M6]** slope definition.
  Key numbers: "merge without terminal punctuation" slope 0.24 (Stack) → 0.46 (+casing) → 0.72 (+layout) → 0.75 (final);
  PuncEv is negative (−0.23) — it rewards this error.
- Why it matters on real outputs (one or two sentences): Punkt's errors concentrate in this blind spot, so it is
  overrated (rank residual +0.067 on GUM) and SaT underrated; fixing Punkt's missed unpunctuated boundaries is rewarded
  at only 16% of its true value (App. D). The evaluator only needs to be right where systems disagree (oracle
  ceiling, App. E). Gold itself is rarely ranked first for the same reason (one sentence, App. D).
- Casing and layout are substitutes (two sentences): each alone gives a large gain; together the second adds little
  (case × layout interaction −0.059 [−0.078, −0.043] on GUM), because both reveal the same boundaries (most unpunctuated boundaries are
  paragraph breaks; the next word is capitalized). P2 had already shown casing features help (App. C). Full 2×2 in App. E.
- Evidence vs consensus, briefly ([F5]): between-document ρ is high for everything including DocAgree (≈0.83–0.96);
  evidence's added value is *within-document* ranking, where consensus is weak because it cannot see errors all systems
  share. This explains why consensus wins on EWT (many short documents, little within-document signal).
- LinES (one sentence, details §6): its conventions invert the blind spot (negative slopes), so better GUM-style evidence does not help.
→ *Evidence is improved; can we also train the fusion for the target we actually care about?*

### 5.4 Q3 — Train the fusion for quality, not only for boundary accuracy
Controlled comparison: same features, folds, items and Q̃; only the objective or a second stage changes.

**[T3] GUM LOGO, global Spearman (Δ vs current Stack [95% CI]; LOSO = Δ when the scored segmenter is unseen in training)**

| Method | Flat (ref91 features) | LOSO Δ | Layout (lower-nl features) | LOSO Δ |
|---|---:|---:|---:|---:|
| Stack (BCE) → Q̃ | 0.877 | — | 0.939 | — |
| Post-hoc ridge QE (S2, amendment 6) | 0.860 | −0.032 | 0.927 | −0.023 |
| Post-hoc F1 regressor (A) / residual QE (R) | 0.873 / 0.873 | −0.024 / −0.024 | 0.930 / 0.930 | −0.017 / −0.016 |
| Fusion trained on quality loss only | 0.849 | −0.066 | 0.933 | −0.007 |
| **Fusion trained on quality loss + BCE** | **0.903** (+0.026 [+0.014, +0.038]) | **+0.022** | **0.948** (+0.009 [+0.004, +0.015]) | **+0.010** |

Text:
- Post-hoc regressors do not help and get worse on an unseen segmenter: they learn the four systems, not quality.
- Quality loss alone under-constrains the probabilities (held-out log loss ×3–5) and generalizes worse.
- Quality + BCE improves agreement, keeps the boundary probabilities (log loss 0.0103 → 0.0107), improves the blind-spot
  slopes, and holds under LOSO → adopted (pre-registered rule). Adding a ranking loss gives nothing more (App. H).
- On re-used tests the gain is smaller (GENTLE +0.007 flat / +0.020 layout; EWT ≈ 0) — honest caveat.
→ *Finally, back to the question the project started with.*

### 5.5 Back to selection: a good evaluator still does not beat Always-SaT
One paragraph: on GUM, SaT's Macro selection regret is already 0.0014; the first evaluator lost to Always-SaT
(−0.026 [−0.037, −0.021]); with the final methods, corpus-level selection picks SaT (regret 0) and per-document routing
is never reliably better (one exception: M3 on GENTLE, P). With a near-oracle default, selection is the wrong target —
which is why the paper evaluates agreement. Full table App. F.
→ *§6 explains the failures linguistically.*

## 6 Error analysis and discussion (0.5 page)
- Three short examples (team to pick from GENTLE/EWT): an unpunctuated boundary (heading / list item / dialogue turn),
  a false split after an abbreviation or numbering, an EWT web-text case where consensus is right and evidence is wrong.
- LinES: annotation conventions (units without terminal punctuation; punctuation inside non-final units) → the blind spot inverts.
- Evaluating vs segmenting (one sentence): the case + layout evidence (cased-nl), used directly as a segmenter, beats
  SaT-3l-sm on all four corpora and the strongest SaT (12l-sm) on GENTLE and EWT (App. G) — a strong evaluator is close
  to a strong segmenter; both share GUM supervision.
- Practical recipe (one sentence): evidence for within-document ranking, system agreement for document difficulty.

## 7 Limitations (0.35 page)
- Small test sets (GENTLE 26 docs, LinES 9); the final evaluator's numbers on EWT/GENTLE/LinES are R, not clean tests.
- English only; four segmenters; UD conventions; layout from UD `# newpar` is idealized (optimistic).
- Shared GUM supervision with SaT-sm; GENTLE follows GUM conventions (favors both); contamination of Stanza/SaT on GUM/EWT.
- Adaptive design (ten dated amendments) mitigated by pre-registration and D/T/R labels; one measurement erratum (ties).
- Selection use case not achieved; consensus stronger on web text.

## 8 Conclusion (0.15 page)
Answer Q1–Q3 in three sentences + one sentence on selection; future work: evidence for unpunctuated boundaries without
layout, adapting to annotation conventions, combining evidence and consensus.

## End matter
Contribution statement (if required); code and data availability (repository, commit, `scripts/*.slurm`).

---

## Appendix plan (not counted in the 8 pages)
- **A. Protocol and history.** Amendment timeline 1–10 (date, decision, which results had been seen); test-reuse rules; tie-exact erratum.
- **B. Preliminary study.** Synthetic candidates on GUM dev; BERT vs Qwen; Soft F1 vs BCE.
- **C. Evidence and scorer variants.** P2 calibration search; candidate-conditioned context; paragraph-[SEP] control;
  normalization; Probe details; Dawid–Skene, HSMM length prior, M3; full per-corpus tables (Pk / WindowDiff / B-cubed robustness).
- **D. Validity checks.** Robustness (unstratified, leave-one-genre-out, seeds); gold as a candidate; full perturbation
  tables (7 types × 4 rates, three corpora); error pricing; leave-one-evidence-out ablation and same-family check; document length.
- **E. Blind-spot diagnosis.** Evidence by left context; oracle-substitution ceiling; full case × layout factorial (both scenarios, four corpora).
- **F. Selection.** Macro / per-document routing regret vs Always-SaT, DIS, oracle headroom.
- **G. SaT analysis and evidence as a segmenter.** Design of SaT, why it is strong; comparisons with SaT-3l-sm, 12l, 12l-sm.
- **H. Quality-oriented fusion details.** All variants (B-q, B-qr, B-qbr), nested cross-fitting, LOSO, boundary metrics, R-corpus tables.
- **I. Reproducibility.** Data releases and checksums, scripts, job ids, hardware.

## Figures and tables (summary)
| Slot | Content | Source | Priority |
|---|---|---|---|
| F1 | Evaluator architecture (+ gold-isolation box) | drawn | 1 |
| F4 | Blind-spot slopes by error type × evaluator version | `results/a8/report.json`, `results/a9/layout_report.json`, `results/a10/report_T3.json` | 2 |
| F3 | Evidence quality vs agreement scatter | `results/a9/layout_report.json` (evidence_as_segmenter, evaluator), `results/a10` | 3 |
| F5 | Between vs within decomposition: final evaluator, M0, DocAgree, three corpora | `results/a6`, `results/a9`, `results/a10` | 4 |
| F2 | Protocol timeline D → T → R (optional, or panel of F1) | drawn | 5 |
| T1 | Data and segmenters | manifests | — |
| T2 | Main progression table | above | — |
| T3 | Quality-oriented fusion (controlled) | `results/a10` | — |
Figures generated by one script `paper/figures/make_figures.py` from the JSON results (to write).

## Notation (keep consistent in all math blocks)
d document; i gap index, n_d gaps in d; y candidate (0/1 over gaps); g gold; p evidence (per-gap probability); x_i features;
θ fusion weights; Q Soft F1; Q̃ normalized Soft F1; F1(y, g) true quality; ρ Spearman; S = set of segmenters; LOSO leave one segmenter out.

## Open decisions
1. Title (proposal title vs alternative).
2. Whether F2 is a separate figure or a panel of F1.
3. Team-size / contribution statement.
4. Who writes what (suggested: Weifeng §3–5 and appendix; Yanhan §1–2, §6 examples, F1 drawing, slides; both §7–8).
5. Next step after this outline: write `paper/main.tex` (ACL template, compile on Overleaf) section by section.
