# Paper outline (v5.1, 2026-11-19)

**What changed from v4 (user feedback):** this is a new task with no prior evaluator, so the main text compares the
evaluator with the evaluators a reader would naturally try (simple heuristics, a language model, an off-the-shelf
segmenter turned into an evaluator, system consensus) — not with our own earlier versions, which move to the appendix.
Agreement with true quality (Spearman) is reported together with a **controlled sensitivity test** that shows which
error types each evaluator notices (with a quality-neutral control). The main text keeps only what a reader needs to see
that the idea is clear, the experiments are complete and the conclusions are careful; everything else is one sentence
plus an appendix. v5.1 adds two classic baselines (GPT-2 perplexity, Dawid–Skene). v1–v4 are in git history.

Format: ACL template, **8 pages main text**; references and appendix do not count. Markers: **[M#]** math to write
(planned, not written); **[F#]** figure; **[T#]** table. Status: **D** = GUM development (leave-one-genre-out);
**T** = pre-registered one-shot test; **R** = re-used test; **P** = post-hoc analysis. All numbers: tie-exact, scoring
Q̃_end, checked against `results/a12`, `results/a13` on 2026-11-19.

---

## 0 For the team (not part of the paper)

### 0.1 The story in five sentences
1. **Problem.** On a new domain there are no gold boundaries, so nobody can tell how good a segmenter's output is; there
   is also no established way to evaluate such a reference-free evaluator.
2. **Method.** The evaluator predicts a boundary probability at every gap (evidence fused from course methods and neural
   taggers, trained for quality estimation) and scores a segmentation by its normalized Soft F1 against these probabilities.
3. **How we test it.** Three views on held-out corpora: agreement with true quality on real outputs of four segmenters,
   agreement on the ranking of the segmenters, and a controlled sensitivity test with a quality-neutral control; all
   under gold isolation, with pre-registered one-shot tests on fresh data.
4. **Results.** It tracks true quality closely on held-out text (Spearman 0.80–0.91; 0.851 on fresh web text, T), far
   above heuristics, a masked LM, GPT-2 perplexity, system consensus and Dawid–Skene aggregation, and on par with the
   strongest supervised segmenter (SaT-12l-sm) plugged into the same scoring; the two have different blind spots.
5. **What limits it.** Evidence, not the formula: every evaluator misses some boundaries without terminal punctuation and
   some false splits after abbreviations, extra cues (layout) move rather than remove these blind spots, consensus cannot
   see errors that all systems share, and different annotation conventions (LinES) break the transfer; selection cannot
   beat always using SaT.

### 0.2 Requirement map (proposal / progress report / course → paper)

| Requirement | Answer | Where |
|---|---|---|
| Proposal question: select a better splitter than Always-SaT? | Not shown: SaT is the best system on every corpus (no headroom). The evaluator does rank the systems correctly, which is the useful part of selection. | §5.1, §5.3, App. F |
| Required baselines: Always-SaT, Direct Internal Splitter | Reported (selection); DIS = our evidence used as a segmenter (App. G). | §5.3, App. F–G |
| Held-out evaluation (develop on GUM, freeze, test once) | EWT test, GENTLE, LinES (T); final evaluator once more on fresh EWT dev and LinES train+dev (T). | §4.4, T2 |
| Supporting outcomes: within-document pairwise, domain-level ranking | Reported for all evaluators. | §5.1, App. C |
| Course methods | Regex rules (L2), Kneser–Ney n-gram (L5), logistic regression / MaxEnt (L6–L7), BIO tagging (L7), P/R/F (L6). | §3.3 |
| Related work ≥ 5 published NLP papers; limitations; presentation | §2 (≈15); §6.2; slides from F1, T2, T3. | — |
| Team-size / contribution statement | To confirm with the instructor (two members). | end matter |

### 0.3 Main text vs appendix
Rule: main text = the evaluator, the baselines, the three evaluation views, and the analysis needed to explain them.

| Material | Main text | Appendix |
|---|---|---|
| Final evaluator (evidence, fusion, scoring Q̃_end) | §3 (method, F1, F2) | B (details, hyperparameters) |
| Evaluators we tried on the way (PuncEv-only, Probe-only, HSMM length prior, Dawid–Skene, QE regression, LLM pilot) | §3.5, one paragraph | A, B |
| Development ladder (Probe → stack → +casing → +layout → +quality training) and controlled training-objective comparison (amendment 10) | one sentence each (§3.4, §5.3) | B |
| Baselines B1–B9 and the agreement table (amendment 13) | **§4.2, §5.1, T2** | C |
| Controlled sensitivity with control (amendments 11, 13) | **§5.2, T3** | D (all levels, slopes, z, real-error pricing, other corpora) |
| One-shot tests and hypotheses (amendments 4, 6, 12) | T2 status labels; §5.1 text | H |
| Scoring-formula analysis (normalization, document end, soft vs hard) | §5.3, three sentences | C |
| Evidence vs consensus decomposition; same-family bias; unfitted ensemble | §5.1 / §5.3, one sentence each | C |
| Case × layout factorial, evidence by left context, oracle ceiling | one sentence (§5.2) | E |
| Selection regret | §5.3, one paragraph | F |
| SaT analysis, evidence as a segmenter | one sentence (§6) | G |
| Protocol history (13 amendments), tie-exact erratum, reproducibility | footnote §4.4 | H |

### 0.4 Page budget (8 pages)
§1 0.9 · §2 0.6 · §3 1.4 · §4 1.2 · §5 2.8 · §6 0.8 · §7 0.2. Floats: F1 (full width), F2 (one column), F3 (one
column), T1 (small), T2 (full width), T3 (full width or one column, small font). F4 optional.

---

## Title and abstract
**Title:** *Cross-Domain Reference-Free Quality Estimation for Sentence Segmentation*

**Abstract (draft, ~170 words).** Sentence segmentation quality is hard to measure in a new domain: gold boundaries are
missing, and no reference-free evaluator exists. We build one: it predicts a boundary probability at every candidate
position, combining rules, an n-gram model, a probe on a frozen BERT and neural boundary taggers, and scores a
segmentation by its normalized Soft F1 against these probabilities. We evaluate it on the outputs of Punkt, Stanza, WtP
and SaT on held-out English corpora, against simple and adapted baselines, with three views: agreement with true
quality, agreement on the ranking of segmenters, and controlled error injection with a quality-neutral control. On fresh
web text the evaluator reaches a Spearman correlation of 0.85 with true quality, far above heuristics, a masked language
model, GPT-2 perplexity and system consensus (0.63), and on par with the strongest supervised segmenter plugged into the
same scoring.
Error injection shows what limits every evaluator: boundaries without terminal punctuation, false splits after
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
- This is a new task: boundary F1, Pk and WindowDiff need a reference; reference-free QE exists for MT, not here. So we
  also have to say how to evaluate such an evaluator (§4).
### 1.2 Approach (F1)
Boundary evidence per gap → normalized Soft F1 per candidate. The evidence model combines course methods and neural
taggers and is trained so that its scores track true quality.
### 1.3 Research questions and contributions
- **RQ1** Can a reference-free score rank real segmentation outputs by their true quality on new domains, and how does it
  compare with simple and adapted evaluators? (§5.1)
- **RQ2** Which segmentation errors does it notice, and what limits its reliability? (§5.2–§5.3)

**Contributions.**
1. **A reference-free evaluator for sentence segmentation** (§3): a scoring rule that turns boundary probabilities into a
   quality score (normalized Soft F1 counting the document end) and an evidence model built from course methods
   (regular expressions, Kneser–Ney n-gram, MaxEnt probe, BIO tagger) fused by logistic regression trained for quality.
2. **A way to evaluate such evaluators** (§4): nine natural baselines, three complementary views (agreement on real
   outputs, system ranking, controlled error injection with a quality-neutral control), gold isolation, and pre-registered
   one-shot tests including two fresh corpora.
3. **Findings** (§5): on held-out text the evaluator reaches Spearman 0.80–0.91 (0.851 on fresh EWT dev, T), +0.23 over
   consensus, +0.28–0.52 over heuristics and a masked LM on flat text (+0.05–0.20 with layout, where a punctuation +
   newline rule becomes strong), and ties with the strongest supervised segmenter in the same scoring, with complementary blind spots; every evaluator misses some unpunctuated boundaries and post-abbreviation
   splits; layout cues move this blind spot; consensus cannot see shared errors; transfer fails under different
   annotation conventions.
→ *§2: related work.*

## 2 Related work (0.6 page)
### 2.1 Sentence segmentation
Punkt (Kiss & Strunk 2006); Stanza (Qi et al. 2020); WtP (Minixhofer et al. 2023); SaT (Frohmann et al. 2024).
### 2.2 Evaluating without a reference, and evaluating evaluators
Reference-based segmentation metrics (P/R/F; Pk, Beeferman et al. 1999; WindowDiff, Pevzner & Hearst 2002); MT quality
estimation and metric meta-evaluation by correlation with human judgments (Specia et al. 2018; Rei et al. 2020;
WMT metrics tasks); pseudo-references and consensus (Albrecht & Hwa 2007; Louis & Nenkova 2013); behavioral /
perturbation tests of metrics (Ribeiro et al. 2020, CheckList; Sai et al. 2021, perturbation checklists for NLG metrics —
**verify**); reference-free metrics favor their own underlying models (Deutsch et al. 2022); reference-free evaluation of
topic segmentation (**verify** citation). Difference: we evaluate sentence-boundary quality of existing splitter outputs.
→ *§3.*

## 3 Method (1.4 pages)
*What the evaluator does; F1 + two formulas + F2 should be enough to understand it.*

**[F1] Evaluator overview.** Raw document → gaps → evidence sources → fusion → p_i; candidate segmentations of four
segmenters → normalized Soft F1 → score; dashed box = never sees target gold.

**[F2] Worked example (one column).** A 3–4 sentence snippet (e.g. from EWT dev) with its gaps marked, a bar for p_i at
each gap, two candidate segmentations (one missing an unpunctuated boundary, one with a false split after "Dr."), their
Q̃ scores and their true F1. Purpose: a reader sees in five seconds how a score is computed and why it tracks quality.

### 3.1 Task
Input: document X with gaps i = 1..n; a candidate y ∈ {0,1}^n from a segmenter. Output: a score s(X, y) computed without
target gold. Validity: how well s orders candidates by true quality F1(y, g). **[M1]**
### 3.2 Scoring a candidate
**[M2]** Q(y, p) = 2(Σ y_i p_i + 1) / (Σ y_i + Σ p_i + 2) — a soft boundary F1 in which the evidence p plays the
reference and the document end (always a boundary) is counted; **Q̃ = Q / max_y' Q(y', p)** (closed form: the best y'
cuts at the k most probable gaps, k = 0..n). Why: comparable across documents (the attainable maximum varies), and the
correct "no split" of a one-sentence document scores 1. Q̃ = 1: as good as the evidence allows.
### 3.3 Boundary evidence
Small inline table: source → what it captures → course topic.
- Regular-expression rules (abbreviations, initials, numbering, quotes) — L2.
- Kneser–Ney trigram: P(`</s>` | left context) — L5.
- MaxEnt probe on frozen BERT `[MASK]` states + insertion likelihood ratio — L6/L7. **[M3]** p_i = σ(w·φ(h_i) + b).
- Zero-shot SaT-12l probability; fine-tuned ModernBERT-large BIO-style boundary tagger (cased input) — L7.
- Paragraph-break flag when the input has layout.
### 3.4 Fusion trained for quality
**[M4]** p_i = σ(θ·x_i); θ minimizes L = L_Q/L_Q⁰ + L_BCE/L_BCE⁰ + λ‖θ‖², with L_Q = mean over training candidates of
(Q̃(y, p_θ) − F1(y, g))² and L_BCE the boundary log loss (each term scaled by its value at the BCE solution; gradients
pass through the top-k sort). Intuition: the boundary loss treats all gaps alike; the quality loss focuses on the gaps
that decide which candidate is better. One sentence: in a controlled comparison this beats boundary-only training and
post-hoc QE regression, which overfits to the four seen systems (App. B).
### 3.5 What we tried on the way (one paragraph)
PuncEv (masked-LM punctuation probability) and the frozen probe alone (our first evaluators, tested once on EWT), an
HSMM length prior, Dawid–Skene aggregation, post-hoc QE regression and an LLM pilot; why each was kept or dropped, one
clause each (App. A–B).
→ *§4: how to judge such scores.*

## 4 Evaluating an evaluator (1.2 pages)
### 4.1 Data, segmenters and input conditions
**[T1]** GUM train (D: training and development, 197 docs, 15 genres); held-out: EWT test (316), GENTLE (26, 8 genres),
LinES test (9); fresh one-shot tests of the final evaluator: EWT dev (318, 5 genres) and LinES train+dev (18).
Candidates: Punkt, Stanza, WtP, SaT-3l-sm → items = document × segmenter. Two input conditions: flat text, and text
with paragraph breaks (UD `# newpar`, seen by segmenters and evaluators). Contamination: SaT-sm and Stanza saw GUM/EWT
training data; GENTLE and LinES are unseen by all.
### 4.2 Baselines (what a reader would try)
B1 punctuation rule (pseudo-reference: cut after every terminal punctuation); B2 sentence-length prior; B3 masked LM
(BERT probability of terminal punctuation, Platt-calibrated) with our scoring; B4 the strongest available segmenter,
SaT-12l-sm, **its output as pseudo-reference**; B5 **its probabilities with our scoring** (the simplest way to turn a
segmenter into an evaluator); B6 multi-system consensus (agreement with the majority of the other segmenters);
B7 DocAgree (how much the four segmenters agree on the document — document difficulty only); B8 **GPT-2 perplexity**, the
classic fluency score (mean token log-likelihood of the text written one sentence per line); B9 **Dawid–Skene**, the
classic EM aggregation of noisy annotators (one confusion matrix per segmenter, leave-one-out posterior, our scoring).
With layout, baselines also see the paragraph breaks. SaT-12l-sm was trained on GUM train (its GUM numbers are inflated).
### 4.3 Three views
- **V1 Agreement with true quality** **[M5]**: global Spearman ρ between score and true F1 over all items; within-
  document pairwise accuracy.
- **V2 System ranking** **[M6]**: Kendall τ between the evaluator's and the true ranking of the four segmenters, averaged
  over genres (and documents).
- **V3 Controlled sensitivity** **[M7]**: inject exactly one error of a type into the gold segmentation (missing boundary
  after terminal punctuation / without it at a paragraph break / without it inside a paragraph; false boundary after
  terminal punctuation (abbreviation, numbering) / after a comma / without punctuation; shifted boundary); **detection
  rate** = probability that the score drops (ties ½). **Control**: move one false boundary to another random non-boundary
  position — true quality unchanged, so every evaluator should be at 0.5. "Detected" = 95% CI above 0.5. Why V3: V1 can
  be high without any sensitivity (B7 never reacts to any error yet has ρ = 0.51–0.77).
### 4.4 Protocol
Gold isolation (scores frozen before gold is loaded); all decisions on GUM by leave-one-genre-out, written down before
results; one-shot tests (EWT test for the probe, GENTLE/LinES for the stack, EWT dev and LinES train+dev for the final
evaluator, with pre-registered hypotheses); genre-stratified document bootstrap (1000) for CIs. Baselines are fixed or
fitted on GUM only. Footnote: tie-exact ranking; history in App. H.
→ *§5.*

## 5 Results (2.8 pages)
### 5.1 RQ1: does the score track true quality?
**[T2] Global Spearman with true F1** (flat text; † = trained on GUM; bold = best per column; status of "Ours": D on GUM,
T on GENTLE/LinES test for the stack version and on the fresh corpora, R elsewhere; baselines P).

| Evaluator | GUM | GENTLE | LinES test | EWT test | **EWT dev (fresh)** | **LinES tr+dev (fresh)** |
|---|---:|---:|---:|---:|---:|---:|
| B1 Punctuation rule | 0.340 | 0.383 | 0.437 | 0.369 | 0.378 | 0.296 |
| B2 Length prior | 0.048 | 0.037 | 0.079 | 0.161 | 0.157 | 0.267 |
| B3 Masked LM | 0.349 | 0.542 | 0.758 | 0.508 | 0.464 | 0.440 |
| B4 SaT-12l-sm output as reference | 0.931† | 0.880 | **0.888** | 0.754 | 0.738 | **0.630** |
| B5 SaT-12l-sm probabilities | **0.936**† | 0.891 | 0.873 | 0.778 | 0.762 | **0.630** |
| B6 Consensus | 0.733 | 0.830 | 0.861 | 0.657 | 0.657 | 0.468 |
| B7 DocAgree | 0.637 | 0.773 | 0.741 | 0.529 | 0.508 | 0.410 |
| B8 GPT-2 perplexity | 0.104 | 0.064 | −0.322 | 0.043 | 0.062 | 0.049 |
| B9 Dawid–Skene | 0.746 | 0.826 | 0.859 | 0.667 | 0.618 | 0.505 |
| **Ours** | 0.903 | **0.907** | 0.873 | **0.787** | **0.797** | 0.563 |

*With paragraph layout* (compact second block): EWT dev (fresh) ours **0.851** vs B1 rule + newlines 0.796, B3 0.698,
B4 0.797, B5 0.835, B6 0.625, B7 0.509, B8 0.039, B9 0.619; GENTLE 0.903 vs B5 0.929; EWT test 0.894 vs B5 0.879.

Text:
- **Against simple baselines and consensus**: large, significant margins on every corpus except the 9-document LinES test
  (EWT dev, flat: +0.42 over the rule, +0.33 over the masked LM, +0.14 over consensus; with layout +0.227 [+0.158, +0.298]
  over consensus, the pre-registered primary test). With layout a punctuation + newline rule is already strong (0.80 on
  EWT dev); ours is still ahead (+0.055 [+0.033, +0.083]).
- **Classic baselines** (B8, B9): GPT-2 perplexity separates candidates of the same document (pairwise 0.71 on EWT dev)
  but not across documents (ρ ≈ 0.04–0.10; −0.32 on LinES): its likelihood mostly reflects what a document is about — the
  same confound that normalization removes from Soft F1. Dawid–Skene aggregation is no better than majority consensus
  (EWT dev 0.618 vs 0.657; ours +0.179 [+0.113, +0.249]).
- **Against the strongest adapted segmenter** (B4/B5): ours beats using SaT-12l-sm's *output* as reference on web text
  (EWT dev +0.060 [+0.008, +0.112] flat, +0.054 [+0.003, +0.105] with layout) and ties with its *probabilities in our
  scoring* everywhere held out (EWT dev +0.035 [−0.014, +0.088], GENTLE +0.015, LinES train+dev −0.066, all n.s.). B5 > B4
  shows that the scoring is itself what turns a segmenter into a good evaluator.
- One sentence each (App. C): system ranking (V2) — ours and B5 rank the four segmenters almost perfectly (EWT dev
  genre-mean τ 1.0 with layout), consensus does not (0.47); SaT-based evaluators slightly overrate the SaT candidate
  (rank residual +0.02 to +0.03), ours less so with layout; an unfitted average of ours and B5 is best on web text
  (EWT dev 0.808 / 0.862, P) but not on LinES.
- LinES (fresh and test): no evaluator transfers well (≤ 0.63 on train+dev); our improvements over the probe do not hold
  there (pre-registered H5) — see §5.3.
→ *Spearman says how well the score tracks quality overall; which errors does it actually notice?*

### 5.2 RQ2a: which errors does each evaluator notice?
**[T3] Detection rate of one injected error** (EWT dev, fresh; * = 95% CI above 0.5; the control should be 0.50).

| Evaluator | Merge at terminal punct. | Merge, no punct., at paragraph break | Merge, no punct., inside paragraph | False split after abbrev./numbering | False split at comma etc. | False split, no punct. | Shifted boundary | Control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 Punctuation rule | 1.00* | 0.01 | 0.01 | 0.00 | 0.97* | 0.97* | 0.89* | 0.50 |
| B2 Length prior | 1.00* | 1.00* | 1.00* | 0.00 | 0.01 | 0.00 | 0.52 | 0.50 |
| B3 Masked LM | 0.81* | 0.54 | 0.28 | 0.23 | 0.99* | 0.97* | 0.95* | 0.51 |
| B4 SaT-12l-sm output | 0.98* | 0.91* | 0.71* | 0.68* | 0.97* | 0.99* | 0.97* | 0.50 |
| B5 SaT-12l-sm probabilities | 0.96* | 0.85* | 0.64* | **0.75*** | 0.99* | 1.00* | 1.00* | 0.49 |
| B6 Consensus | 1.00* | 0.52 | 0.51 | 0.55 | 0.97* | 0.98* | 0.96* | 0.50 |
| B7 DocAgree | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| B8 GPT-2 perplexity | 0.82* | 0.86* | **0.89*** | 0.53 | 0.87* | 0.95* | 0.98* | 0.49 |
| B9 Dawid–Skene | 1.00* | 0.63* | 0.57 | 0.50 | 0.98* | 1.00* | 0.98* | 0.50 |
| **Ours (flat)** | 0.99* | **0.93*** | **0.84*** | 0.66* | 0.97* | 0.99* | 1.00* | 0.50 |
| Ours (layout input) | 0.99* | 1.00* | 0.31 | 0.69* | 1.00* | 1.00* | 1.00* | 0.49 |

Text (order the columns hard → easy in the final table):
- Every evaluator that reads the text notices errors marked by punctuation (≈ 1.0); the control is at 0.50 for all, so
  these are responses to quality, not to change.
- Two error types separate the evaluators: **missing boundaries without terminal punctuation** and **false splits after
  abbreviations/numbering**. Heuristics fail one side completely (the rule ignores unpunctuated boundaries; the length
  prior always prefers shorter sentences); the masked LM, consensus and Dawid–Skene are near chance on both (aggregating
  the systems cannot reveal an error that most of them make).
- Among evaluators that are not one-sided (the length prior catches every merge and no split), ours (flat) and GPT-2
  perplexity are the most sensitive to unpunctuated merges (0.93 / 0.84 and 0.86 / 0.89 vs 0.85 / 0.64 for B5); B5 is the
  best on post-abbreviation splits (0.75 vs 0.66 ours, 0.53 GPT-2) — complementary blind spots. Dawid–Skene shares the
  consensus blind spot (0.50 on post-abbreviation splits).
- With layout, our evaluator catches every merge at a paragraph break but becomes blind inside paragraphs (0.31): it has
  learned "unpunctuated boundary ≈ newline". The same holds on GUM and GENTLE (App. D; factorial in App. E).
- B7 shows why V1 alone is not enough: ρ = 0.51 here, detection 0.50 everywhere.
→ *What determines how reliable an evaluator is?*

### 5.3 RQ2b: what limits reliability?
- **The formula matters only in two ways** (three sentences, App. C): normalization (+0.24 for PuncEv, +0.05 for ours on
  GUM) and counting the document end (single-sentence documents: +0.348 [+0.251, +0.449] on fresh EWT dev, T); beyond
  that, soft ≈ hard pseudo-reference for our evidence (+0.005 / +0.009), although soft scoring helps turn SaT into an
  evaluator (B5 vs B4).
- **The evidence matters most** (one sentence + App. B): along our development each evidence component added agreement
  (GUM 0.57 → 0.90 flat); our course-method evidence reaches the level of the best supervised segmenter, but neither
  covers both hard error types (T3).
- **Consensus vs evidence** (one sentence + App. C): consensus is good at *which document is hard* (between-document ρ
  0.92) and poor at *which segmentation is better* (within-document 0.54 vs ours 0.80, EWT dev).
- **Domain shift**: annotation conventions set the ceiling — on LinES (fresh and test) all evaluators fall to ≤ 0.63 and
  post-abbreviation splits are detected at 0.2–0.4 (App. D).
- **Selection** (one paragraph, App. F): SaT is the best segmenter on every corpus; no evaluator selects better than
  always using it — the useful output is the ranking (§5.1), not a switch.
- **Conclusion of §5:** an evaluator is only as reliable as its evidence on the gaps that separate the candidates; adding
  cues moves its blind spots rather than removing them.

## 6 Discussion and limitations (0.8 page)
### 6.1 Main findings
1. Reference-free quality estimation for sentence segmentation works: with good evidence a simple scoring rule tracks
   true quality closely on held-out text and ranks the segmenters correctly.
2. Course-method evidence reaches the level of a state-of-the-art supervised segmenter; the two fail on different error
   types, and combining them helps on web text.
3. Controlled error injection is necessary next to correlation: it reveals what each evaluator cannot see (consensus:
   shared errors; layout-trained evidence: in-paragraph boundaries).
One sentence: the evidence used directly as a segmenter matches or beats SaT (App. G).
### 6.2 Limitations
- English only; four segmenters; UD annotation; idealized paragraph layout from UD.
- Small held-out sets (GENTLE 26 docs, LinES 9 + 18); EWT dev is new documents but the same domain as EWT test.
- Baseline comparisons on the fresh corpora are post hoc (our evaluator's numbers there are the one-shot test).
- Training-data overlap: SaT-sm and Stanza with GUM/EWT; our evidence and SaT-sm both supervised on GUM; GENTLE follows
  GUM conventions.
- Adaptive design (13 dated amendments, pre-registered); the document-end correction was found post hoc and then
  confirmed on a fresh test.
- Selection not shown to beat Always-SaT; transfer to different annotation conventions (LinES) fails.
### 6.3 Future work
Evidence for unpunctuated boundaries without layout; adaptation to annotation conventions; combining evidence sources
with different blind spots; domains where the best segmenter changes.

## 7 Conclusion (0.2 page)
We built and evaluated a reference-free quality estimator for sentence segmentation. On held-out and fresh text it
tracks true quality far better than heuristics, a language model or system consensus and matches the strongest
supervised segmenter used as an evaluator, while controlled error injection shows exactly which errors it misses and why
domain conventions limit it.

## End matter
Team contributions; code and data availability (repository, commit, `scripts/*.slurm`).

---

## Appendix plan (not counted in the 8 pages)
- **A. Preliminary exploration.** Synthetic perturbations on GUM dev; Qwen vs BERT; BCE vs Soft F1; early calibration.
- **B. Method details and development.** Probe, rules, n-gram, taggers, hyperparameters; the development ladder (old main
  table: Probe → stack → +casing → +layout → +quality training, all corpora, both conditions); the controlled
  training-objective comparison (boundary loss vs quality loss vs post-hoc QE, leave-one-segmenter-out); derivation of
  the top-k maximum and the gradient of the quality loss.
- **C. Agreement, full.** T2 with CIs, within-document pairwise, system-ranking τ, rank residuals (same-family bias), the
  layout condition, the unfitted ensemble, normalized vs raw vs hard scoring, between/within decomposition.
- **D. Sensitivity, full.** All severities (1 error … 40% of boundaries), slopes, z-scores (with their caveats), real-error
  pricing, GUM / GENTLE / LinES train+dev.
- **E. Blind-spot diagnosis.** Evidence by left context; oracle ceiling; case × layout factorial.
- **F. Selection.** Regret vs Always-SaT, Direct Internal Splitter, oracle headroom.
- **G. SaT analysis and evidence as a segmenter.**
- **H. Protocol and reproducibility.** Amendment timeline 1–13, test-reuse rules, one-shot hypotheses (amendment 12 table),
  tie-exact erratum, data releases, scripts, job ids.

## Figures and tables
| Slot | Content | Source |
|---|---|---|
| F1 | Evaluator overview (+ gold-isolation box) | drawn |
| F2 | Worked example: snippet, p_i bars, two candidates, Q̃ vs true F1 | `results/a11/examples.md` (pick), evidence caches |
| F3 | Scatter of score vs true F1 on EWT dev: consensus / B5 / ours (what ρ = 0.63 vs 0.85 looks like) | `results/a12`, `results/a13` |
| F4 (optional) | Detection rate vs number of injected errors for the two hard error types | `results/a13/report.json` |
| T1 | Data and segmenters | manifests |
| T2 | Agreement (V1), baselines vs ours | `results/a13/report.json` |
| T3 | Sensitivity (V3) on EWT dev | `results/a13/report.json` |
All plots from one script `paper/figures/make_figures.py` (to write).

## Notation
X document; i gap, n gaps; y candidate (0/1 per gap); g gold; p evidence; x_i features; θ fusion weights; Q Soft F1
(document end counted); Q̃ normalized; F1(y, g) true quality; ρ Spearman; τ Kendall.

## Open decisions
1. F4 in or out (page budget).
2. Team-size / contribution statement; division of writing (suggested: Weifeng §3–5 and appendix; Yanhan §1–2, F1/F2
   drawings, examples, slides; both §6–7).
3. Next step: `paper/main.tex` (ACL template, Overleaf), starting with §3–4.
