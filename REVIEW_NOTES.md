# Notes for code reviewers

This repository is an NYU CSCI-UA.0469 course project: **reference-free quality estimation for sentence
segmentation**. Given raw text and the output of a sentence segmenter, the evaluator scores the output without the
target corpus's gold boundaries. The paper's numbers come from the modules listed below; please look for bugs that
could change those numbers or invalidate the protocol.

## What the code computes

1. **Data** (`prepare_real_data`, `surface_data`, `contracts`, `layout_data`): documents are rebuilt from UD CoNLL-U
   surface tokens; a *gap* is a whitespace position between two tokens; a segmentation is a 0/1 vector over gaps; gold
   is a separate artifact joined by `doc_id` + text SHA256. Paragraph layout (optional) comes from UD `# newpar`.
2. **Systems** (`real_splitters`, `run_real_splitters`): Punkt, Stanza, WtP, SaT-3l-sm produce candidate segmentations;
   boundaries not at a gap are logged as alignment failures.
3. **Evidence** (per-gap boundary probability `p`): frozen-BERT probe (`boundary_probe`, `ewt_test`, `a6_test`),
   zero-shot SaT-12l and a fine-tuned ModernBERT binary tagger (`a6_gpu`, `a9_*`), regex rules and Kneser–Ney n-gram
   features (`a6_combine`, `gap_features`), fused by logistic regression; the final fusion is trained with a quality
   loss + BCE (`a10_quality`).
4. **Scoring**: normalized Soft F1. Training (`a10_quality.QualityProblem`) uses the form without the document-end term;
   evaluation (`a11_analysis.soft_f1_end`) counts the document end — intentional and documented (see below).
5. **Meta-evaluation** (`quality_agreement`, `a9_layout.bootstrap/fast_stats`, `a6_combine`, `a11_analysis`,
   `a12_test`, `a13_baselines`, `a13b_classic`, `a13c_softhard`): global Spearman between score and true F1 over
   (document × system) items, within-document pairwise accuracy, system-ranking Kendall τ, genre-stratified document
   bootstrap; controlled error injection (`a8_supplement.candidates`, `a11_analysis.candidates2`) with a
   quality-neutral control (`a13_baselines.control_rows`).

## Invariants that must hold

- **Gold isolation**: scorers and splitters read only raw/surface artifacts; gold of a test corpus is written only after
  its frozen scores exist (`a6_test.gold`, `a12_test.gold`, `materialize_dev_gold`). GUM test is never read.
- **Cross-fitting on GUM**: every GUM number is leave-one-genre-out (LOGO); a model that scores genre g must not have
  seen genre g's labels (including nested features in `a10_quality.fold_job`).
- **Alignment of arrays**: per-gap arrays are concatenated in `surfaces()` document order (sorted by `doc_id`); items are
  document-major in `SYSTEMS` order; flat and newline versions of a corpus have identical gaps.
- **Reproducibility**: perturbation seeds `(SEED, doc_index, type, level)`; types 0–6 and levels 0–3 of
  `candidates2` must reproduce `a8_supplement.candidates` exactly.

## Intentional choices (not bugs, but tell us if they are implemented wrongly)

- Scores and gold F1 are rounded to 1e-10 before ranking (`quality_agreement.tie_exact`) so identical candidates tie.
- The quality-training loss omits the document-end term used at evaluation; on GUM the two forms differ per item by
  ~0.001–0.002 and give the same global Spearman.
- Item gold F1 counts unaligned (non-gap) system boundaries and non-gap gold boundaries (`quality_agreement.load_items`);
  it can therefore differ from a gap-only F1.
- In the layout condition baselines also see paragraph breaks (hard references ∪ breaks; soft evidence 1 at breaks).
- The control in the sensitivity test is exchangeable by construction (expected detection 0.5 for any scorer).
- SaT on newline text: `split_on_input_newlines` is reproduced on offsets (`real_splitters.with_input_newlines`).

## Where to look hardest

1. Leakage of gold or of held-out genres into anything that scores held-out items.
2. Off-by-one / ordering mismatches between gap offsets, `doc_slices`, candidate matrices, evidence arrays and items.
3. Statistics: paired bootstrap (same resample for both arms), CI computation, NaN handling, Kendall/Spearman with ties.
4. Scoring formulas: Soft F1, the top-k maximum (k = 0..n with the end term), hard F1 conventions for empty sets.
5. Perturbation generator and control: correct error types, positions and severities; correct detection credit.
6. Inconsistent definitions of the same quantity across amendments (e.g. consensus M0 vs gap-level consensus).

Tests: `python -m unittest discover -s tests` (CPU only; numpy, scipy, torch needed for some modules).
Out of scope: the preliminary synthetic-perturbation / Qwen study (`run_experiment`, `data`, `lm`, `candidates`,
`evaluate`, `report`, …) and older diagnostics, whose results appear only in the paper's appendix.
