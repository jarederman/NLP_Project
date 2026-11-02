# Progress Report After the Proposal

**Meeting date:** November 2, 2026  
**Project:** Cross-Domain Reference-Free Quality Estimation for Sentence Segmentation  
**Team:** Weifeng Jiang (wj2301), Yanhan Zhu (yz11501)

## 1. Current research question

Our core question is now deliberately narrow: given an unlabeled target-domain corpus and outputs from Punkt, Stanza, WtP, and SaT, can a frozen BERT + Platt + Soft F1 evaluator rank these systems and select one with lower Macro Selection Regret than the fixed Always-SaT policy?

The project no longer requires the candidate-aware V2 model or boundary auditing for minimum completion. Those are extensions after the real-system selection experiment succeeds.

## 2. Work completed since the proposal

### 2.1 Scope and evaluation protocol frozen

- Core evaluator: V1 BERT + Platt calibration + Soft F1.
- Core candidate systems: Punkt, Stanza, WtP, and SaT.
- Required baselines: Direct Internal Splitter and Always-SaT.
- Required held-out setting: train/develop on GUM, freeze all choices, then evaluate once on EWT test.
- Primary outcome: Macro Selection Regret.
- Supporting outcomes: domain-level system ranking and within-document pairwise ranking accuracy.
- V2, error auditing, additional systems, GENTLE, and human annotation are extensions.

### 2.2 Dataset provenance fixed in code

The original downloader followed the moving `master` branch and did not record the exact GUM snapshot. This explains why the pilot's 197 train / 30 dev counts could not be reconciled reproducibly with the current repository.

The revised scripts now:

- pin UD English GUM and UD English EWT to release `r2.18`;
- fetch GUM train/dev and EWT test separately;
- save the release identifier;
- calculate SHA256 checksums for downloaded files;
- keep GUM test unread in the minimum protocol.

This work prepares reproducible data acquisition, but the new frozen data have not yet been downloaded or processed on the cluster.

### 2.3 Gold-isolated data contracts implemented

A new `src/contracts.py` separates three artifact types:

1. `RawDocument`: the only record available to splitters and the scorer;
2. `GoldBoundaries`: stored separately and opened during final meta-evaluation;
3. `CandidateSegmentation`: character-offset output plus system version/configuration.

Every artifact includes the same text SHA256. Validation rejects offsets associated with a different text snapshot, duplicate/unsorted offsets, document-final boundaries, and mismatched document IDs. This does not replace process isolation, but it makes accidental Gold leakage and alignment errors detectable.

### 2.4 Macro/Micro-aligned selection evaluation implemented

A new `src/selection.py` implements:

- hard Boundary Precision/Recall/F1 counts;
- expected Soft F1 counts;
- Macro quality and Macro Gold F1;
- pooled Micro quality and pooled Micro Gold F1 as a separate sensitivity analysis;
- Direct Internal Splitter thresholding;
- predicted and Gold system rankings;
- Macro and Micro Selection Regret without mixing aggregation conventions;
- Oracle headroom over Always-SaT;
- within-document pairwise ranking accuracy with Gold-quality ties excluded.

### 2.5 CPU tests completed

Seven unit tests currently pass. They verify:

- text checksum and offset isolation;
- rejection of Gold associated with a different text snapshot;
- JSONL round-trip behavior;
- Direct Internal thresholding;
- separate Macro and Micro outputs;
- Selection Regret and Always-SaT headroom;
- exclusion of equal-Gold-quality pairs from pairwise evaluation.

Command used:

```bash
python3 -m unittest discover -s tests -v
```

Result: `Ran 7 tests ... OK`.

## 3. What has not yet been completed

We should not present the infrastructure work as a new experimental result. The following remain unfinished:

- frozen GUM/EWT data have not been downloaded and parsed on the cluster;
- faithful raw-document reconstruction and Gold-independent BERT evidence are not yet complete;
- Punkt, Stanza, WtP, and SaT have not yet been run on the frozen data;
- Oracle headroom over Always-SaT is unknown;
- V1 has not yet been evaluated on real splitter outputs;
- no EWT Selection Regret result exists yet.

The existing 0.894 pairwise accuracy and 0.877 within-document Spearman remain results on synthetic GUM-dev perturbations only.

## 4. Immediate next experiment

1. Run the frozen GUM/EWT download scripts on the cluster and record counts/checksums.
2. Produce Gold-isolated raw-document and boundary-offset JSONL files.
3. Implement a common interface `split(raw_text) -> character offsets` for Punkt, Stanza, WtP, and SaT.
4. Run the four systems on GUM dev and calculate their Gold Macro F1 and per-document wins.
5. Calculate Oracle headroom over Always-SaT before investing in further evaluator development.
6. If headroom exists, run frozen V1 selection and then perform the one-time EWT evaluation.

## 5. Decision gate

The next result determines the paper's final story:

- **Meaningful oracle headroom:** continue with domain-conditioned splitter selection and test whether V1 reduces regret.
- **Near-zero headroom:** report that a strong fixed SaT policy leaves little room for dynamic selection, and move boundary auditing to the most relevant extension.

## 6. Questions for the professor

1. Is Macro Selection Regret an appropriate primary metric, with pooled Micro results reported only as sensitivity analysis?
2. Is one frozen GUM-to-EWT cross-domain test sufficient for the minimum course project, with GUM test and GENTLE optional?
3. Are four representative splitters enough, or is a rule-based spaCy/punctuation baseline required in the core experiment?
4. For Gold-independent BERT evidence on raw text, would a uniform punctuation-masking transformation be acceptable if the untouched raw text and offset mapping are preserved separately?

## 7. Short oral update (about 60 seconds)

> Since the proposal, we narrowed the minimum project to the existing BERT–Platt–Soft-F1 evaluator, four real splitters, two critical baselines, and one frozen GUM-to-EWT test. We also fixed two methodological issues before running more models. First, our old GUM downloader followed the moving master branch, so we now pin UD release 2.18 and record checksums. Second, we implemented separate raw-text, Gold-boundary, and system-output artifacts linked by character offsets and text hashes, which makes leakage and alignment errors auditable. We also implemented aligned Macro and Micro evaluation, with Macro Selection Regret as the primary metric, plus Direct Internal and Always-SaT comparisons. Seven CPU unit tests pass. We do not yet claim new real-system results: the next step is to run Punkt, Stanza, WtP, and SaT on the frozen data and first measure whether there is meaningful oracle headroom over Always-SaT.

## 8. One-sentence takeaway

We have moved from a promising synthetic pilot to a reproducible and leakage-auditable real-system evaluation pipeline; the next empirical milestone is to determine whether real cross-domain splitter selection has enough oracle headroom to be useful.
