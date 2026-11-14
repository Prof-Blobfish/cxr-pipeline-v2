# Research log

Append-only. Never edit past entries; add a new dated entry to correct one.
Decision details live in each notebook's final "decisions" cell; this log links to them.

## 2026-11-03: Milestone 1: EDA
- Explored NIH ChestX-ray14 (112,120 images, 30,805 patients). Decisions -> `notebooks/01_eda.ipynb` (Findings -> decisions).
- Disclosure: EDA viewed label distributions across the full dataset, and 02 viewed per-split distributions including test (e.g. Pneumothorax 10.4% test vs 3.1% train). Descriptive only; no model results. This partly motivated the drain hypothesis.

## 2026-11-04: Milestone 2: patient-level split
- Official NIH test list + 12.5% of train_val patients as val (seed 42), frozen in `splits/nih_split.csv`. -> `notebooks/02_split.ipynb`.

## 2026-11-04: Milestone 3: Dataset + transforms
- 224², ImageNet norm, mild affine + brightness/contrast, no flips. -> `notebooks/03_dataset.ipynb`.
- Dead end/fix: 3 macOS `._*.png` files broke the image count; `build_image_paths` now skips dotfiles.

## 2026-11-05: Milestone 4: model
- DenseNet-121 (ImageNet), Linear(1024->14) head. Init loss ≈ ln 2; single-batch overfit passed. -> `notebooks/04_model.ipynb`.

## 2026-11-05: Research direction + test isolation
- Tentative question: audit study (expert-label re-scoring; Pneumothorax drain shortcut; ensembling on hard cases). Not final; to be decided on val.
- Test rules: val decides everything; test runs once after the analysis plan, checkpoint hashes, and thresholds are committed and pushed.
- Added `load_dev_labels`, `NIHDataset` test guard, `src/paths.py`; removed `test_ds` from 03.
- Coverage check (IDs only): expert all-findings = 100% test; NEATX ≈ half test. -> `notebooks/02b_label_coverage.ipynb`.

## 2026-11-05: Milestone 5: loss + class imbalance
- `src/losses.py`: train-only prevalence and pos_weight (`none`/`sqrt`/`ratio`), prior-logit bias init. -> `notebooks/05_loss.ipynb`.
- Refines the EDA decision "use pos_weight": weighting is now a val experiment for milestone 6 (none vs sqrt), since it distorts calibration and its AUROC benefit on NIH is uncertain.
- Know-nothing loss baseline: 0.164.

## 2026-11-05: Log maintenance
- Formatting-only edit to earlier entries: replaced Unicode arrows with "->". No content changed (verifiable in git history).

## 2026-11-09: Milestone 6: training + none vs sqrt
- `src/train.py`, `src/metrics.py` (manual tie-aware AUROC), FP32 classifier head. Runs auto-append to `research/runs.csv`. -> `notebooks/06_training.ipynb`.
- 4 runs (none/sqrt x seed 0/1, 5 epochs, commit `97cf6d5`). Val macro13: none 0.8403 / 0.8410, sqrt 0.8380 / 0.8373.
- Decision: pos_weight = none (pre-declared rule; sqrt never beat none, seed spread 0.0007). sqrt prior-corrected calibration ≈ none; raw sqrt over-predicts ~3.5x.
- Observation, not acted on: sqrt Hernia AUROC 0.963 vs none 0.891 (22 val positives).

## 2026-11-11: Milestone 7: val uncertainty, seed ensemble, operating points
- `src/bootstrap.py` (patient-level weighted bootstrap AUROC), `src/operating_points.py` (thresholds, SHA-256, write-once JSON). -> `notebooks/07_validation.ipynb`.
- Patient bootstrap (2,000, paired): ensemble macro13 0.8454 [0.8365, 0.8533]; singles 0.8403 [0.8314, 0.8483] / 0.8410 [0.8323, 0.8490].
- Decision: primary model = 2-seed ensemble (mean logits). Pre-declared rule met: +0.0051 [0.0035, 0.0069] vs s0, +0.0044 [0.0027, 0.0060] vs s1.
- M6 re-check with paired CIs (no change): sqrt - none [-0.0054, +0.0009] / [-0.0069, -0.0006].
- Decision: thresholds = val sensitivity >= 0.90 per class, frozen in `research/operating_points.json` with checkpoint and val-logit hashes. Youden rejected as unstable (median resample range 0.13 vs 0.07).
- Descriptive: Edema pooled AUROC 0.926 > AP 0.852 and PA 0.886 (view confound; not acted on).
- Environment: pandas 3.1.1 -> 3.0.6 (Smart App Control began blocking 3.1.1's unsigned `_libs/ops` extension on 2026-11-11; the file was unchanged since install; force-reinstall didn't help and 3.1.1 is the latest release). M7 outputs are identical under 3.0.6.

## 2026-11-12: Milestone 8 (part A): drain loader + pre-declared analyses
- `load_dev_drains` in `src/data.py`: NEATX drain labels filtered to dev images before use; raises on test rows. Dev coverage: 1,564 train / 219 val rows, all NIH Pneumothorax-positive (val: 91 drain, 128 no drain; 78 val positives unannotated). On dev the two annotators never disagree across 0.5; aggregated `Drain` = 1 iff both rate >= 0.75.
- Pre-declared in `research/analysis_plan.md` before any drain result was computed: A1 drain shortcut (H1), A2 error analysis at the frozen thresholds, A3 Grad-CAM selection rule. Val results will be labeled exploratory.
- NIH bounding boxes: all 880 images are in test (checked by Image Index only) -> no localization metric on dev.

## 2026-11-12: Milestone 8 (part B): drains, errors, Grad-CAM (val, exploratory)
- `bootstrap_mean` (`src/bootstrap.py`), `src/gradcam.py`. -> `notebooks/08_drains_errors_gradcam.ipynb`.
- A1: Pneumothorax AUROC D+ 0.949 vs D- 0.817, difference +0.131 [+0.086, +0.181]; sensitivity at the frozen threshold 0.989 vs 0.844. H1 criterion met on val (exploratory; possible size confound noted).
- A2: macro13 sensitivity first study minus follow-up -0.061 [-0.100, -0.027]; AP vs PA no clear difference.
- A3: Grad-CAM equals CAM for this head; maps qualitative only.
- Observation: re-inference does not bit-reproduce the saved val logits (bf16 + cudnn.benchmark; up to ~0.035 logits, run-to-run; a few frozen-threshold decisions flip).

## 2026-11-14: Milestone 9 (part A): expert labels, defined before any score
- `EXPERT_FOUR`, `EXPERT_TO_NIH`, `load_dev_expert_four`, `load_dev_expert_readers` in `src/data.py` (dev images filtered before any label column is used).
- Dev label facts (labels only, no scores): the four-findings "validation" file = 2,148 train + 266 val images. Val: 103 patients, 89% AP; expert positives Airspace opacity 117 (43 patients), Nodule or mass 35 (18), Pneumothorax 0, Fracture 2 -> only Airspace opacity and Nodule or mass are evaluable on val. The aggregated labels are adjudicated, not a reader majority.
- Pre-declared in `research/analysis_plan.md` before any score was computed on these images: A4 (four findings), A5 (all findings, test only).
- Isolation note: an early file preview printed the header and first 3 data rows of the three expert files that contain test images. No score was computed and nothing was decided from them.

## Open questions
- Final research question (decide after val exploration).
- Retrain on train+val before test? Default: no (thresholds come from val). Must be decided in the analysis plan.
- Does pos_weight help the rarest classes (Hernia)? Only n=22 on val; any check must be pre-declared, never used for selection.
- Pre-declare a view-stratified (AP/PA) analysis in the plan? Edema's pooled AUROC beats both views on val.
- Test entry point must first reproduce the saved val logits from the hashed checkpoints.
- Test-time Grad-CAM localization on the 880 bounding-box test images (e.g. pointing game)? Decide before freezing the plan.
- Reproduction tolerances for the test entry point (val re-inference differs from saved logits by up to ~0.035; some threshold decisions flip). State them in the plan; consider deterministic inference for the test run.
- Drained pneumothoraces may be larger: can size or another proxy separate "tube shortcut" from "easier cases"? Decide before freezing the plan.
- Expert Pneumothorax (A4) is only evaluable on test (0 val positives). Combine with NEATX drains on test (expert-labeled D+ / D-)? Decide before freezing the plan.
