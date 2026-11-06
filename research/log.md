# Research log

Append-only. Never edit past entries; add a new dated entry to correct one.
Decision details live in each notebook's final "decisions" cell; this log links to them.

## 2026-11-03: Milestone 1: EDA
- Explored NIH ChestX-ray14 (112,120 images, 30,805 patients). Decisions → `notebooks/01_eda.ipynb` (Findings → decisions).
- Disclosure: EDA viewed label distributions across the full dataset, and 02 viewed per-split distributions including test (e.g. Pneumothorax 10.4% test vs 3.1% train). Descriptive only; no model results. This partly motivated the drain hypothesis.

## 2026-11-04: Milestone 2: patient-level split
- Official NIH test list + 12.5% of train_val patients as val (seed 42), frozen in `splits/nih_split.csv`. → `notebooks/02_split.ipynb`.

## 2026-11-04: Milestone 3: Dataset + transforms
- 224², ImageNet norm, mild affine + brightness/contrast, no flips. → `notebooks/03_dataset.ipynb`.
- Dead end/fix: 3 macOS `._*.png` files broke the image count; `build_image_paths` now skips dotfiles.

## 2026-11-05: Milestone 4: model
- DenseNet-121 (ImageNet), Linear(1024→14) head. Init loss ≈ ln 2; single-batch overfit passed. → `notebooks/04_model.ipynb`.

## 2026-11-05: Research direction + test isolation
- Tentative question: audit study (expert-label re-scoring; Pneumothorax drain shortcut; ensembling on hard cases). Not final; to be decided on val.
- Test rules: val decides everything; test runs once after the analysis plan, checkpoint hashes, and thresholds are committed and pushed.
- Added `load_dev_labels`, `NIHDataset` test guard, `src/paths.py`; removed `test_ds` from 03.
- Coverage check (IDs only): expert all-findings = 100% test; NEATX ≈ half test. → `notebooks/02b_label_coverage.ipynb`.

## Open questions
- Final research question (decide after val exploration).
- Retrain on train+val before test? Default: no (thresholds come from val). Must be decided in the analysis plan.
