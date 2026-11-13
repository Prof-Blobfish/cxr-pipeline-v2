# Analysis plan — DRAFT (not frozen)

Status: DRAFT. Will be frozen by commit + git tag `analysis-plan-v1` + push, before the single test run.

## Hypotheses
TBD (see research/log.md, "Open questions").

## Primary metrics and subgroups
TBD.

## Confidence intervals
Patient-level bootstrap (resample patients with replacement, not images), 2,000 resamples, percentile 95% CIs. Any comparison between models or subgroups uses the same resamples (paired CI of the difference). Implementation: `src/bootstrap.py`.

## Pre-declared analyses (defined before computing on val; val = exploratory dry run, test = confirmatory)

### A1. Pneumothorax drain shortcut
- Labels: NEATX aggregated `Drain` (non-expert; available only for NIH Pneumothorax-positive images), loaded through `load_dev_drains` on dev and only through the test entry point on test.
- Score: the frozen primary score (`research/operating_points.json`, mean of member logits), Pneumothorax column.
- Groups: D+ = Pneumothorax-positive with drain; D- = Pneumothorax-positive without drain; N = all Pneumothorax-negative images of the split. Positives without a NEATX annotation are excluded from D+ and D- and counted.
- Metrics: AUROC(D+ vs N), AUROC(D- vs N) and their difference (D+ minus D-); sensitivity at the frozen Pneumothorax threshold in D+ and in D-, and its difference. Overall Pneumothorax AUROC reported alongside.
- CIs: patient-level bootstrap as above; all quantities from the same resamples (paired), N shared.
- Hypothesis H1 (test): AUROC(D+ vs N) > AUROC(D- vs N), supported if the lower bound of the paired 95% CI of the difference is > 0. Effect size and CI are reported whatever the outcome.
- Secondary (ensembling on hard cases): AUROC(D- vs N) of the ensemble minus each member, paired CIs.
- Descriptive only: share of AP views in D+ vs D-.

### A2. Error analysis at the frozen operating points (descriptive)
- Per class: sensitivity and specificity at the frozen thresholds, overall, by View Position (AP / PA) and by follow-up status (Follow-up # = 0 vs > 0).
- Summary: macro13 sensitivity difference AP minus PA, and first study minus follow-up, with paired patient-bootstrap CIs. Nothing is selected or changed on these.

### A3. Grad-CAM (qualitative, val only)
- Map: Grad-CAM of the ensemble Pneumothorax logit at the last DenseNet-121 feature map (7x7); each member's map before ReLU, averaged, then ReLU, upsampled to 224x224.
- Images (fixed rule, set before any map is viewed), from val, at most one image per patient per group, ties broken by Image Index:
  - 4 D+ and 4 D- positives, random sample with seed 0;
  - the 4 highest-scoring Pneumothorax negatives (most confident false positives);
  - the 4 lowest-scoring Pneumothorax positives (most confident false negatives).
- All 16 maps are shown; none are dropped. No localization metric on dev (all NIH bounding boxes are in test).

## Success criteria
TBD.

## Frozen artifacts (filled at freeze time)
- Checkpoint SHA-256 hashes:
- Thresholds / ensemble weights: `research/operating_points.json` (equal-weight mean of 2 seed logits; per-class thresholds at val sensitivity >= 0.90; checkpoint SHA-256 hashes inside, copied here at freeze time).
- Retrain on train+val: TBD (default: no)
