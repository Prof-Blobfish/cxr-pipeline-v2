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

### A4. Expert four-findings re-labeling (val = dry run on 266 images; test = confirmatory)
- Labels: aggregated (adjudicated) expert labels, loaded through `load_dev_expert_four` on dev (the "validation" file only) and only through the test entry point on test. NIH labels of the same images are the comparison.
- Scores (`EXPERT_TO_NIH` in `src/data.py`): max over the mapped NIH classes of the frozen primary score; NIH reference label = 1 if any mapped NIH label is 1. Airspace opacity <- Atelectasis, Consolidation, Edema, Infiltration, Pneumonia; Nodule or mass <- Mass, Nodule; Pneumothorax <- Pneumothorax. Fracture has no NIH class: prevalence only.
- Evaluable: a finding is evaluated on a split if it has >= 10 positive images under both the expert and the NIH label, decided from label counts before any score is computed; otherwise counts only.
- Primary (estimation, no directional hypothesis): AUROC against the expert label and against the NIH label on the same images, and the difference (expert minus NIH), paired patient bootstrap.
- Secondary: sensitivity and specificity at the frozen thresholds against both label sets (a mapped finding is predicted positive if any of its NIH classes is at or above its frozen threshold); AUROC against the expert label of the ensemble minus each member, paired CIs.
- Descriptive: NIH-vs-expert label agreement per finding; number of NIH "No Finding" images with any expert-positive finding.
- Readers (descriptive only, `load_dev_expert_readers`): per finding, the distribution of YES reads, the number of HEDGE reads, and Fleiss' kappa (YES vs not YES; HEDGE counted as not YES) on images with exactly 3 reads. No model-vs-reader comparison.

### A5. Expert all-findings re-labeling (test only, 810 images)
- Labels: aggregated expert all-findings labels (YES/NO; "Pleural Thickening" = NIH "Pleural_Thickening"), read only by the test entry point. There are no dev images, so there is no dry run; the metric code is the A4 code, dry-run on val.
- Classes: the 13 selection classes; a class is evaluated if it has >= 10 positive images under both the expert and the NIH label, decided from label counts before any score is computed.
- Primary (estimation): per class AUROC against the expert and the NIH label and the difference; macro over the evaluated classes, difference expert minus NIH, paired patient bootstrap.
- Secondary: "Abnormal": score = max of the 14 frozen scores, AUROC against the expert "Abnormal" label and against NIH "any of the 14 findings". Expert macro AUROC of the ensemble minus each member, paired CIs.

## Success criteria
TBD.

## Frozen artifacts (filled at freeze time)
- Checkpoint SHA-256 hashes:
- Thresholds / ensemble weights: `research/operating_points.json` (equal-weight mean of 2 seed logits; per-class thresholds at val sensitivity >= 0.90; checkpoint SHA-256 hashes inside, copied here at freeze time).
- Retrain on train+val: TBD (default: no)
