# Analysis plan — DRAFT (not frozen)

Status: DRAFT. Will be frozen by commit + git tag `analysis-plan-v1` + push, before the single test run.

## Hypotheses
TBD (see research/log.md, "Open questions").

## Primary metrics and subgroups
TBD.

## Confidence intervals
Patient-level bootstrap (resample patients with replacement, not images), 2,000 resamples, percentile 95% CIs. Any comparison between models or subgroups uses the same resamples (paired CI of the difference). Implementation: `src/bootstrap.py`.

## Success criteria
TBD.

## Frozen artifacts (filled at freeze time)
- Checkpoint SHA-256 hashes:
- Thresholds / ensemble weights: `research/operating_points.json` (equal-weight mean of 2 seed logits; per-class thresholds at val sensitivity >= 0.90; checkpoint SHA-256 hashes inside, copied here at freeze time).
- Retrain on train+val: TBD (default: no)
