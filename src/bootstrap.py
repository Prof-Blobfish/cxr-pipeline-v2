import numpy as np
import pandas as pd

from src.data import NIH_DISEASES


# One row per replicate: how many times each patient was drawn. Patients are the independent units, not images
def patient_resample_counts(patient_ids: pd.Series, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    codes, uniques = pd.factorize(patient_ids) # Gives patient codes (c[codes] maps patient count to img weights)
    n_patients = len(uniques)
    rng = np.random.default_rng(seed)

    # Drawing with replacement -> result is counts (2000 * 3501 int32 = 28 MB)
    draws = rng.integers(0, n_patients, size=(n_boot, n_patients))
    counts = np.stack([np.bincount(d, minlength=n_patients) for d in draws]).astype(np.int32)

    return counts, codes # Return counts + codes instead of img weights makes pairing easy (same counts for every model) and subgroups easy (scores[mask], codes[mask])


# AUROC on every replicate without copying images: an image's weight = how many times its patient was drawn
def bootstrap_auroc(scores: np.ndarray, targets: np.ndarray, counts: np.ndarray, codes: np.ndarray) -> np.ndarray:
    targets = targets.astype(bool)
    _, inverse = np.unique(scores, return_inverse=True) # Runs once per class outside loop
    n_levels = int(inverse.max()) + 1
    inv_pos, inv_neg = inverse[targets], inverse[~targets]
    codes_pos, codes_neg = codes[targets], codes[~targets]
    out = np.empty(len(counts))
    for b, c in enumerate(counts): # Loop is just 2 bincounts + a cumsum
        pos = np.bincount(inv_pos, weights=c[codes_pos], minlength=n_levels)
        neg = np.bincount(inv_neg, weights=c[codes_neg], minlength=n_levels)
        out[b] = (pos * (np.cumsum(neg) - 0.5 * neg)).sum() / (pos.sum() * neg.sum()) # Weighted, tie-aware Mann-Whitney formula
    return out


# Shape (n_boot, 14), columns in NIH_DISEASES order
def bootstrap_per_class_auroc(logits: np.ndarray, labels: np.ndarray, counts: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return np.stack(
        [bootstrap_auroc(logits[:, i], labels[:, i], counts, codes) for i in range(len(NIH_DISEASES))], axis=1
    )


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> np.ndarray:
    tail = (1 - level) / 2 * 100
    return np.percentile(samples, [tail, 100 - tail], axis=0) # 1-D macro array or a (B, 14) per-class array


# Weighted mean of a per-img value on every replicate, e.g. hit (score >= threshold) over positives -> sensitivity
def bootstrap_mean(values: np.ndarray, counts: np.ndarray, codes: np.ndarray) -> np.ndarray:
    w = counts[:, codes] # counts[:, codes] = (2000 x n_images) img weights; pairing works automatically
    return (w @ values) / w.sum(axis=1) # w @ values sums them per replicate, and dividing by weight total gives weighted mean
