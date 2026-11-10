import numpy as np

from src.data import NIH_DISEASES

SELECTION_CLASSES = [d for d in NIH_DISEASES if d != "Hernia"] # Hernia only has 22 pos -> too noisy to pick models on


# Prob that a random pos outscores (gets a higher logit than) a random neg for one class
def auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    targets = targets.astype(bool)
    n_pos = int(targets.sum())
    n_neg = len(targets) - n_pos
    if n_pos == 0 or n_neg == 0: # If a class has no positives, which only happens in smoke tests
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2
    ranks = avg_rank[inverse]
    return float((ranks[targets].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# Returns a dict of AUROC scores for each class
def per_class_auroc(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {d: auroc(logits[:, i], labels[:, i]) for i, d in enumerate(NIH_DISEASES)}


# Returns an avg of all AUROC scores across all classes (minus Hernia)
def macro_auroc(per_class: dict[str, float], classes: list[str] = SELECTION_CLASSES) -> float:
    return float(np.mean([per_class[c] for c in classes]))
