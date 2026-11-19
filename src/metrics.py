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


# Agreement among raters beyond chance. table: (items, categories) counts; every row must sum to the same number of reads
def fleiss_kappa(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    n = table.sum(axis=1)
    if not (n == n[0]).all() or n[0] < 2:
        raise ValueError("Fleiss' kappa needs the same number (>= 2) of reads on every item")
    n = n[0]
    p_j = table.sum(axis=0) / table.sum() # share of all reads in each category
    P_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1)) # per item: share of agreeing reader pairs
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum() # observed agreement vs agreement expected by chance
    if P_e == 1: # every read in one category -> kappa undefined
        return float("nan")
    return float((P_bar - P_e) / (1 - P_e))
