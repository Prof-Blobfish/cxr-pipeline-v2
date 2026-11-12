import hashlib
import json
from pathlib import Path

import numpy as np


# Rule everywhere (val now, test later): predict pos if score >= threshold
def sens_spec(scores: np.ndarray, targets: np.ndarray, threshold: float) -> tuple[float, float]:
    targets = targets.astype(bool)
    pred = scores >= threshold
    return float(pred[targets].mean()), float((~pred[~targets]).mean())


# Highest threshold that still catches >= target of the positives. Uses only the positives' scores -> prevalence-free
def sensitivity_threshold(scores: np.ndarray, targets: np.ndarray, target: float = 0.90) -> float:
    pos = np.sort(scores[targets.astype(bool)]) # Sort positives
    k = int(np.floor((1 - target) * len(pos) + 1e-9)) # Positives allowed below; 1e-9 stops 12.999... from flooring to 12
    return float(pos[k])


# Threshold maximizing sensitivity + specificity - 1 (kept for comparison, not frozen)
def youden_threshold(scores: np.ndarray, targets: np.ndarray) -> float:
    targets = targets.astype(bool)
    pos, neg = np.sort(scores[targets]), np.sort(scores[~targets])
    candidates = np.unique(scores)
    sens = 1 - np.searchsorted(pos, candidates, side="left") / len(pos)
    spec = np.searchsorted(neg, candidates, side="left") / len(neg)
    return float(candidates[np.argmax(sens + spec - 1)])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Frozen files are written once. Rerunning with identical content is fine; different content raises
def write_frozen_json(obj: dict, path: Path) -> None:
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise RuntimeError(f"{path.name} already exists with different content; frozen files are never overwritten")
        return
    path.write_text(json.dumps(obj, indent=2) + "\n")
