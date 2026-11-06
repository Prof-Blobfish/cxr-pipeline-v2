import pandas as pd
import torch
import torch.nn as nn

from src.data import NIH_DISEASES


# Guard stricter than assert_dev_only (val isn't allowed)
def _require_train_only(df: pd.DataFrame) -> None:
    if "split" not in df.columns:
        raise ValueError("DataFrame has no 'split' column, so it can't be verified as train-only")
    splits = set(df["split"].unique())
    if splits != {"train"}:
        raise ValueError(f"Class statistics must come from train rows only, got splits: {sorted(splits)}")


# Both loss weights and bias init need prevalence; computed in one place and they can never disagree
def compute_prevalence(train_df: pd.DataFrame) -> torch.Tensor:
    _require_train_only(train_df)
    return torch.tensor(train_df[NIH_DISEASES].mean().to_numpy(), dtype=torch.float32)


def compute_pos_weight(train_df: pd.DataFrame, mode: str = "none") -> torch.Tensor | None:
    if mode == "none":
        return None
    prevalence = compute_prevalence(train_df)
    ratio = (1 - prevalence) / prevalence # N_neg / N_pos written with prevalences
    if mode == "ratio":
        return ratio
    if mode == "sqrt":
        return ratio.sqrt()
    raise ValueError(f"Unknown pos_weight mode: {mode!r} (expected 'none', 'sqrt', or 'ratio')") # !r prints the value with quotes


# With pos_weight=None, it's just the prior; with weights it adds log w
def prior_logits(prevalence: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    logits = torch.log(prevalence / (1 - prevalence))
    if pos_weight is not None:
        logits = logits + torch.log(pos_weight)
    return logits


# The one place loss gets constructed
def build_loss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
