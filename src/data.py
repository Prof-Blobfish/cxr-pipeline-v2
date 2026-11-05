from pathlib import Path

import numpy as np
import pandas as pd

NIH_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]


def load_labels(nih_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(nih_dir / "Data_Entry_2017.csv")

    df = df.drop(columns=["Unnamed: 11"])
    df = df.rename(columns={
        "OriginalImage[Width": "width",
        "Height]": "height",
        "OriginalImagePixelSpacing[x": "spacing_x",
        "y]": "spacing_y",
    })

    labels = df["Finding Labels"].str.get_dummies(sep="|")
    labels = labels.drop(columns=["No Finding"])

    assert sorted(labels.columns) == sorted(NIH_DISEASES), labels.columns.tolist()

    return pd.concat([df, labels[NIH_DISEASES]], axis=1)


# Assigns train -> val -> test splits in that order
def assign_splits(
        df: pd.DataFrame, nih_dir: Path, val_frac: float = 0.125, seed: int = 42
) -> pd.DataFrame:
    test_images = set((nih_dir / "test_list.txt").read_text().split())
    is_test = df["Image Index"].isin(test_images)

    trainval_patients = np.sort(df.loc[~is_test, "Patient ID"].unique())
    rng = np.random.default_rng(seed)
    n_val = round(val_frac * len(trainval_patients))
    val_patients = rng.choice(trainval_patients, size=n_val, replace=False) # Come from trainval patients, separate from test patients

    df = df.copy()
    df["split"] = "train"
    df.loc[df["Patient ID"].isin(val_patients), "split"] = "val"
    df.loc[is_test, "split"] = "test" # Patient-disjoint from val patients which only come from train_val patients

    splits_per_patient = df.groupby("Patient ID")["split"].nunique()
    assert (splits_per_patient == 1).all(), "A patient appears in more than one split" # Guarantees no patient ends up in two splits

    return df


# Builds a filename -> full-path lookup dictionary; makes each lookup instant instead of searching 12 folders per lookup
def build_image_paths(nih_dir: Path) -> dict[str, Path]:
    paths = {
        p.name: p
        for p in nih_dir.glob("images_*/images/*.png")
        if not p.name.startswith(".")
    }
    assert len(paths) == 112_120, f"Expected 112,120 images, found {len(paths):,}" # Catches missing or duplicate files right away
    return paths


# Joins fresh labels from the CSV with the frozen split on file (nih_split.csv)
def load_split_labels(nih_dir: Path, split_csv: Path) -> pd.DataFrame:
    labels = load_labels(nih_dir)
    split = pd.read_csv(split_csv)[["Image Index", "split"]]
    df = labels.merge(split, on="Image Index", how="inner", validate="one_to_one") # validate raises an error if any Image Index appears twice on either side
    assert len(df) == len(labels), "Some images are missing from the split file"
    return df
