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


def assign_splits(
        df: pd.DataFrame, nih_dir: Path, val_frac: float = 0.125, seed: int = 42
) -> pd.DataFrame:
    test_images = set((nih_dir / "test_list.txt").read_text().split())
    is_test = df["Image Index"].isin(test_images)

    trainval_patients = np.sort(df.loc[~is_test, "Patient ID"].unique())
    rng = np.random.default_rng(seed)
    n_val = round(val_frac * len(trainval_patients))
    val_patients = rng.choice(trainval_patients, size=n_val, replace=False)

    df = df.copy()
    df["split"] = "train"
    df.loc[df["Patient ID"].isin(val_patients), "split"] = "val"
    df.loc[is_test, "split"] = "test"

    splits_per_patient = df.groupby("Patient ID")["split"].nunique()
    assert (splits_per_patient == 1).all(), "A patient appears in more than one split"

    return df
