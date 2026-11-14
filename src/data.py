from pathlib import Path

import numpy as np
import pandas as pd

NIH_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

DEV_SPLITS = ("train", "val") # No test loaders exist here on purpose to avoid accidentally using one


def assert_dev_only(df: pd.DataFrame) -> None: # Called right after filtering to ensure the filter excludes test
    if "split" not in df.columns:
        raise ValueError("DataFrame has no 'split' column, so it can't be verified as test-free")
    n_test = int((df["split"] == "test").sum())
    if n_test:
        raise RuntimeError(
            f"{n_test:,} test rows found in dev data. "
            "Test is only reachable through the dedicated test entry point."
        )


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


def load_dev_labels(nih_dir: Path, split_csv: Path) -> pd.DataFrame:
    df = _load_all_split_labels(nih_dir, split_csv)
    df = df[df["split"].isin(DEV_SPLITS)].reset_index(drop=True) # reset_index gives clean 0..N-1 row numbers after filtering; avoids subtle misalignment
    assert_dev_only(df)
    return df


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
def _load_all_split_labels(nih_dir: Path, split_csv: Path) -> pd.DataFrame: # Private function only called by functions like load_dev_labels
    labels = load_labels(nih_dir)
    split = pd.read_csv(split_csv)[["Image Index", "split"]]
    df = labels.merge(split, on="Image Index", how="inner", validate="one_to_one") # validate raises an error if any Image Index appears twice on either side
    assert len(df) == len(labels), "Some images are missing from the split file"
    return df


# NEATX drain labels exist only for Pneumothorax-positive images. Filtered to dev images before any label column is used
def load_dev_drains(neatx_csv: Path, dev_df: pd.DataFrame) -> pd.DataFrame:
    assert_dev_only(dev_df) # Raises on any test rows before the file is even read
    raw = pd.read_csv(neatx_csv)
    drains = raw[raw["Image Index"].isin(dev_df["Image Index"])] # Filter .isin() runs before any label col is used ("mixed files must be filtered inside a loader")
    drains = drains.merge(dev_df[["Image Index", "split"]], on="Image Index", how="left", validate="one_to_one")
    assert_dev_only(drains)
    if not drains["Drain"].isin([0, 1]).all():
        raise ValueError("NEATX 'Drain' should be 0/1")
    return drains[["Image Index", "split", "Drain"]].rename(columns={"Drain": "drain"}).reset_index(drop=True)


EXPERT_FOUR = ["Fracture", "Pneumothorax", "Airspace opacity", "Nodule or mass"]

# Model score for an expert finding = max over these NIH classes of the frozen score; NIH ref label = any of them. Fracture has no NIH class
EXPERT_TO_NIH = {
    "Airspace opacity": ["Atelectasis", "Consolidation", "Edema", "Infiltration", "Pneumonia"],
    "Nodule or mass": ["Mass", "Nodule"],
    "Pneumothorax": ["Pneumothorax"],
}


# Aggregated (adjudicated) expert four-findings labels. On dev only the "validation" file is read; it covers train and val images
def load_dev_expert_four(labels_csv: Path, dev_df: pd.DataFrame) -> pd.DataFrame:
    assert_dev_only(dev_df)
    raw = pd.read_csv(labels_csv)
    expert = raw[raw["Image Index"].isin(dev_df["Image Index"])]
    expert = expert.merge(dev_df[["Image Index", "split"]], on="Image Index", how="left", validate="one_to_one")
    assert_dev_only(expert)
    if not expert[EXPERT_FOUR].isin(["YES", "NO"]).all().all():
        raise ValueError("Expert four-findings labels should be YES/NO")
    flags = (expert[EXPERT_FOUR] == "YES").astype(int).add_prefix("expert ") # Expert file has its own Pneumothorax col. W/o prefix -> silent collision
    return pd.concat([expert[["Image Index", "split"]], flags], axis=1)


# Individual readers of the four-findings sets: one row per image and reader, YES/NO/HEDGE. The file mixes test and dev images
def load_dev_expert_readers(readers_csv: Path, dev_df: pd.DataFrame) -> pd.DataFrame: # Same shape as load_dev_drains
    assert_dev_only(dev_df)
    raw = pd.read_csv(readers_csv).rename(columns={"Image ID": "Image Index", "Nodule/mass": "Nodule or mass"})
    readers = raw[raw["Image Index"].isin(dev_df["Image Index"])]
    readers = readers.merge(dev_df[["Image Index", "split"]], on="Image Index", how="left", validate="many_to_one") # validate="many_to_one": each img has ~3 reader rows but exactly one split row
    assert_dev_only(readers)
    if readers.duplicated(["Image Index", "Reader ID"]).any(): # Protects the "3 reads per img" assumption Fleiss' kappa needs
        raise ValueError("A reader labeled the same image twice")
    if not readers[EXPERT_FOUR].isin(["YES", "NO", "HEDGE"]).all().all(): # First .all() gives one result per col, second reduces those to a single T/F
        raise ValueError("Reader labels should be YES/NO/HEDGE")
    return readers[["Image Index", "split", "Reader ID", *EXPERT_FOUR]].reset_index(drop=True)
