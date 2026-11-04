from pathlib import Path

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
