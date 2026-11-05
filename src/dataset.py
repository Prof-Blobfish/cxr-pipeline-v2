from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data import NIH_DISEASES


class NIHDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_paths: dict[str, Path], transform): # Everything is precomputed, since __getitem__ runs ~75,000 times per epoch
        self.paths = [str(image_paths[name]) for name in df["Image Index"]] # List of plain strings (small and pickle faster)
        self.labels = torch.tensor(df[NIH_DISEASES].to_numpy(), dtype=torch.float32) # BCEWithLogitsLoss needs float targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as im:
            image = im.convert("RGB") # Replicates the gray channel to 3 and drops the alpha channel from RGBA files
        return self.transform(image), self.labels[idx]
