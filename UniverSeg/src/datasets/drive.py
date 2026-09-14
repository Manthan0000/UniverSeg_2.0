from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DRIVEDataset(Dataset):
    def __init__(self, root, size=(128, 128)):
        self.root = Path(root)
        self.size = size

        image_dir = self.root / "training" / "images"
        manual_dir = self.root / "training" / "1st_manual"

        self.samples = []

        for image_path in sorted(image_dir.glob("*.tif")):
            number = image_path.stem.split("_")[0]
            mask_path = manual_dir / f"{number}_manual1.gif"

            if mask_path.exists():
                self.samples.append((image_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)

        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image)[None]
        mask = torch.from_numpy(mask)[None]

        return image, mask