from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PanDentalDataset(Dataset):
    def __init__(self, root, size=(128, 128)):
        self.root = Path(root)
        self.size = size

        image_dir = self.root / "Images"
        mask_dir = self.root / "Segmentation1"

        self.samples = []

        for image_path in image_dir.glob("*.png"):
            mask_path = mask_dir / image_path.name

            if mask_path.exists():
                self.samples.append((image_path, mask_path))

        self.samples.sort(
            key=lambda x: int(x[0].stem)
        )

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