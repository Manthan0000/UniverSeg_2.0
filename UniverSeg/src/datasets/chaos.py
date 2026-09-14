from pathlib import Path
import numpy as np
import torch
import pydicom
from PIL import Image
from torch.utils.data import Dataset


class CHAOSDataset(Dataset):
    def __init__(self, root, size=(128, 128)):
        self.root = Path(root)
        self.size = size
        self.samples = []

        ct_root = self.root / "CT"

        for patient in sorted(
            ct_root.iterdir(),
            key=lambda x: int(x.name)
        ):
            dicom_dir = patient / "DICOM_anon"
            ground_dir = patient / "Ground"

            if not dicom_dir.exists() or not ground_dir.exists():
                continue

            images = sorted(dicom_dir.glob("*.dcm"))
            masks = sorted(ground_dir.glob("liver_GT_*.png"))

            for image, mask in zip(images, masks):
                mask_array = np.array(
                    Image.open(mask)
                )

                # Keep only slices containing liver
                if np.any(mask_array > 0):
                    self.samples.append((image, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = pydicom.dcmread(image_path).pixel_array.astype(np.float32)
        mask = np.array(
            Image.open(mask_path),
            dtype=np.float32
        )

        image_min = image.min()
        image_max = image.max()

        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)

        image = Image.fromarray(image)
        mask = Image.fromarray((mask > 0).astype(np.uint8))

        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        image = torch.from_numpy(
            np.array(image, dtype=np.float32)
        )[None]

        mask = torch.from_numpy(
            np.array(mask, dtype=np.float32)
        )[None]

        return image, mask