from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ACDCDataset(Dataset):

    def __init__(self, root, size=(128, 128)):
        self.root = Path(root)
        self.size = size
        self.samples = []

        for patient_dir in sorted(self.root.glob("patient*")):

            for frame in ["frame01", "frame12"]:

                image_path = patient_dir / f"{patient_dir.name}_{frame}.nii.gz"
                mask_path = patient_dir / f"{patient_dir.name}_{frame}_gt.nii.gz"

                if not image_path.exists() or not mask_path.exists():
                    continue

                mask = nib.load(mask_path).get_fdata()

                for slice_idx in range(mask.shape[2]):

                    slice_mask = mask[:, :, slice_idx]

                    if np.any(slice_mask > 0):
                        self.samples.append(
                            (image_path, mask_path, slice_idx)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, mask_path, slice_idx = self.samples[idx]

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        image = image[:, :, slice_idx]
        mask = mask[:, :, slice_idx]

        image = image.astype(np.float32)
        mask = mask.astype(np.uint8)

        image_min = image.min()
        image_max = image.max()

        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)

        image = Image.fromarray(image)
        image = image.resize(self.size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32)

        mask = (mask > 0).astype(np.uint8)

        mask = Image.fromarray(mask)
        mask = mask.resize(self.size, Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)

        image = torch.from_numpy(image)[None]
        mask = torch.from_numpy(mask)[None]

        return image, mask