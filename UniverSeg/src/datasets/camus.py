from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CAMUSDataset(Dataset):
    def __init__(self, root, size=(128, 128)):
        self.root = Path(root)
        self.size = size
        self.samples = []

        for patient in sorted(self.root.glob("patient*")):
            for view in ["2CH", "4CH"]:
                for phase in ["ED", "ES"]:
                    image_path = patient / f"{patient.name}_{view}_{phase}.nii.gz"
                    mask_path = patient / f"{patient.name}_{view}_{phase}_gt.nii.gz"

                    if image_path.exists() and mask_path.exists():
                        self.samples.append((image_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = nib.load(image_path).get_fdata().squeeze()
        mask = nib.load(mask_path).get_fdata().squeeze()

        image = image.astype(np.float32)
        mask = mask.astype(np.uint8)

        image_min, image_max = image.min(), image.max()

        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)

        image = Image.fromarray(image)
        mask = Image.fromarray((mask > 0).astype(np.uint8))

        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        image = torch.from_numpy(image)[None]
        mask = torch.from_numpy(mask)[None]

        return image, mask