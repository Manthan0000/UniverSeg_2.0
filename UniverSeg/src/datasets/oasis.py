from pathlib import Path

import nibabel as nib
import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset


def process_img(path, size=(128, 128)):
    img = (nib.load(path).get_fdata() * 255).astype(np.uint8).squeeze()
    img = PIL.Image.fromarray(img)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img).astype(np.float32) / 255
    img = np.rot90(img, -1)
    return img.copy()


def process_seg(path, size=(128, 128)):
    seg = nib.load(path).get_fdata().astype(np.int8).squeeze()
    seg = PIL.Image.fromarray(seg)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg).astype(np.float32)
    seg = np.rot90(seg, -1)
    return seg.copy()


class OASISDataset(Dataset):

    def __init__(self, root, label=1):
        self.root = Path(root)
        self.label = label

        self.files = sorted(
            self.root.glob("*/slice_norm.nii.gz")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        seg_path = Path(
            str(img_path).replace(
                "slice_norm", "slice_seg24"
            )
        )

        img = process_img(img_path)
        seg = process_seg(seg_path)

        img = torch.from_numpy(img)[None]
        seg = torch.from_numpy(
            (seg == self.label).astype(np.float32)
        )[None]

        return img, seg