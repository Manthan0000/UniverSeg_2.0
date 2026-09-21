import sys
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT.parent / "UniverSeg-main"

sys.path.insert(0, str(OFFICIAL_REPO))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from universeg import universeg

from datasets.oasis import OASISDataset
from datasets.acdc import ACDCDataset
from datasets.drive import DRIVEDataset
from datasets.camus import CAMUSDataset
from datasets.pandental import PanDentalDataset
from datasets.chaos import CHAOSDataset


parser = argparse.ArgumentParser(
    description="Visualize UniverSeg prediction"
)

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "OASIS",
        "ACDC",
        "DRIVE",
        "CAMUS",
        "PanDental",
        "CHAOS"
    ],
    help="Dataset name"
)

parser.add_argument(
    "--query",
    type=int,
    required=True,
    help="Query sample index"
)

args = parser.parse_args()

DATASET_NAME = args.dataset
QUERY_INDEX = args.query

SUPPORT_INDICES = {
    "OASIS": [12, 72, 267, 339],
    "ACDC": [59, 149, 265, 429],
    "DRIVE": [3, 15, 7, 13],
    "CAMUS": [34, 184, 220, 350],
    "PanDental": [30, 62, 108, 111],
    "CHAOS": [1521, 1524, 1529, 1531]
}

support_indices = SUPPORT_INDICES[DATASET_NAME]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = universeg(pretrained=False).to(device)

checkpoint_path = (
    PROJECT_ROOT
    / "models"
    / "ours"
    / "checkpoints"
    / "universeg_5000_steps.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()

print("Our UniverSeg model loaded successfully")


DATA_ROOT = PROJECT_ROOT / "data" / "raw"


if DATASET_NAME == "OASIS":

    dataset = OASISDataset(
        DATA_ROOT / "OASIS"
    )

elif DATASET_NAME == "ACDC":

    dataset = ACDCDataset(
        DATA_ROOT / "ACDC"
    )

elif DATASET_NAME == "DRIVE":

    dataset = DRIVEDataset(
        DATA_ROOT / "drive"
    )

elif DATASET_NAME == "CAMUS":

    dataset = CAMUSDataset(
        DATA_ROOT / "CAMUS"
    )

elif DATASET_NAME == "PanDental":

    dataset = PanDentalDataset(
        DATA_ROOT / "PanDental"
    )

elif DATASET_NAME == "CHAOS":

    dataset = CHAOSDataset(
        DATA_ROOT / "CHAOS"
    )

else:

    raise ValueError(
        f"Unknown dataset: {DATASET_NAME}"
    )


print(
    f"{DATASET_NAME} loaded: {len(dataset)} samples"
)


if QUERY_INDEX < 0 or QUERY_INDEX >= len(dataset):

    raise ValueError(
        f"Query index {QUERY_INDEX} is invalid. "
        f"Valid range: 0 to {len(dataset) - 1}"
    )


for idx in support_indices:

    if idx >= len(dataset):

        raise ValueError(
            f"Support index {idx} is invalid "
            f"for {DATASET_NAME}"
        )


support_images = []
support_labels = []

for idx in support_indices:

    image, mask = dataset[idx]

    support_images.append(image)
    support_labels.append(mask)


support_images = torch.stack(
    support_images
).unsqueeze(0).to(device)

support_labels = torch.stack(
    support_labels
).unsqueeze(0).to(device)


query_image, query_label = dataset[QUERY_INDEX]

query_image = query_image.unsqueeze(0).to(device)
query_label = query_label.unsqueeze(0).to(device)


print("Support:", support_indices)
print("Query:", QUERY_INDEX)

print(
    "Query shape:",
    query_image.shape
)

print(
    "Support shape:",
    support_images.shape
)


with torch.no_grad():

    prediction = model(
        query_image,
        support_images,
        support_labels
    )


print(
    "Prediction shape:",
    prediction.shape
)


probability = torch.sigmoid(
    prediction
)

pred_mask = (
    probability > 0.5
).float()

print(
    "Prediction converted to binary mask"
)


intersection = (
    pred_mask * query_label
).sum()

dice = (
    2 * intersection
    /
    (
        pred_mask.sum()
        + query_label.sum()
        + 1e-6
    )
)

dice_value = dice.item()

print(
    f"Dice Score: {dice_value:.4f}"
)


output_dir = (
    PROJECT_ROOT
    / "results"
    / "predictions"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    output_dir
    / f"{DATASET_NAME}_query_{QUERY_INDEX}.png"
)


plt.figure(
    figsize=(12, 4)
)


# Query image
plt.subplot(1, 3, 1)

plt.imshow(
    query_image[0, 0].cpu(),
    cmap="gray"
)

plt.title(
    "Query Image"
)

plt.axis("off")


# Ground truth
plt.subplot(1, 3, 2)

plt.imshow(
    query_label[0, 0].cpu(),
    cmap="gray"
)

plt.title(
    "Ground Truth"
)

plt.axis("off")


# Prediction
plt.subplot(1, 3, 3)

plt.imshow(
    pred_mask[0, 0].cpu(),
    cmap="gray"
)

plt.title(
    f"Our Prediction\nDice: {dice_value:.4f}"
)

plt.axis("off")


plt.tight_layout()



plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

print(
    f"Saved visualization: {output_path}"
)

plt.show()