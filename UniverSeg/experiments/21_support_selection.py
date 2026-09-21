import sys
import random
from pathlib import Path

import torch
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT.parent / "UniverSeg-main"

sys.path.insert(0, str(OFFICIAL_REPO))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --------------------------------------------------
# Imports
# --------------------------------------------------

from universeg import universeg
from datasets.oasis import OASISDataset

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data" / "raw"

dataset = OASISDataset(
    DATA_ROOT / "OASIS"
)

print("OASIS samples:", len(dataset))

# --------------------------------------------------
# Model
# --------------------------------------------------

model = universeg(
    pretrained=False
).to(device)

checkpoint_path = (
    PROJECT_ROOT
    / "models"
    / "ours"
    / "checkpoints"
    / "universeg_5000_steps.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.eval()

print("Our 5000-step model loaded.")

# --------------------------------------------------
# Experiment settings
# --------------------------------------------------

SUPPORT_SIZE = 4

# Same queries for both experiments
QUERY_INDICES = list(range(100, 120))

# Previously selected support set
SELECTED_SUPPORT = [12, 72, 267, 339]

# Fixed random seed for reproducibility
random.seed(42)

# Make sure random supports don't overlap queries
available_support = [
    i for i in range(len(dataset))
    if i not in QUERY_INDICES
]

RANDOM_SUPPORT = random.sample(
    available_support,
    SUPPORT_SIZE
)

print()
print("Random support:", RANDOM_SUPPORT)
print("Selected support:", SELECTED_SUPPORT)
print("Queries:", QUERY_INDICES)

# --------------------------------------------------
# Dice function
# --------------------------------------------------

def calculate_dice(prediction, target):

    prediction = torch.sigmoid(
        prediction
    )

    prediction = (
        prediction > 0.5
    ).float()

    intersection = (
        prediction * target
    ).sum()

    dice = (
        2 * intersection
        /
        (
            prediction.sum()
            + target.sum()
            + 1e-8
        )
    )

    return dice.item()

# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_support(support_indices):

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

    scores = []

    for query_idx in QUERY_INDICES:

        query_image, query_label = dataset[
            query_idx
        ]

        query_image = (
            query_image
            .unsqueeze(0)
            .to(device)
        )

        query_label = (
            query_label
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            prediction = model(
                query_image,
                support_images,
                support_labels
            )

        dice = calculate_dice(
            prediction,
            query_label
        )

        scores.append(dice)

    return scores

# --------------------------------------------------
# Random support evaluation
# --------------------------------------------------

print()
print("Evaluating RANDOM support...")

random_scores = evaluate_support(
    RANDOM_SUPPORT
)

random_mean = np.mean(
    random_scores
)

# --------------------------------------------------
# Selected support evaluation
# --------------------------------------------------

print()
print("Evaluating SELECTED support...")

selected_scores = evaluate_support(
    SELECTED_SUPPORT
)

selected_mean = np.mean(
    selected_scores
)

# --------------------------------------------------
# Result
# --------------------------------------------------

improvement = (
    selected_mean
    - random_mean
)
print()
print("=" * 50)
print("SUPPORT SELECTION RESULT")
print("=" * 50)

print(
    f"Random Support Mean Dice   : "
    f"{random_mean:.4f}"
)
print(
    f"Selected Support Mean Dice : "
    f"{selected_mean:.4f}"
)
print(
    f"Improvement                : "
    f"{improvement:+.4f}"
)
print("=" * 50)