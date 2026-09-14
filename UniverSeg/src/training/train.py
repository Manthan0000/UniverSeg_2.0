import sys
import random
from pathlib import Path

import torch
from torch.optim import Adam


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = PROJECT_ROOT.parent / "UniverSeg-main"

sys.path.insert(0, str(OFFICIAL_REPO))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# --------------------------------------------------
# Imports
# --------------------------------------------------

from universeg import universeg

from datasets.oasis import OASISDataset
from datasets.acdc import ACDCDataset
from datasets.drive import DRIVEDataset
from datasets.camus import CAMUSDataset
from datasets.pandental import PanDentalDataset
from datasets.chaos import CHAOSDataset

from tasks.train_tasks import MultiDomainTaskSampler
from training.losses import soft_dice_loss


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Dataset paths
# --------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data" / "raw"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

datasets = {
    "OASIS": OASISDataset(DATA_ROOT / "OASIS"),
    "ACDC": ACDCDataset(DATA_ROOT / "ACDC"),
    "DRIVE": DRIVEDataset(DATA_ROOT / "drive"),
    "CAMUS": CAMUSDataset(DATA_ROOT / "CAMUS"),
    "PanDental": PanDentalDataset(DATA_ROOT / "PanDental"),
    "CHAOS": CHAOSDataset(DATA_ROOT / "CHAOS"),
}


# --------------------------------------------------
# Dataset sizes
# --------------------------------------------------

for name, dataset in datasets.items():
    print(f"{name}: {len(dataset)}")


# --------------------------------------------------
# Task sampler
# --------------------------------------------------

task_sampler = MultiDomainTaskSampler(
    datasets
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = universeg(
    pretrained=False
).to(device)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = Adam(
    model.parameters(),
    lr=1e-4
)


# --------------------------------------------------
# Training settings
# --------------------------------------------------

NUM_STEPS = 5000
SUPPORT_SIZE = 4
CHECKPOINT_INTERVAL = 500


# --------------------------------------------------
# Checkpoint directory
# --------------------------------------------------

checkpoint_dir = (
    PROJECT_ROOT
    / "models"
    / "ours"
    / "checkpoints"
)

checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Training
# --------------------------------------------------

print()
print("Starting improved training...")
print("Datasets:", list(datasets.keys()))
print("Support size:", SUPPORT_SIZE)
print("Steps:", NUM_STEPS)
print()


model.train()

for step in range(1, NUM_STEPS + 1):

    task = task_sampler.sample_task(
        support_size=SUPPORT_SIZE
    )

    support_images = (
        task["support_images"]
        .unsqueeze(0)
        .to(device)
    )

    support_labels = (
        task["support_labels"]
        .unsqueeze(0)
        .to(device)
    )

    query_image = (
        task["query_image"]
        .unsqueeze(0)
        .to(device)
    )

    query_label = (
        task["query_label"]
        .unsqueeze(0)
        .to(device)
    )

    optimizer.zero_grad()

    prediction = model(
        query_image,
        support_images,
        support_labels
    )

    loss = soft_dice_loss(
        prediction,
        query_label
    )

    loss.backward()

    optimizer.step()

    if step % 100 == 0:

        print(
            f"Step {step}/{NUM_STEPS} | "
            f"Dataset: {task['dataset']} | "
            f"Loss: {loss.item():.4f}"
        )

    if step % CHECKPOINT_INTERVAL == 0:

        checkpoint_path = (
            checkpoint_dir
            / f"universeg_improved_{step}_steps.pth"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"Checkpoint saved: {step} steps"
        )


# --------------------------------------------------
# Final checkpoint
# --------------------------------------------------

final_checkpoint = (
    checkpoint_dir
    / "universeg_improved_5000_steps.pth"
)

torch.save(
    model.state_dict(),
    final_checkpoint
)

print()
print("Improved training completed.")
print(
    f"Final checkpoint saved: {final_checkpoint}"
)