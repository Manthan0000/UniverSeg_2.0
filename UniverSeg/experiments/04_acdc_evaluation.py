import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "UniverSeg-main"))

from universeg import universeg
from src.datasets.acdc import ACDCDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = universeg(pretrained=True).to(device)
model.eval()

dataset = ACDCDataset(
    Path(__file__).resolve().parents[1] / "data" / "raw" / "ACDC"
)

support_indices = [0, 1, 2, 3]
query_indices = list(range(100, 120))

support_images = []
support_labels = []

for idx in support_indices:
    image, mask = dataset[idx]
    support_images.append(image)
    support_labels.append(mask)

support_images = torch.stack(support_images).unsqueeze(0).to(device)
support_labels = torch.stack(support_labels).unsqueeze(0).to(device)

dice_scores = []

for idx in query_indices:

    query_image, query_mask = dataset[idx]
    query_input = query_image.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(
            query_input,
            support_images,
            support_labels
        )

    prediction = prediction.cpu().squeeze().numpy()
    prediction = (prediction > 0.5).astype(float)

    query_mask = query_mask.squeeze().numpy()

    intersection = (prediction * query_mask).sum()

    dice = (2 * intersection) / (
        prediction.sum() + query_mask.sum() + 1e-8
    )

    dice_scores.append(dice)

    print(f"Query {idx}: Dice = {dice:.4f}")


print()
print("=" * 40)
print(f"Number of queries: {len(dice_scores)}")
print(f"Mean Dice: {sum(dice_scores) / len(dice_scores):.4f}")
print(f"Best Dice: {max(dice_scores):.4f}")
print(f"Worst Dice: {min(dice_scores):.4f}")
print("=" * 40)