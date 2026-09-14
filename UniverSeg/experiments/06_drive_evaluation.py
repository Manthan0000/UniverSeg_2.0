import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "UniverSeg-main"))

from universeg import universeg
from src.datasets.drive import DRIVEDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = universeg(pretrained=True).to(device)
model.eval()

dataset = DRIVEDataset(
    Path(__file__).resolve().parents[1] / "data" / "raw" / "drive"
)

support_indices = [0, 1, 2, 3]

support_images = []
support_labels = []

for idx in support_indices:
    image, mask = dataset[idx]
    support_images.append(image)
    support_labels.append(mask)

support_images = torch.stack(support_images).unsqueeze(0).to(device)
support_labels = torch.stack(support_labels).unsqueeze(0).to(device)

scores = []

for query_index in range(4, 20):
    query_image, query_mask = dataset[query_index]
    query_input = query_image.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(
            query_input,
            support_images,
            support_labels
        )

    prediction = prediction.cpu().squeeze().numpy()
    prediction_binary = (prediction > 0.5).astype(float)
    query_mask = query_mask.squeeze().numpy()

    intersection = (prediction_binary * query_mask).sum()

    dice = (2 * intersection) / (
        prediction_binary.sum() + query_mask.sum() + 1e-8
    )

    scores.append(dice)

    print(f"Query {query_index}: {dice:.4f}")

print(f"\nMean Dice: {sum(scores) / len(scores):.4f}")
print(f"Best Dice: {max(scores):.4f}")
print(f"Worst Dice: {min(scores):.4f}")