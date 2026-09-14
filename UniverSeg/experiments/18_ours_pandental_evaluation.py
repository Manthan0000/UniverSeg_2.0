import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "UniverSeg-main"))

from universeg import universeg
from src.datasets.pandental import PanDentalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path(__file__).resolve().parents[1]

dataset = PanDentalDataset(root / "data/raw/PanDental")

model = universeg(pretrained=False).to(device)

checkpoint = root / "models/ours/checkpoints/universeg_5000_steps.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))

model.eval()

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

for i in range(4, len(dataset)):

    query_image, query_mask = dataset[i]

    query_input = query_image.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(
            query_input,
            support_images,
            support_labels
        )

    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float().cpu()

    intersection = (prediction * query_mask).sum()

    dice = (2 * intersection) / (
        prediction.sum() + query_mask.sum() + 1e-8
    )

    scores.append(dice.item())

scores = np.array(scores)

print("Dataset: PanDental")
print("Queries:", len(scores))
print(f"Mean Dice: {scores.mean():.4f}")
print(f"Best Dice: {scores.max():.4f}")
print(f"Worst Dice: {scores.min():.4f}")