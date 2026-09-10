import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "UniverSeg-main"))

from universeg import universeg
from src.datasets.oasis import OASISDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = universeg(pretrained=True)
model = model.to(device)
model.eval()

dataset = OASISDataset(
    root=Path(__file__).resolve().parents[1] / "data" / "raw" / "OASIS",
    label=1
)

support_indices = [0, 1, 2, 3]
query_index = 10

support_images = []
support_labels = []

for idx in support_indices:
    image, mask = dataset[idx]
    support_images.append(image)
    support_labels.append(mask)

query_image, query_mask = dataset[query_index]

support_images = torch.stack(support_images).unsqueeze(0).to(device)
support_labels = torch.stack(support_labels).unsqueeze(0).to(device)
query_image = query_image.unsqueeze(0).to(device)

print("Device:", device)
print("Support images:", support_images.shape)
print("Support labels:", support_labels.shape)
print("Query image:", query_image.shape)

with torch.no_grad():
    prediction = model(
        query_image,
        support_images,
        support_labels
    )

# prediction = prediction.cpu().squeeze().numpy()
# query_image = query_image.cpu().squeeze().numpy()
# query_mask = query_mask.numpy().squeeze()

prediction = prediction.cpu().squeeze().numpy()

prediction_binary = (prediction > 0.5).astype(float)

query_image = query_image.cpu().squeeze().numpy()
query_mask = query_mask.numpy().squeeze()

intersection = (prediction_binary * query_mask).sum()

dice = (2 * intersection) / (
    prediction_binary.sum() + query_mask.sum() + 1e-8
)

print(f"Dice Score: {dice:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(query_image, cmap="gray")
plt.title("Query MRI")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(query_mask, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
# plt.imshow(prediction, cmap="gray")
# plt.title("UniverSeg Prediction")
plt.imshow(prediction_binary, cmap="gray")
plt.title(f"Prediction (Dice: {dice:.4f})")
plt.axis("off")

plt.tight_layout()
plt.show()