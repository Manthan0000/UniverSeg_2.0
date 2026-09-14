import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "UniverSeg-main"))

from universeg import universeg
from src.datasets.chaos import CHAOSDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = universeg(pretrained=True).to(device)
model.eval()

dataset = CHAOSDataset(
    Path(__file__).resolve().parents[1] / "data" / "raw" / "CHAOS"
)

support_indices = [100, 200, 300, 400]
query_index = 500

support_images = []
support_labels = []

for idx in support_indices:
    image, mask = dataset[idx]
    support_images.append(image)
    support_labels.append(mask)

support_images = torch.stack(support_images).unsqueeze(0).to(device)
support_labels = torch.stack(support_labels).unsqueeze(0).to(device)

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

print("Device:", device)
print("Dataset size:", len(dataset))
print("Support:", support_images.shape)
print("Query:", query_input.shape)
print(f"CHAOS Dice Score: {dice:.4f}")