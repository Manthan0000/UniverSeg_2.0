import sys
import csv
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT.parent / "UniverSeg-main"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OFFICIAL_REPO))

from universeg import universeg
from src.datasets.oasis import OASISDataset
from src.datasets.acdc import ACDCDataset
from src.datasets.drive import DRIVEDataset
from src.datasets.camus import CAMUSDataset
from src.datasets.pandental import PanDentalDataset
from src.datasets.chaos import CHAOSDataset


def compute_dice(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float().cpu()
    intersection = (prediction * target).sum()
    dice = (2.0 * intersection) / (prediction.sum() + target.sum() + 1e-8)
    return dice.item()


def evaluate_dataset(model, dataset, support_indices, query_indices, device):
    support_images = []
    support_labels = []

    for idx in support_indices:
        image, mask = dataset[idx]
        support_images.append(image)
        support_labels.append(mask)

    support_images = torch.stack(support_images).unsqueeze(0).to(device)
    support_labels = torch.stack(support_labels).unsqueeze(0).to(device)

    scores = []
    for i in query_indices:
        query_image, query_mask = dataset[i]
        query_input = query_image.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(query_input, support_images, support_labels)

        scores.append(compute_dice(prediction, query_mask))

    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "best": float(scores.max()),
        "worst": float(scores.min()),
        "queries": len(scores),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = PROJECT_ROOT / "data" / "raw"
    ckpt_dir = PROJECT_ROOT / "models" / "ours" / "checkpoints"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = [
        (
            "OASIS",
            OASISDataset(data_root / "OASIS"),
            [0, 1, 2, 3],
            lambda n: list(range(4, min(24, n))),
        ),
        (
            "ACDC",
            ACDCDataset(data_root / "ACDC"),
            [0, 1, 2, 3],
            lambda n: list(range(4, min(24, n))),
        ),
        (
            "DRIVE",
            DRIVEDataset(data_root / "drive"),
            [0, 1, 2, 3],
            lambda n: list(range(4, n)),
        ),
        (
            "CAMUS",
            CAMUSDataset(data_root / "CAMUS"),
            [0, 1, 2, 3],
            lambda n: list(range(4, n)),
        ),
        (
            "PanDental",
            PanDentalDataset(data_root / "PanDental"),
            [0, 1, 2, 3],
            lambda n: list(range(4, n)),
        ),
        (
            "CHAOS",
            CHAOSDataset(data_root / "CHAOS"),
            [100, 200, 300, 400],
            lambda n: [i for i in range(n) if i not in {100, 200, 300, 400}],
        ),
    ]

    # Load models
    m_base = universeg(pretrained=False, attention_mode=None).to(device)
    m_base.load_state_dict(
        torch.load(ckpt_dir / "universeg_5000_steps.pth", map_location=device)
    )
    m_base.eval()

    m_img = universeg(pretrained=False, attention_mode="per_image").to(device)
    m_img.load_state_dict(
        torch.load(ckpt_dir / "universeg_attn_per_image_5000_steps.pth", map_location=device)
    )
    m_img.eval()

    m_pix = universeg(pretrained=False, attention_mode="per_pixel").to(device)
    m_pix.load_state_dict(
        torch.load(ckpt_dir / "universeg_attn_per_pixel_5000_steps.pth", map_location=device)
    )
    m_pix.eval()

    rows = []
    print()
    print("=" * 78)
    print(f"{'Dataset':10s} | {'Queries':7s} | {'Uniform Base':12s} | {'Per-Image Attn':14s} | {'Per-Pixel Attn':14s}")
    print("-" * 78)

    for name, ds, sup_idx, qry_fn in datasets_cfg:
        q_idx = qry_fn(len(ds))
        r_base = evaluate_dataset(m_base, ds, sup_idx, q_idx, device)
        r_img = evaluate_dataset(m_img, ds, sup_idx, q_idx, device)
        r_pix = evaluate_dataset(m_pix, ds, sup_idx, q_idx, device)

        print(
            f"{name:10s} | {r_base['queries']:7d} | "
            f"{r_base['mean']:12.4f} | {r_img['mean']:14.4f} | {r_pix['mean']:14.4f}"
        )

        rows.append(
            {
                "Dataset": name,
                "Samples": len(ds),
                "Queries": r_base["queries"],
                "Baseline_Uniform": f"{r_base['mean']:.4f}",
                "Per_Image_Attention": f"{r_img['mean']:.4f}",
                "Per_Pixel_Attention": f"{r_pix['mean']:.4f}",
            }
        )

    print("=" * 78)

    csv_path = metrics_dir / "attention_clean_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Dataset",
                "Samples",
                "Queries",
                "Baseline_Uniform",
                "Per_Image_Attention",
                "Per_Pixel_Attention",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved clean evaluation table to: {csv_path}")

    # Plot bar chart comparison
    labels = [r["Dataset"] for r in rows]
    base_vals = [float(r["Baseline_Uniform"]) for r in rows]
    img_vals = [float(r["Per_Image_Attention"]) for r in rows]
    pix_vals = [float(r["Per_Pixel_Attention"]) for r in rows]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, base_vals, width, label="Baseline (Uniform Mean)", color="#4C72B0")
    ax.bar(x, img_vals, width, label="Per-Image Attention", color="#55A868")
    ax.bar(x + width, pix_vals, width, label="Per-Pixel Attention", color="#DD8452")

    ax.set_ylabel("Mean Dice Score")
    ax.set_title("Clean Support Set Evaluation: Uniform vs. Attention-Weighted CrossBlock")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    plot_path = metrics_dir / "03_attention_clean_comparison.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved clean comparison plot to: {plot_path}")


if __name__ == "__main__":
    main()
