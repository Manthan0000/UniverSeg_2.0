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


def compute_dice(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float().cpu()
    intersection = (prediction * target).sum()
    dice = (2.0 * intersection) / (prediction.sum() + target.sum() + 1e-8)
    return dice.item()


def get_mean_per_support_weights(model, support_size: int = 4) -> np.ndarray:
    """
    Aggregates the learned attention weights across all 7 CrossBlocks
    into a single S-dimensional weight profile (averaging spatially if per-pixel).
    """
    attn_dict = model.get_attention_weights()
    block_vecs = []
    for w in attn_dict.values():
        if w is None:
            continue
        # w is (1, S, 1, H, W) or (1, S, 1, 1, 1)
        vec = w.detach().mean(dim=(0, 2, 3, 4)).cpu().numpy()
        block_vecs.append(vec)
    if not block_vecs:
        return np.ones(support_size) / support_size
    return np.mean(block_vecs, axis=0)


def evaluate_with_corruption(
    models: dict,
    target_dataset,
    other_dataset,
    support_indices: list,
    query_indices: list,
    num_corrupted: int,
    corruption_type: str,
    device: torch.device,
):
    """
    Evaluates Baseline, Per-Image, and Per-Pixel models when `num_corrupted`
    out of S=4 support examples are corrupted via:
      - 'wrong_mask': invert binary mask (1 - mask)
      - 'wrong_domain': replace support (image, mask) with sample from `other_dataset`
    """
    support_images = []
    support_labels = []

    for idx in support_indices:
        img, msk = target_dataset[idx]
        support_images.append(img.clone())
        support_labels.append(msk.clone())

    # Corrupt the last `num_corrupted` support examples (e.g. index 3, then 2, then 1)
    S = len(support_indices)
    corrupt_slots = list(range(S - num_corrupted, S))

    for slot in corrupt_slots:
        if corruption_type == "wrong_mask":
            # Flip foreground/background mask (1 - mask)
            support_labels[slot] = 1.0 - support_labels[slot]
        elif corruption_type == "wrong_domain":
            # Replace with image-mask pair from a completely different imaging modality/anatomy
            other_img, other_msk = other_dataset[slot % len(other_dataset)]
            support_images[slot] = other_img.clone()
            support_labels[slot] = other_msk.clone()

    support_images_t = torch.stack(support_images).unsqueeze(0).to(device)
    support_labels_t = torch.stack(support_labels).unsqueeze(0).to(device)

    results = {}
    for model_name, model in models.items():
        scores = []
        weight_profiles = []

        for q_idx in query_indices:
            q_img, q_msk = target_dataset[q_idx]
            q_in = q_img.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(q_in, support_images_t, support_labels_t)

            scores.append(compute_dice(pred, q_msk))
            weight_profiles.append(get_mean_per_support_weights(model, S))

        results[model_name] = {
            "mean_dice": float(np.mean(scores)),
            "std_dice": float(np.std(scores)),
            "mean_weights": np.mean(weight_profiles, axis=0),
        }

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = PROJECT_ROOT / "data" / "raw"
    ckpt_dir = PROJECT_ROOT / "models" / "ours" / "checkpoints"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    oasis = OASISDataset(data_root / "OASIS")
    acdc = ACDCDataset(data_root / "ACDC")
    drive = DRIVEDataset(data_root / "drive")
    camus = CAMUSDataset(data_root / "CAMUS")
    pandental = PanDentalDataset(data_root / "PanDental")

    # Load all 3 models
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

    models = {
        "Baseline_Uniform": m_base,
        "Per_Image_Attention": m_img,
        "Per_Pixel_Attention": m_pix,
    }

    eval_tasks = [
        ("OASIS", oasis, drive, [0, 1, 2, 3], list(range(4, 24))),
        ("ACDC", acdc, drive, [0, 1, 2, 3], list(range(4, 24))),
        ("CAMUS", camus, drive, [0, 1, 2, 3], list(range(4, 54))),
        ("PanDental", pandental, drive, [0, 1, 2, 3], list(range(4, 54))),
    ]

    corruption_types = ["wrong_mask", "wrong_domain"]
    corrupt_counts = [0, 1, 2, 3]

    csv_rows = []
    summary_by_type = {ct: {k: {} for k in corrupt_counts} for ct in corruption_types}
    oasis_weight_log = {}

    for c_type in corruption_types:
        print()
        print("=" * 84)
        print(f"ROBUSTNESS TEST: {c_type.upper()} (Support Size S = 4)")
        print("=" * 84)
        print(
            f"{'Corrupted':10s} | {'Dataset':10s} | {'Uniform Base':13s} | "
            f"{'Per-Image':13s} | {'Per-Pixel':13s} | {'Corrupt Support Weight (Img/Pix)'}"
        )
        print("-" * 84)

        for k in corrupt_counts:
            agg_base, agg_img, agg_pix = [], [], []

            for ds_name, target_ds, other_ds, sup_idx, qry_idx in eval_tasks:
                res = evaluate_with_corruption(
                    models=models,
                    target_dataset=target_ds,
                    other_dataset=other_ds,
                    support_indices=sup_idx,
                    query_indices=qry_idx,
                    num_corrupted=k,
                    corruption_type=c_type,
                    device=device,
                )

                b_dice = res["Baseline_Uniform"]["mean_dice"]
                i_dice = res["Per_Image_Attention"]["mean_dice"]
                p_dice = res["Per_Pixel_Attention"]["mean_dice"]

                agg_base.append(b_dice)
                agg_img.append(i_dice)
                agg_pix.append(p_dice)

                if k > 0:
                    corrupt_w_img = float(np.mean(res["Per_Image_Attention"]["mean_weights"][4 - k :]))
                    corrupt_w_pix = float(np.mean(res["Per_Pixel_Attention"]["mean_weights"][4 - k :]))
                else:
                    corrupt_w_img = 0.25
                    corrupt_w_pix = 0.25

                if ds_name == "OASIS":
                    oasis_weight_log[(c_type, k)] = {
                        "Per_Image": res["Per_Image_Attention"]["mean_weights"],
                        "Per_Pixel": res["Per_Pixel_Attention"]["mean_weights"],
                    }

                print(
                    f"{k}/4        | {ds_name:10s} | {b_dice:13.4f} | "
                    f"{i_dice:13.4f} | {p_dice:13.4f} | "
                    f"{corrupt_w_img:.4f} / {corrupt_w_pix:.4f}"
                )

                csv_rows.append(
                    {
                        "Corruption_Type": c_type,
                        "Num_Corrupted": k,
                        "Dataset": ds_name,
                        "Baseline_Uniform": f"{b_dice:.4f}",
                        "Per_Image_Attention": f"{i_dice:.4f}",
                        "Per_Pixel_Attention": f"{p_dice:.4f}",
                        "Corrupted_Support_Weight_PerImage": f"{corrupt_w_img:.4f}",
                        "Corrupted_Support_Weight_PerPixel": f"{corrupt_w_pix:.4f}",
                    }
                )

            mean_b = float(np.mean(agg_base))
            mean_i = float(np.mean(agg_img))
            mean_p = float(np.mean(agg_pix))
            summary_by_type[c_type][k] = {
                "Baseline": mean_b,
                "Per_Image": mean_i,
                "Per_Pixel": mean_p,
            }
            print("-" * 84)
            print(
                f"{k}/4 MEAN   | {'ALL 4':10s} | {mean_b:13.4f} | "
                f"{mean_i:13.4f} | {mean_p:13.4f} |"
            )
            print("-" * 84)

            csv_rows.append(
                {
                    "Corruption_Type": c_type,
                    "Num_Corrupted": k,
                    "Dataset": "MEAN_4_DATASETS",
                    "Baseline_Uniform": f"{mean_b:.4f}",
                    "Per_Image_Attention": f"{mean_i:.4f}",
                    "Per_Pixel_Attention": f"{mean_p:.4f}",
                    "Corrupted_Support_Weight_PerImage": "-",
                    "Corrupted_Support_Weight_PerPixel": "-",
                }
            )

    csv_path = metrics_dir / "robustness_corruption_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Corruption_Type",
                "Num_Corrupted",
                "Dataset",
                "Baseline_Uniform",
                "Per_Image_Attention",
                "Per_Pixel_Attention",
                "Corrupted_Support_Weight_PerImage",
                "Corrupted_Support_Weight_PerPixel",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved robustness CSV to: {csv_path}")

    # ---------------------------------------------------------
    # Plot 1: Robustness Degradation Curves (Wrong Mask & Wrong Domain)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    titles = {
        "wrong_mask": "A. Wrong-Mask Corruption (Inverted Support Masks)",
        "wrong_domain": "B. Wrong-Domain Corruption (Out-of-Domain Supports)",
    }

    for ax, c_type in zip(axes, corruption_types):
        ks = corrupt_counts
        b_curve = [summary_by_type[c_type][k]["Baseline"] for k in ks]
        i_curve = [summary_by_type[c_type][k]["Per_Image"] for k in ks]
        p_curve = [summary_by_type[c_type][k]["Per_Pixel"] for k in ks]

        ax.plot(ks, b_curve, "o--", linewidth=2.2, label="Baseline (Uniform 1/S)", color="#4C72B0")
        ax.plot(ks, i_curve, "s-", linewidth=2.5, label="Per-Image Attention", color="#55A868")
        ax.plot(ks, p_curve, "^-", linewidth=2.5, label="Per-Pixel Attention", color="#DD8452")

        ax.set_title(titles[c_type], fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Corrupted Support Examples (out of S = 4)")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{k}/4 ({k*25}%)" for k in ks])
        ax.set_ylim(0.0, 0.95)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[0].set_ylabel("Mean Dice Score (Averaged Across Datasets)")
    fig.suptitle(
        "Support-Set Robustness: Uniform CrossBlock vs. Attention-Weighted CrossBlock",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    degrad_plot_path = metrics_dir / "04_robustness_corruption_curves.png"
    fig.savefig(degrad_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved degradation curves plot to: {degrad_plot_path}")

    # ---------------------------------------------------------
    # Plot 2: Learned Softmax Weights Analysis (OASIS 1/4 and 2/4 Corrupted)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    scenarios = [
        ("wrong_mask", 0, "Clean Supports (0/4 Corrupted)"),
        ("wrong_mask", 1, "Wrong Mask: Support #4 Corrupted"),
        ("wrong_domain", 2, "Wrong Domain: Supports #3, #4 Corrupted"),
    ]
    x = np.arange(4)
    width = 0.25

    for ax, (c_type, k, title) in zip(axes, scenarios):
        w_unif = np.array([0.25, 0.25, 0.25, 0.25])
        w_img = oasis_weight_log[(c_type, k)]["Per_Image"]
        w_pix = oasis_weight_log[(c_type, k)]["Per_Pixel"]

        ax.bar(x - width, w_unif, width, label="Uniform Baseline", color="#4C72B0", alpha=0.85)
        ax.bar(x, w_img, width, label="Per-Image Attention", color="#55A868")
        ax.bar(x + width, w_pix, width, label="Per-Pixel Attention (Mean)", color="#DD8452")

        ax.axhline(0.25, color="gray", linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        support_labels_txt = [
            f"Sup #{i+1}\n({'Corrupt' if i >= 4 - k else 'Clean'})" for i in range(4)
        ]
        ax.set_xticklabels(support_labels_txt)
        ax.set_ylim(0.0, 0.65)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Learned Softmax Weight $\\alpha_i$")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(
        "Learned Attention Weight Redistribution Under Support Corruption (OASIS)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    weight_plot_path = metrics_dir / "05_attention_weight_analysis.png"
    fig.savefig(weight_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved attention weight redistribution plot to: {weight_plot_path}")


if __name__ == "__main__":
    main()
