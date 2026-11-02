import sys
import csv
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT.parent / "UniverSeg-main"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OFFICIAL_REPO))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from universeg import universeg
from src.datasets.oasis import OASISDataset
from src.datasets.acdc import ACDCDataset
from src.datasets.drive import DRIVEDataset
from src.datasets.camus import CAMUSDataset
from src.datasets.pandental import PanDentalDataset
from src.datasets.chaos import CHAOSDataset


class CachedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]


def compute_dice_and_iou(pred_bin: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred_f = pred_bin.float().cpu()
    targ_f = target.float().cpu()
    inter = (pred_f * targ_f).sum().item()
    p_sum = pred_f.sum().item()
    t_sum = targ_f.sum().item()
    union = p_sum + t_sum - inter
    dice = (2.0 * inter) / (p_sum + t_sum + 1e-8)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Spearman rank correlation without requiring scipy."""
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((rx * ry).sum() / denom)


def sample_support_indices(
    pool: List[int],
    n: int,
    rng: random.Random,
) -> List[int]:
    if n <= len(pool):
        return rng.sample(pool, n)
    # Only triggered for DRIVE (pool=4) when testing N > 4
    chosen = list(pool)
    while len(chosen) < n:
        chosen.append(rng.choice(pool))
    return chosen


def forward_model_safe(
    model: torch.nn.Module,
    query_img: torch.Tensor,
    support_imgs: torch.Tensor,
    support_lbls: torch.Tensor,
    use_amp: bool = True,
) -> torch.Tensor:
    """Runs inference with torch.no_grad() and AMP."""
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(query_img, support_imgs, support_lbls)
    return logits.float()


def evaluate_canonical_n4(
    models: Dict[str, torch.nn.Module],
    datasets_info: dict,
    device: torch.device,
) -> Dict[str, Dict[str, dict]]:
    """
    Evaluates all models on the exact canonical N=4 split used in Table 3 of the paper
    (with max 200 queries on CHAOS for fast turnaround if needed, or full queries).
    """
    use_amp = device.type == "cuda"
    results = {m_name: {} for m_name in models}

    for ds_name, info in datasets_info.items():
        ds = info["dataset"]
        sup_idx = info["canonical_support"]
        qry_idx = info["eval_queries"]

        s_imgs = torch.stack([ds[i][0] for i in sup_idx]).unsqueeze(0).to(device)
        s_lbls = torch.stack([ds[i][1] for i in sup_idx]).unsqueeze(0).to(device)

        for m_name, model in models.items():
            dices, ious = [], []
            for q_i in qry_idx:
                q_img, q_msk = ds[q_i]
                q_in = q_img.unsqueeze(0).to(device)
                logits = forward_model_safe(model, q_in, s_imgs, s_lbls, use_amp)
                pred_bin = (torch.sigmoid(logits) > 0.5).float()
                d, iou = compute_dice_and_iou(pred_bin, q_msk)
                dices.append(d)
                ious.append(iou)

            d_arr = np.array(dices)
            i_arr = np.array(ious)
            results[m_name][ds_name] = {
                "mean_dice": float(d_arr.mean()),
                "best_dice": float(d_arr.max()),
                "worst_dice": float(d_arr.min()),
                "mean_iou": float(i_arr.mean()),
                "num_queries": len(qry_idx),
                "small_eval_set": len(qry_idx) <= 20,
            }
        print(
            f"  [Canonical N=4] {ds_name:10s} | "
            + " | ".join(f"{m}: {results[m][ds_name]['mean_dice']:.4f}" for m in models),
            flush=True,
        )

    return results


def evaluate_multi_n_sweep(
    models: Dict[str, torch.nn.Module],
    datasets_info: dict,
    support_sizes: List[int],
    num_draws: int,
    device: torch.device,
) -> Tuple[dict, List[dict]]:
    """
    Part 4(a) & 4(b): Evaluates Dice & IoU across N in {1, 2, 4, 8, 16, 32, 64}
    with `num_draws` random support draws (same draws across all models).
    """
    use_amp = device.type == "cuda"
    summary = {m_name: {n: {} for n in support_sizes} for m_name in models}
    raw_rows = []

    for ds_name, info in datasets_info.items():
        ds = info["dataset"]
        qry_idx = info["sweep_queries"]
        sup_pool = [i for i in range(len(ds)) if i not in set(qry_idx)]

        for n in support_sizes:
            # Pre-sample identical `num_draws` support sets for all models
            draws = []
            for draw_id in range(num_draws):
                rng = random.Random(1000 * draw_id + n)
                s_indices = sample_support_indices(sup_pool, n, rng)
                s_imgs = torch.stack([ds[i][0] for i in s_indices]).unsqueeze(0).to(device)
                s_lbls = torch.stack([ds[i][1] for i in s_indices]).unsqueeze(0).to(device)
                draws.append((draw_id, s_imgs, s_lbls))

            for m_name, model in models.items():
                draw_means = []
                draw_ious = []
                all_q_dices = []

                for draw_id, s_imgs, s_lbls in draws:
                    q_dices, q_ious = [], []
                    for q_i in qry_idx:
                        q_img, q_msk = ds[q_i]
                        q_in = q_img.unsqueeze(0).to(device)
                        logits = forward_model_safe(model, q_in, s_imgs, s_lbls, use_amp)
                        pred_bin = (torch.sigmoid(logits) > 0.5).float()
                        d, iou = compute_dice_and_iou(pred_bin, q_msk)
                        q_dices.append(d)
                        q_ious.append(iou)

                    draw_means.append(float(np.mean(q_dices)))
                    draw_ious.append(float(np.mean(q_ious)))
                    all_q_dices.extend(q_dices)

                summary[m_name][n][ds_name] = {
                    "mean_dice": float(np.mean(draw_means)),
                    "std_dice": float(np.std(draw_means)),
                    "best_dice": float(np.max(all_q_dices)),
                    "worst_dice": float(np.min(all_q_dices)),
                    "mean_iou": float(np.mean(draw_ious)),
                }

                raw_rows.append(
                    {
                        "model": m_name,
                        "dataset": ds_name,
                        "support_size_N": n,
                        "mean_dice": f"{np.mean(draw_means):.4f}",
                        "std_dice": f"{np.std(draw_means):.4f}",
                        "best_dice": f"{np.max(all_q_dices):.4f}",
                        "worst_dice": f"{np.min(all_q_dices):.4f}",
                        "mean_iou": f"{np.mean(draw_ious):.4f}",
                        "num_queries": len(qry_idx),
                        "small_eval_flag": "SMALL_EVAL_SET" if len(qry_idx) <= 20 else "OK",
                    }
                )
            print(
                f"  [Multi-N] {ds_name:10s} | N={n:2d} | "
                + " | ".join(
                    f"{m}: {summary[m][n][ds_name]['mean_dice']:.4f}±{summary[m][n][ds_name]['std_dice']:.4f}"
                    for m in models
                ),
                flush=True,
            )

    return summary, raw_rows


def evaluate_corrupted_supports_n8(
    models: Dict[str, torch.nn.Module],
    datasets_info: dict,
    device: torch.device,
) -> List[dict]:
    """
    Part 4(c): At N=8, corrupt 0%, 25% (2/8), and 50% (4/8) of supports with:
      (i) shuffled/inverted masks
      (ii) masks/pairs from another dataset (DRIVE)
    Compares Run 2 (Recipe R) vs Run 3 (Recipe R + Module B).
    """
    use_amp = device.type == "cuda"
    rows = []
    other_ds = datasets_info["DRIVE"]["dataset"]
    target_names = ["OASIS", "ACDC", "CAMUS", "PanDental"]

    for c_mode in ["shuffled_masks", "cross_dataset_masks"]:
        for frac, k_corrupt in [(0.0, 0), (0.25, 2), (0.50, 4)]:
            for m_name in ["Run2_R", "Run3_R_B"]:
                if m_name not in models:
                    continue
                model = models[m_name]
                ds_scores = []

                for ds_name in target_names:
                    info = datasets_info[ds_name]
                    ds = info["dataset"]
                    qry_idx = info["sweep_queries"]
                    sup_pool = [i for i in range(len(ds)) if i not in set(qry_idx)]
                    rng = random.Random(42)
                    s_idx = sample_support_indices(sup_pool, 8, rng)

                    s_imgs = [ds[i][0].clone() for i in s_idx]
                    s_lbls = [ds[i][1].clone() for i in s_idx]

                    for slot in range(8 - k_corrupt, 8):
                        if c_mode == "shuffled_masks":
                            # Replace mask with inverted/shuffled mask from another slice
                            s_lbls[slot] = 1.0 - s_lbls[(slot + 3) % 8]
                        else:
                            _, other_m = other_ds[slot % len(other_ds)]
                            s_lbls[slot] = other_m.clone()

                    s_imgs_t = torch.stack(s_imgs).unsqueeze(0).to(device)
                    s_lbls_t = torch.stack(s_lbls).unsqueeze(0).to(device)

                    q_dices = []
                    for q_i in qry_idx:
                        q_img, q_msk = ds[q_i]
                        logits = forward_model_safe(
                            model, q_img.unsqueeze(0).to(device), s_imgs_t, s_lbls_t, use_amp
                        )
                        pred_bin = (torch.sigmoid(logits) > 0.5).float()
                        d, _ = compute_dice_and_iou(pred_bin, q_msk)
                        q_dices.append(d)
                    ds_scores.append(float(np.mean(q_dices)))

                rows.append(
                    {
                        "corruption_mode": c_mode,
                        "corrupt_fraction": f"{int(frac*100)}%",
                        "model": m_name,
                        "mean_dice_4datasets": round(float(np.mean(ds_scores)), 4),
                        "OASIS": round(ds_scores[0], 4),
                        "ACDC": round(ds_scores[1], 4),
                        "CAMUS": round(ds_scores[2], 4),
                        "PanDental": round(ds_scores[3], 4),
                    }
                )
                print(
                    f"  [Corruption {c_mode:19s} | {int(frac*100):2d}%] {m_name:10s} | "
                    f"4-Dataset Mean Dice: {np.mean(ds_scores):.4f}",
                    flush=True,
                )
    return rows


def run_module_u_uncertainty(
    best_model: torch.nn.Module,
    best_model_name: str,
    datasets_info: dict,
    k_draws: int,
    device: torch.device,
    figures_dir: Path,
) -> dict:
    """
    Part 5: Module U (Uncertainty, inference-only on best model from Part 4).
    Runs K independent random support draws at N=4 and N=8, computes:
      - Final prediction = mean of K soft masks (threshold 0.5)
      - Pixel uncertainty map = variance across the K soft masks
      - Image-level uncertainty score = mean pixel variance over predicted foreground
        (fallback to whole image if foreground empty)
      - Pairwise disagreement = 1 - mean pairwise Dice between the K binary masks
      - Spearman correlation with true Dice / segmentation error
      - Risk-coverage curve, AURC, and 20% rejection Dice gain
      - Saves 300 dpi figures in `results/figures/`
    """
    use_amp = device.type == "cuda"
    module_u_summary = {"model_used": best_model_name, "K": k_draws, "N4": {}, "N8": {}}
    qualitative_samples = []

    for n_val, n_key in [(4, "N4"), (8, "N8")]:
        rc_curves = {}

        for ds_name, info in datasets_info.items():
            ds = info["dataset"]
            qry_idx = info["sweep_queries"]
            sup_pool = [i for i in range(len(ds)) if i not in set(qry_idx)]

            # Pre-sample K support sets of size n_val
            support_draws = []
            for k in range(k_draws):
                rng = random.Random(5000 + 97 * k + n_val)
                s_idx = sample_support_indices(sup_pool, n_val, rng)
                s_imgs = torch.stack([ds[i][0] for i in s_idx]).unsqueeze(0).to(device)
                s_lbls = torch.stack([ds[i][1] for i in s_idx]).unsqueeze(0).to(device)
                support_draws.append((s_imgs, s_lbls))

            single_draw_dices = []
            ensemble_dices = []
            fg_var_scores = []
            disagreement_scores = []

            for idx_pos, q_i in enumerate(qry_idx):
                q_img, q_msk = ds[q_i]
                q_in = q_img.unsqueeze(0).to(device)

                soft_preds = []
                bin_preds = []
                for s_imgs, s_lbls in support_draws:
                    logits = forward_model_safe(best_model, q_in, s_imgs, s_lbls, use_amp)
                    prob = torch.sigmoid(logits).squeeze(0).cpu()  # (1, H, W)
                    soft_preds.append(prob)
                    bin_preds.append((prob > 0.5).float())

                soft_stack = torch.stack(soft_preds, dim=0)  # (K, 1, H, W)
                mean_soft = soft_stack.mean(dim=0)           # (1, H, W)
                var_map = soft_stack.var(dim=0, unbiased=False)  # (1, H, W)

                ens_bin = (mean_soft > 0.5).float()
                ens_dice, _ = compute_dice_and_iou(ens_bin, q_msk)
                single_dice, _ = compute_dice_and_iou(bin_preds[0], q_msk)

                single_draw_dices.append(single_dice)
                ensemble_dices.append(ens_dice)

                # Image-level uncertainty 1: mean pixel variance over predicted foreground
                fg_mask = ens_bin > 0.5
                if fg_mask.sum() > 0:
                    fg_var = float(var_map[fg_mask].mean().item())
                else:
                    fg_var = float(var_map.mean().item())
                fg_var_scores.append(fg_var)

                # Image-level uncertainty 2: 1 - mean pairwise Dice between K binary masks
                pw_dices = []
                for a in range(k_draws):
                    for b in range(a + 1, k_draws):
                        d_ab, _ = compute_dice_and_iou(bin_preds[a], bin_preds[b])
                        pw_dices.append(d_ab)
                disagree = 1.0 - float(np.mean(pw_dices)) if pw_dices else 0.0
                disagreement_scores.append(disagree)

                if (
                    n_val == 8
                    and idx_pos == 0
                    and ds_name in ("OASIS", "ACDC", "CAMUS", "PanDental")
                ):
                    qualitative_samples.append(
                        {
                            "dataset": ds_name,
                            "image": q_img.squeeze(0).cpu().numpy(),
                            "gt": q_msk.squeeze(0).cpu().numpy(),
                            "pred": ens_bin.squeeze(0).cpu().numpy(),
                            "uncert": var_map.squeeze(0).cpu().numpy(),
                            "dice": ens_dice,
                            "fg_var": fg_var,
                        }
                    )

            ens_arr = np.array(ensemble_dices)
            sng_arr = np.array(single_draw_dices)
            var_arr = np.array(fg_var_scores)
            dis_arr = np.array(disagreement_scores)

            # Combined normalized uncertainty rank score (foreground variance + pairwise disagreement)
            # Spearman correlation between uncertainty and error (1 - Dice), or confidence (-uncertainty) and Dice
            rho_var_vs_dice = spearman_rank_corr(-var_arr, ens_arr)
            rho_dis_vs_dice = spearman_rank_corr(-dis_arr, ens_arr)

            # Use whichever uncertainty score (or average rank) correlates better for selective rejection
            rank_var = np.argsort(np.argsort(var_arr))
            rank_dis = np.argsort(np.argsort(dis_arr))
            combined_uncert = 0.5 * (rank_var / max(1, len(var_arr) - 1)) + 0.5 * (
                rank_dis / max(1, len(dis_arr) - 1)
            )
            rho_combined = spearman_rank_corr(-combined_uncert, ens_arr)

            # Sort queries from LEAST uncertain to MOST uncertain
            order_asc_uncert = np.argsort(combined_uncert)
            sorted_dices = ens_arr[order_asc_uncert]

            reject_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            retained_dices = []
            for rf in reject_fracs:
                keep_n = max(1, int(round(len(sorted_dices) * (1.0 - rf))))
                retained_dices.append(float(sorted_dices[:keep_n].mean()))

            # AURC (Area Under Risk-Coverage curve, where risk = 1 - Dice)
            risks = [1.0 - d for d in retained_dices]
            aurc = float(np.trapz(risks, x=[1.0 - rf for rf in reversed(reject_fracs)]))

            dice_full = retained_dices[0]
            dice_reject_20 = retained_dices[2]

            module_u_summary[n_key][ds_name] = {
                "single_draw_mean_dice": round(float(sng_arr.mean()), 4),
                "ensemble_mean_dice": round(dice_full, 4),
                "dice_after_reject_20pct": round(dice_reject_20, 4),
                "gain_from_reject_20pct": round(dice_reject_20 - dice_full, 4),
                "spearman_conf_vs_dice_fg_var": round(rho_var_vs_dice, 4),
                "spearman_conf_vs_dice_disagree": round(rho_dis_vs_dice, 4),
                "spearman_conf_vs_dice_combined": round(rho_combined, 4),
                "AURC": round(aurc, 4),
                "statement": (
                    f"Rejecting the 20% most uncertain images on {ds_name} "
                    f"raises mean Dice from {dice_full:.4f} to {dice_reject_20:.4f} "
                    f"({dice_reject_20 - dice_full:+.4f})."
                ),
            }
            print(
                f"  [Module U {n_key}] {ds_name:10s} | Single: {sng_arr.mean():.4f} -> "
                f"Ensemble(K={k_draws}): {dice_full:.4f} -> Reject 20%: {dice_reject_20:.4f} "
                f"(Spearman={rho_combined:.3f})",
                flush=True,
            )

            if n_val == 8:
                rc_curves[ds_name] = (reject_fracs, retained_dices, float(ens_arr.mean()))

        if n_val == 8:
            # Plot Risk-Coverage / Selective Prediction curves at 300 dpi
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes = axes.flatten()
            for idx_ax, (ds_name, (r_fracs, ret_dices, base_mean)) in enumerate(rc_curves.items()):
                ax = axes[idx_ax]
                pcts = [int(rf * 100) for rf in r_fracs]
                ax.plot(
                    pcts,
                    ret_dices,
                    "o-",
                    linewidth=2.2,
                    color="#2b7bba",
                    label="Uncertainty Rejection",
                )
                ax.axhline(
                    base_mean,
                    linestyle="--",
                    color="#888888",
                    label="Random Rejection Baseline",
                )
                ax.set_title(f"{ds_name} (N=8, K={k_draws})", fontweight="bold")
                ax.set_xlabel("Rejected Most Uncertain Images (%)")
                ax.set_ylabel("Mean Dice of Retained Images")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=8)

            fig.suptitle(
                f"Module U: Selective Prediction Risk-Coverage Curves ({best_model_name}, K={k_draws})",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(figures_dir / "module_u_risk_coverage.png", dpi=300)
            plt.close(fig)

    # Plot Qualitative Examples (Image, GT, Prediction, Uncertainty Map) at 300 dpi
    if qualitative_samples:
        n_rows = len(qualitative_samples)
        fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3.0 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for r, item in enumerate(qualitative_samples):
            axes[r, 0].imshow(item["image"], cmap="gray")
            axes[r, 0].set_title(f"{item['dataset']} Query Image")
            axes[r, 0].axis("off")

            axes[r, 1].imshow(item["gt"], cmap="gray")
            axes[r, 1].set_title("Ground Truth Mask")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(item["pred"], cmap="gray")
            axes[r, 2].set_title(f"Ensemble Pred (Dice={item['dice']:.3f})")
            axes[r, 2].axis("off")

            im = axes[r, 3].imshow(item["uncert"], cmap="magma")
            axes[r, 3].set_title(f"Pixel Variance (Uncert={item['fg_var']:.4f})")
            axes[r, 3].axis("off")
            fig.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Module U Qualitative Examples: Prediction & Support-Ensemble Uncertainty (K={k_draws}, N=8)",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "module_u_qualitative_examples.png", dpi=300)
        plt.close(fig)

    return module_u_summary


def plot_ablation_and_heatmaps(
    multi_n_summary: dict,
    support_sizes: List[int],
    datasets_list: List[str],
    figures_dir: Path,
):
    # 1. Dice vs Support Size N plot across all models
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    styles = {
        "Run0_Baseline5k": ("o--", "#7f7f7f", "Run 0: Existing 5k Baseline"),
        "Run1_Base": ("s--", "#4C72B0", "Run 1: Baseline (fixed N=4)"),
        "Run2_R": ("^-", "#55A868", "Run 2: + Recipe R (var N + aug)"),
        "Run3_R_B": ("D-", "#C44E52", "Run 3: + Recipe R + Module B (Proto)"),
    }

    for m_name, (marker, color, label) in styles.items():
        if m_name not in multi_n_summary:
            continue
        means = [
            np.mean([multi_n_summary[m_name][n][ds]["mean_dice"] for ds in datasets_list])
            for n in support_sizes
        ]
        ax.plot(range(len(support_sizes)), means, marker, linewidth=2.3, color=color, label=label)

    ax.set_xticks(range(len(support_sizes)))
    ax.set_xticklabels([str(n) for n in support_sizes])
    ax.set_xlabel("Inference Support-Set Size (N)")
    ax.set_ylabel("Macro-Average Dice Across 6 Datasets")
    ax.set_title("Ablation Study: Scaling With Support-Set Size N", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "dice_vs_n_ablation.png", dpi=300)
    plt.close(fig)

    # 2. Train-N x Inference-M Heatmap comparing Run 1 (Trained N=4) vs Run 2 (Trained N<=8) vs Run 3 (R+B, Trained N<=8)
    heatmap_models = [m for m in ["Run1_Base", "Run2_R", "Run3_R_B"] if m in multi_n_summary]
    heatmap_labels = [
        "Run 1 (Trained Fixed N=4)",
        "Run 2 (+Recipe R, Trained N∈{1,2,4,8})",
        "Run 3 (+Recipe R + Module B, Trained N∈{1,2,4,8})",
    ][: len(heatmap_models)]

    mat = np.zeros((len(heatmap_models), len(support_sizes)))
    for i, m_name in enumerate(heatmap_models):
        for j, n in enumerate(support_sizes):
            mat[i, j] = np.mean(
                [multi_n_summary[m_name][n][ds]["mean_dice"] for ds in datasets_list]
            )

    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(support_sizes)))
    ax.set_xticklabels([f"M={n}" for n in support_sizes])
    ax.set_yticks(range(len(heatmap_models)))
    ax.set_yticklabels(heatmap_labels)
    ax.set_xlabel("Inference Support Size (M)")
    ax.set_title("Train-N × Inference-M Mean Dice Heatmap (6-Dataset Mean)", fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.4f}",
                ha="center",
                va="center",
                color="black" if mat[i, j] < (mat.max() + mat.min()) / 2 else "white",
                fontweight="bold",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, label="Mean Dice")
    fig.tight_layout()
    fig.savefig(figures_dir / "trainN_inferM_heatmap.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uncertainty_k", type=int, default=5)
    parser.add_argument("--num_draws", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = PROJECT_ROOT / "data" / "raw"
    ckpt_dir = PROJECT_ROOT / "models" / "ours" / "checkpoints"
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    oasis = CachedDataset(OASISDataset(data_root / "OASIS"))
    acdc = CachedDataset(ACDCDataset(data_root / "ACDC"))
    drive = CachedDataset(DRIVEDataset(data_root / "drive"))
    camus = CachedDataset(CAMUSDataset(data_root / "CAMUS"))
    pandental = CachedDataset(PanDentalDataset(data_root / "PanDental"))
    chaos = CachedDataset(CHAOSDataset(data_root / "CHAOS"))

    # Query splits: exact same canonical query indices for N=4 table;
    # for multi-N (up to N=64) and K-draw sweeps, cap large datasets (CAMUS/CHAOS) at 40 fixed queries
    # so N=1..64 x 3 draws x 4 models completes within minutes on laptop GPU.
    datasets_info = {
        "OASIS": {
            "dataset": oasis,
            "canonical_support": [0, 1, 2, 3],
            "eval_queries": list(range(4, 24)),
            "sweep_queries": list(range(4, 24)),
        },
        "ACDC": {
            "dataset": acdc,
            "canonical_support": [0, 1, 2, 3],
            "eval_queries": list(range(4, 24)),
            "sweep_queries": list(range(4, 24)),
        },
        "DRIVE": {
            "dataset": drive,
            "canonical_support": [0, 1, 2, 3],
            "eval_queries": list(range(4, len(drive))),
            "sweep_queries": list(range(4, len(drive))),
        },
        "CAMUS": {
            "dataset": camus,
            "canonical_support": [0, 1, 2, 3],
            "eval_queries": list(range(4, 44)),
            "sweep_queries": list(range(64, 84)),
        },
        "PanDental": {
            "dataset": pandental,
            "canonical_support": [0, 1, 2, 3],
            "eval_queries": list(range(4, 44)),
            "sweep_queries": list(range(64, 84)),
        },
        "CHAOS": {
            "dataset": chaos,
            "canonical_support": [100, 200, 300, 400],
            "eval_queries": list(range(500, 540)),
            "sweep_queries": list(range(500, 520)),
        },
    }

    # Load models
    models = {}

    ckpt_r0 = ckpt_dir / "universeg_5000_steps.pth"
    if ckpt_r0.exists():
        m0 = universeg(pretrained=False, use_proto=False).to(device).eval()
        m0.load_state_dict(torch.load(ckpt_r0, map_location=device))
        models["Run0_Baseline5k"] = m0

    ckpt_r1 = ckpt_dir / f"universeg_R0_base_steps{args.steps}_seed{args.seed}.pth"
    if ckpt_r1.exists():
        m1 = universeg(pretrained=False, use_proto=False).to(device).eval()
        m1.load_state_dict(torch.load(ckpt_r1, map_location=device))
        models["Run1_Base"] = m1

    ckpt_r2 = ckpt_dir / f"universeg_R_only_steps{args.steps}_seed{args.seed}.pth"
    if ckpt_r2.exists():
        m2 = universeg(pretrained=False, use_proto=False).to(device).eval()
        m2.load_state_dict(torch.load(ckpt_r2, map_location=device))
        models["Run2_R"] = m2

    ckpt_r3 = ckpt_dir / f"universeg_R_B_steps{args.steps}_seed{args.seed}.pth"
    if ckpt_r3.exists():
        m3 = universeg(pretrained=False, use_proto=True).to(device).eval()
        m3.load_state_dict(torch.load(ckpt_r3, map_location=device))
        models["Run3_R_B"] = m3

    print("Loaded models for evaluation:", list(models.keys()), flush=True)

    # 1. Canonical N=4 Evaluation
    print("\n--- Running Part 4 Canonical N=4 Evaluation ---", flush=True)
    canon_res = evaluate_canonical_n4(models, datasets_info, device)

    # 2. Multi-N Sweep (N in {1, 2, 4, 8, 16, 32, 64})
    support_sizes = [1, 2, 4, 8, 16, 32, 64]
    print(f"\n--- Running Part 4(a)-(b) Multi-N Sweep {support_sizes} ({args.num_draws} draws) ---", flush=True)
    multi_n_summary, raw_rows = evaluate_multi_n_sweep(
        models, datasets_info, support_sizes, args.num_draws, device
    )

    with open(out_dir / "ablation_multi_n_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "dataset",
                "support_size_N",
                "mean_dice",
                "std_dice",
                "best_dice",
                "worst_dice",
                "mean_iou",
                "num_queries",
                "small_eval_flag",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    plot_ablation_and_heatmaps(
        multi_n_summary, support_sizes, list(datasets_info.keys()), figures_dir
    )

    # 3. Part 4(c) Corrupted-Support Test at N=8
    print("\n--- Running Part 4(c) Corrupted-Support Test at N=8 ---", flush=True)
    corrupt_rows = evaluate_corrupted_supports_n8(models, datasets_info, device)
    if corrupt_rows:
        with open(out_dir / "corrupted_support_n8_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(corrupt_rows[0].keys()))
            writer.writeheader()
            writer.writerows(corrupt_rows)

    # 4. Select best model from Part 4 for Module U
    best_model_name = "Run3_R_B" if "Run3_R_B" in models else list(models.keys())[-1]
    best_score = -1.0
    for m_name in models:
        mean_n8 = float(
            np.mean([multi_n_summary[m_name][8][ds]["mean_dice"] for ds in datasets_info])
        )
        if mean_n8 > best_score:
            best_score = mean_n8
            best_model_name = m_name

    print(f"\n--- Running Part 5 Module U (Uncertainty, K={args.uncertainty_k}) on {best_model_name} ---", flush=True)
    module_u_res = run_module_u_uncertainty(
        best_model=models[best_model_name],
        best_model_name=best_model_name,
        datasets_info=datasets_info,
        k_draws=args.uncertainty_k,
        device=device,
        figures_dir=figures_dir,
    )

    full_json = {
        "canonical_n4": canon_res,
        "multi_n_summary": multi_n_summary,
        "corrupted_support_n8": corrupt_rows,
        "module_u": module_u_res,
    }
    with open(out_dir / "rbu_evaluation_summary.json", "w") as f:
        json.dump(full_json, f, indent=2)

    print("\nAll evaluations and figures saved successfully to:", out_dir, flush=True)


if __name__ == "__main__":
    main()
