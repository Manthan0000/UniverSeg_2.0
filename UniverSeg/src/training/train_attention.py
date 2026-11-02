import sys
import random
import argparse
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

from universeg import universeg
from datasets.oasis import OASISDataset
from datasets.acdc import ACDCDataset
from datasets.drive import DRIVEDataset
from datasets.camus import CAMUSDataset
from datasets.pandental import PanDentalDataset
from datasets.chaos import CHAOSDataset
from tasks.train_tasks import MultiDomainTaskSampler
from training.losses import soft_dice_loss


class CachedDataset:
    """Caches dataset (image, mask) pairs in memory after first disk read for fast training."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]


def corrupt_mask(mask: torch.Tensor, mode: str = "random") -> torch.Tensor:
    """Corrupt a single support mask (1, H, W) to simulate wrong mask annotation."""
    if mode == "random":
        mode = random.choice(["invert", "empty", "noise", "shift"])

    if mode == "invert":
        return 1.0 - mask
    elif mode == "empty":
        return torch.zeros_like(mask)
    elif mode == "noise":
        return (torch.rand_like(mask) > 0.5).float()
    elif mode == "shift":
        shift_h = random.choice([-48, -32, 32, 48])
        shift_w = random.choice([-48, -32, 32, 48])
        return torch.roll(mask, shifts=(shift_h, shift_w), dims=(-2, -1))
    return 1.0 - mask


def sample_robust_task(
    task_sampler: MultiDomainTaskSampler,
    datasets: dict,
    support_size: int = 4,
    corruption_prob: float = 0.65,
):
    """
    Samples a support-query episode. With probability `corruption_prob`,
    corrupts 1 to 2 support examples via either wrong-mask or wrong-domain
    so the attention mechanism learns to down-weight unreliable supports.
    """
    task = task_sampler.sample_task(support_size=support_size)
    dataset_name = task["dataset"]
    support_images = task["support_images"].clone()
    support_labels = task["support_labels"].clone()
    clean_mask = torch.ones(support_size, dtype=torch.float32)

    if random.random() < corruption_prob and support_size >= 2:
        num_corrupt = random.randint(1, max(1, support_size // 2))
        corrupt_indices = random.sample(range(support_size), num_corrupt)
        corruption_type = random.choice(["wrong_mask", "wrong_domain"])

        other_domains = [d for d in datasets.keys() if d != dataset_name]

        for idx in corrupt_indices:
            clean_mask[idx] = 0.0
            if corruption_type == "wrong_mask":
                support_labels[idx] = corrupt_mask(support_labels[idx])
            else:
                other_domain = random.choice(other_domains)
                other_ds = datasets[other_domain]
                rand_i = random.randrange(len(other_ds))
                other_img, other_lbl = other_ds[rand_i]
                support_images[idx] = other_img
                support_labels[idx] = other_lbl

    task["support_images"] = support_images
    task["support_labels"] = support_labels
    task["clean_mask"] = clean_mask
    return task


def compute_attention_clean_mass_loss(
    attn_weights_dict: dict,
    clean_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Encourages total attention probability mass on clean support examples to approach 1.0
    (and mass on corrupted support examples to approach 0.0), without constraining the
    relative distribution among the clean support examples.
    When all supports are clean, this loss is 0.0.
    """
    if clean_mask.all():
        return torch.tensor(0.0, device=clean_mask.device)

    # clean_mask: (S,) -> (1, S, 1, 1, 1)
    mask_view = clean_mask.view(1, -1, 1, 1, 1)
    losses = []
    for w in attn_weights_dict.values():
        if w is None:
            continue
        # w is (B, S, 1, 1, 1) for per_image or (B, S, 1, H, W) for per_pixel
        clean_mass = (w * mask_view).sum(dim=1).clamp(min=1e-6, max=1.0)
        losses.append(-torch.log(clean_mass).mean())

    if not losses:
        return torch.tensor(0.0, device=clean_mask.device)
    return torch.stack(losses).mean()


def train_attention_model(
    attention_mode: str,
    datasets: dict,
    device: torch.device,
    num_steps: int = 800,
    support_size: int = 4,
    warm_start: bool = True,
    lr_attn: float = 2e-3,
    lr_backbone: float = 1e-5,
):
    random.seed(42)
    torch.manual_seed(42)

    task_sampler = MultiDomainTaskSampler(datasets)

    model = universeg(
        pretrained=False,
        attention_mode=attention_mode,
    ).to(device)

    checkpoint_dir = PROJECT_ROOT / "models" / "ours" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline_ckpt = checkpoint_dir / "universeg_5000_steps.pth"

    if warm_start and baseline_ckpt.exists():
        state_dict = torch.load(baseline_ckpt, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[{attention_mode}] Warm-started from {baseline_ckpt.name} (starts identical to baseline)")
    else:
        print(f"[{attention_mode}] Training from scratch")

    attn_params = [
        p for n, p in model.named_parameters() if "attn_" in n
    ]
    backbone_params = [
        p for n, p in model.named_parameters() if "attn_" not in n
    ]

    print(
        f"[{attention_mode}] Attention parameters: {sum(p.numel() for p in attn_params)} | "
        f"Backbone parameters: {sum(p.numel() for p in backbone_params)}"
    )

    optimizer = Adam(
        [
            {"params": attn_params, "lr": lr_attn},
            {"params": backbone_params, "lr": lr_backbone if warm_start else 1e-4},
        ]
    )

    model.train()

    for step in range(1, num_steps + 1):
        # Freeze backbone for the first 75% of warm-start steps so the attention
        # heads specialize in scoring support compatibility while preserving clean features,
        # then jointly fine-tune for the remaining 25%.
        if warm_start:
            freeze_backbone = step <= int(0.75 * num_steps)
            for p in backbone_params:
                p.requires_grad = not freeze_backbone

        task = sample_robust_task(
            task_sampler,
            datasets,
            support_size=support_size,
            corruption_prob=0.65,
        )

        support_images = task["support_images"].unsqueeze(0).to(device)
        support_labels = task["support_labels"].unsqueeze(0).to(device)
        query_image = task["query_image"].unsqueeze(0).to(device)
        query_label = task["query_label"].unsqueeze(0).to(device)
        clean_mask = task["clean_mask"].to(device)

        optimizer.zero_grad()
        prediction = model(query_image, support_images, support_labels)

        dice_l = soft_dice_loss(prediction, query_label)
        attn_l = compute_attention_clean_mass_loss(
            model.get_attention_weights(),
            clean_mask,
        )
        loss = dice_l + 0.35 * attn_l

        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            print(
                f"[{attention_mode}] Step {step:4d}/{num_steps} | "
                f"Dataset: {task['dataset']:9s} | "
                f"DiceLoss: {dice_l.item():.4f} | AttnLoss: {attn_l.item():.4f}",
                flush=True,
            )

    out_path = checkpoint_dir / f"universeg_attn_{attention_mode}_5000_steps.pth"
    torch.save(model.state_dict(), out_path)
    print(f"[{attention_mode}] Saved checkpoint to: {out_path}\n", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["per_image", "per_pixel", "both"],
        default="both",
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--support-size", type=int, default=4)
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = PROJECT_ROOT / "data" / "raw"
    datasets = {
        "OASIS": CachedDataset(OASISDataset(data_root / "OASIS")),
        "ACDC": CachedDataset(ACDCDataset(data_root / "ACDC")),
        "DRIVE": CachedDataset(DRIVEDataset(data_root / "drive")),
        "CAMUS": CachedDataset(CAMUSDataset(data_root / "CAMUS")),
        "PanDental": CachedDataset(PanDentalDataset(data_root / "PanDental")),
        "CHAOS": CachedDataset(CHAOSDataset(data_root / "CHAOS")),
    }

    for name, ds in datasets.items():
        print(f"{name}: {len(ds)}")

    modes = ["per_image", "per_pixel"] if args.mode == "both" else [args.mode]
    for mode in modes:
        train_attention_model(
            attention_mode=mode,
            datasets=datasets,
            device=device,
            num_steps=args.steps,
            support_size=args.support_size,
            warm_start=not args.from_scratch,
        )


if __name__ == "__main__":
    main()
