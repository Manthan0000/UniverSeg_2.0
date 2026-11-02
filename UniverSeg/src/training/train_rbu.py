import sys
import csv
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch

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
    """Caches dataset (image, mask) pairs in RAM after first read to eliminate disk I/O bottlenecks."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_all_datasets(data_root: Path) -> dict:
    return {
        "OASIS": CachedDataset(OASISDataset(data_root / "OASIS")),
        "ACDC": CachedDataset(ACDCDataset(data_root / "ACDC")),
        "DRIVE": CachedDataset(DRIVEDataset(data_root / "drive")),
        "CAMUS": CachedDataset(CAMUSDataset(data_root / "CAMUS")),
        "PanDental": CachedDataset(PanDentalDataset(data_root / "PanDental")),
        "CHAOS": CachedDataset(CHAOSDataset(data_root / "CHAOS")),
    }


def run_50_step_benchmark(datasets: dict, device: torch.device, seed: int = 0) -> dict:
    """
    Runs 50 steps of each variant at N=8 (worst-case training support size)
    and reports time per step, peak GPU memory (MB), and parameter count.
    """
    variants = [
        ("Run1_Baseline", False, False, False),
        ("Run2_RecipeR", True, True, False),
        ("Run3_RecipeR_ModuleB", True, True, True),
    ]
    benchmark_results = {}

    print("=" * 80)
    print("50-STEP SMOKE TEST & HARDWARE BENCHMARK (Evaluated at Worst-Case N = 8)")
    print("=" * 80)

    use_amp = device.type == "cuda"

    for name, var_n, aug, use_proto in variants:
        set_all_seeds(seed)
        sampler = MultiDomainTaskSampler(datasets, seed=seed)
        model = universeg(pretrained=False, use_proto=use_proto).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        param_count = sum(p.numel() for p in model.parameters())

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for step in range(1, 51):
            # Force N=8 as required by prompt to measure peak memory and speed at N=8
            task = sampler.sample_task(support_size=8, variable_n=False, aug=aug)
            s_img = task["support_images"].unsqueeze(0).to(device)
            s_lbl = task["support_labels"].unsqueeze(0).to(device)
            q_img = task["query_image"].unsqueeze(0).to(device)
            q_lbl = task["query_label"].unsqueeze(0).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(q_img, s_img, s_lbl)
                loss = soft_dice_loss(pred.float(), q_lbl.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_mem_mb = 0.0

        elapsed = time.perf_counter() - t0
        sec_per_step = elapsed / 50.0
        est_8000_min = (sec_per_step * 8000.0) / 60.0

        benchmark_results[name] = {
            "params": param_count,
            "sec_per_step": round(sec_per_step, 4),
            "peak_vram_mb": round(peak_mem_mb, 2),
            "est_8000_steps_min": round(est_8000_min, 2),
        }

        print(
            f"{name:22s} | Params: {param_count:8d} | "
            f"Time/step: {sec_per_step*1000:6.1f} ms | "
            f"Peak VRAM (N=8): {peak_mem_mb:6.1f} MB | "
            f"Est. 8000 steps: {est_8000_min:5.1f} min",
            flush=True,
        )

    print("=" * 80)
    return benchmark_results


def train_single_run(
    run_name: str,
    datasets: dict,
    device: torch.device,
    steps: int = 8000,
    seed: int = 0,
    variable_n: bool = False,
    aug: bool = False,
    use_proto: bool = False,
    out_dir: Path = PROJECT_ROOT / "results",
) -> dict:
    set_all_seeds(seed)
    sampler = MultiDomainTaskSampler(datasets, seed=seed)

    model = universeg(pretrained=False, use_proto=use_proto).to(device)
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = PROJECT_ROOT / "models" / "ours" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    loss_csv_path = logs_dir / f"{run_name}_seed{seed}_steps{steps}_loss.csv"
    ckpt_path = ckpt_dir / f"{run_name}_steps{steps}_seed{seed}.pth"
    cfg_path = logs_dir / f"{run_name}_seed{seed}_steps{steps}_config.json"

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    print(f"\n>>> Starting {run_name} (steps={steps}, seed={seed}, variable_n={variable_n}, aug={aug}, use_proto={use_proto})")
    t0 = time.perf_counter()

    loss_rows = []
    running_loss = 0.0

    for step in range(1, steps + 1):
        task = sampler.sample_task(
            support_size=4,
            variable_n=variable_n,
            aug=aug,
        )
        s_img = task["support_images"].unsqueeze(0).to(device)
        s_lbl = task["support_labels"].unsqueeze(0).to(device)
        q_img = task["query_image"].unsqueeze(0).to(device)
        q_lbl = task["query_label"].unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(q_img, s_img, s_lbl)
            loss = soft_dice_loss(pred.float(), q_lbl.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = float(loss.item())
        running_loss += loss_val
        loss_rows.append(
            {
                "step": step,
                "dataset": task["dataset"],
                "support_size": task["support_size"],
                "loss": f"{loss_val:.6f}",
            }
        )

        if step % 200 == 0 or step == 1:
            avg_l = running_loss / (1 if step == 1 else 200)
            if step > 1:
                running_loss = 0.0
            elapsed_so_far = time.perf_counter() - t0
            print(
                f"[{run_name}] Step {step:4d}/{steps} | "
                f"Dataset: {task['dataset']:9s} (N={task['support_size']}) | "
                f"Loss: {avg_l:.4f} | Elapsed: {elapsed_so_far:5.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem_mb = 0.0

    total_time = time.perf_counter() - t0
    sec_per_step = total_time / steps

    torch.save(model.state_dict(), ckpt_path)

    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "dataset", "support_size", "loss"])
        writer.writeheader()
        writer.writerows(loss_rows)

    config_summary = {
        "run_name": run_name,
        "steps": steps,
        "seed": seed,
        "variable_n": variable_n,
        "aug": aug,
        "use_proto": use_proto,
        "param_count": param_count,
        "sec_per_step": round(sec_per_step, 5),
        "total_train_time_sec": round(total_time, 2),
        "peak_vram_mb": round(peak_mem_mb, 2),
        "checkpoint_path": str(ckpt_path),
        "loss_csv_path": str(loss_csv_path),
    }

    with open(cfg_path, "w") as f:
        json.dump(config_summary, f, indent=2)

    print(
        f">>> Finished {run_name} in {total_time/60:.2f} min "
        f"({sec_per_step*1000:.1f} ms/step, Peak VRAM: {peak_mem_mb:.1f} MB). "
        f"Saved: {ckpt_path.name}",
        flush=True,
    )
    return config_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable_n", action="store_true")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--use_proto", action="store_true")
    parser.add_argument("--uncertainty_k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--run_name", type=str, default="universeg_custom")
    parser.add_argument("--benchmark_only", action="store_true")
    parser.add_argument("--run_all_ablation", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = PROJECT_ROOT / "data" / "raw"
    datasets = load_all_datasets(data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark_only:
        bench = run_50_step_benchmark(datasets, device, seed=args.seed)
        with open(out_dir / "benchmark_50steps.json", "w") as f:
            json.dump(bench, f, indent=2)
        return

    if args.run_all_ablation:
        bench_path = out_dir / "benchmark_50steps.json"
        if not bench_path.exists():
            bench = run_50_step_benchmark(datasets, device, seed=args.seed)
            with open(bench_path, "w") as f:
                json.dump(bench, f, indent=2)

        runs_cfg = [
            ("universeg_R0_base", False, False, False),
            ("universeg_R_only", True, True, False),
            ("universeg_R_B", True, True, True),
        ]
        summaries = {}
        for r_name, var_n, aug, use_p in runs_cfg:
            summaries[r_name] = train_single_run(
                run_name=r_name,
                datasets=datasets,
                device=device,
                steps=args.steps,
                seed=args.seed,
                variable_n=var_n,
                aug=aug,
                use_proto=use_p,
                out_dir=out_dir,
            )
        with open(out_dir / "training_runs_summary.json", "w") as f:
            json.dump(summaries, f, indent=2)
        return

    train_single_run(
        run_name=args.run_name,
        datasets=datasets,
        device=device,
        steps=args.steps,
        seed=args.seed,
        variable_n=args.variable_n,
        aug=args.aug,
        use_proto=args.use_proto,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
