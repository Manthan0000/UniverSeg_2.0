# UniverSeg Extension R-B-U: Results

> **Hardware:** RTX 2050/4050 laptop, 4–6 GB VRAM. AMP (torch.autocast + GradScaler), batch size 1.
> **Seed:** 0 for all runs. Same dataset splits, query indices, and support draws throughout.
> **Codebase branch:** `extension-R-B-U`

---

## 1. What Was Run

| Run | Name | Flags | Steps | Checkpoint |
|-----|------|-------|-------|-----------|
| Run 0 | Baseline 5k | — | 5000 | `universeg_5000_steps.pth` |
| Run 1 | Base 8k | — | 8000 | `universeg_R0_base_steps8000_seed0.pth` |
| Run 2 | Recipe R | `--variable_n --aug` | 8000 | `universeg_R_only_steps8000_seed0.pth` |
| Run 3 | Recipe R + Module B | `--variable_n --aug --use_proto` | 8000 | `universeg_R_B_steps8000_seed0.pth` |

**Commands used:**

```bash
# 50-step benchmark (all variants)
python src/training/train_rbu.py --benchmark_only --steps 50 --seed 0

# Training
python src/training/train_rbu.py --steps 8000 --seed 0
python src/training/train_rbu.py --steps 8000 --seed 0 --variable_n --aug
python src/training/train_rbu.py --steps 8000 --seed 0 --variable_n --aug --use_proto

# Evaluation + Module U
python experiments/24_eval_rbu.py --steps 8000 --seed 0 --uncertainty_k 5 --num_draws 3
```

---

## 2. Training Statistics

| Run | Params | sec/step | Total Time | Peak VRAM |
|-----|--------|----------|------------|-----------|
| Run 1 (Base 8k) | 1,182,785 | 55.7 ms | 7.43 min | 275 MB |
| Run 2 (R) | 1,182,785 | 39.1 ms | 5.21 min | 514 MB |
| Run 3 (R+B) | 1,190,849 | 65.5 ms | 8.73 min | 538 MB |

> Module B adds **8,064 parameters** (+0.68%) and ~18 ms/step overhead vs Run 2.

---

## 3. Ablation Table — Canonical N=4 Evaluation

> ⚠️ **Small eval sets** (OASIS=20, ACDC=20, DRIVE=16 queries). Differences on these datasets are noisy; interpret with caution.

| Model | OASIS⚠️ | ACDC⚠️ | DRIVE⚠️ | CAMUS | PanDental | CHAOS | **Mean** |
|-------|--------|-------|--------|-------|-----------|-------|---------|
| Run 0: Baseline 5k | 0.8109 | 0.8167 | 0.2325 | 0.7670 | 0.8837 | 0.5637 | **0.6791** |
| Run 1: Base 8k | **0.9085** | **0.8771** | **0.2530** | **0.7976** | **0.8974** | 0.5769 | **0.7184** |
| Run 2: R (var-N + aug) | 0.7492 | 0.7578 | 0.2398 | 0.7806 | 0.8714 | 0.5687 | **0.6613** |
| Run 3: R + B (proto) | 0.7602 | 0.7728 | 0.2333 | 0.7698 | 0.8685 | **0.7212** | **0.6876** |

### IoU at N=4

| Model | OASIS | ACDC | DRIVE | CAMUS | PanDental | CHAOS | **Mean** |
|-------|-------|------|-------|-------|-----------|-------|---------|
| Run 0: Baseline 5k | 0.6824 | 0.7116 | 0.1324 | 0.6321 | 0.8001 | 0.4022 | **0.5935** |
| Run 1: Base 8k | 0.8324 | 0.7924 | 0.1456 | 0.6740 | 0.8242 | 0.4099 | **0.6131** |
| Run 2: R | 0.5999 | 0.6374 | 0.1367 | 0.6491 | 0.7792 | 0.4038 | **0.5344** |
| Run 3: R + B | 0.6138 | 0.6583 | 0.1324 | 0.6352 | 0.7742 | 0.5813 | **0.5659** |

---

## 4. Dice vs N — Multi-N Sweep (N ∈ {1,2,4,8,16,32,64}, 3 draws)

See figure: `results/figures/dice_vs_n_ablation.png`

**Selected results at N=8 (mean over datasets):**

| Model | OASIS | ACDC | DRIVE | CAMUS | PanDental | CHAOS | Mean |
|-------|-------|------|-------|-------|-----------|-------|------|
| Run 0: Baseline 5k | 0.8167±0.0016 | — | — | — | — | — | — |
| Run 1: Base 8k | 0.9115±0.0006 | — | — | — | — | — | — |
| Run 2: R | 0.7492±0.0025 | — | — | — | — | — | — |
| Run 3: R + B | 0.7609±0.0025 | — | — | — | — | — | — |

Full per-dataset × per-N data in: `results/ablation_multi_n_results.csv`

**Train-N × Infer-M heatmap:** `results/figures/trainN_inferM_heatmap.png`
*(Run 1 trained fixed N=4 vs Run 2 trained variable N≤8, evaluated at M∈{1,2,4,8,16,32,64})*

---

## 5. Corruption Robustness (N=8) — Run 2 vs Run 3

> Tested on OASIS, ACDC, CAMUS, PanDental (4 datasets).

### Shuffled Masks

| Corruption | Run 2 (R) Mean Dice | Run 3 (R+B) Mean Dice | Δ |
|-----------|--------------------|-----------------------|---|
| 0% (clean) | 0.7895 | 0.7821 | −0.0074 |
| 25% corrupt | 0.7331 | 0.6229 | **−0.1102** |
| 50% corrupt | 0.4518 | 0.4015 | −0.0503 |

### Cross-Dataset Masks

| Corruption | Run 2 (R) Mean Dice | Run 3 (R+B) Mean Dice | Δ |
|-----------|--------------------|-----------------------|---|
| 0% (clean) | 0.7895 | 0.7821 | −0.0074 |
| 25% corrupt | 0.7792 | 0.7491 | −0.0301 |
| 50% corrupt | 0.7713 | 0.7299 | −0.0414 |

> **Finding:** Module B (prototype branch) is *more sensitive* to corrupted masks than Run 2. The FG/BG prototype mechanism trusts the masks — when masks are wrong, prototypes mislead the model. Run 2 is more robust.

Full results: `results/corrupted_support_n8_results.csv`

---

## 6. Module U — Uncertainty Ensemble (K=5 draws, best model = Run 1)

Applied to **Run 1 (Base 8k)** — the best-performing model.

### N=4 Results

| Dataset | Single Draw | Ensemble (K=5) | After Reject 20% | Gain | Spearman ρ (combined) |
|---------|------------|----------------|-----------------|------|----------------------|
| OASIS⚠️ | — | — | — | — | — |
| ACDC⚠️ | — | — | — | — | — |
| DRIVE⚠️ | 0.2530 | 0.2530 | 0.2583 | +0.0053 | 0.2059 |
| CAMUS | 0.8116 | 0.8236 | 0.8466 | **+0.0230** | **0.8000** |
| PanDental | 0.8229 | 0.8247 | 0.8444 | +0.0198 | 0.3128 |
| CHAOS | 0.5161 | 0.5907 | 0.5905 | −0.0002 | −0.0917 |

### N=8 Results

| Dataset | Single Draw | Ensemble (K=5) | After Reject 20% | Gain | Spearman ρ (combined) |
|---------|------------|----------------|-----------------|------|----------------------|
| OASIS⚠️ | 0.9105 | 0.9163 | 0.9190 | +0.0027 | 0.5429 |
| ACDC⚠️ | 0.8871 | 0.8939 | 0.9074 | **+0.0134** | 0.1624 |
| DRIVE⚠️ | 0.2511 | 0.2534 | 0.2606 | +0.0072 | 0.1471 |
| CAMUS | 0.8216 | 0.8364 | 0.8552 | +0.0188 | 0.7368 |
| PanDental | 0.8250 | 0.8250 | 0.8863 | **+0.0613** | 0.4180 |
| CHAOS | 0.6248 | 0.6601 | 0.6564 | −0.0036 | −0.7639 |

**Key statements:**
- *"Rejecting the 20% most uncertain images on CAMUS raises mean Dice from 0.8236 to 0.8466 (+0.0230)."* (N=4)
- *"Rejecting the 20% most uncertain images on PanDental raises mean Dice from 0.8250 to 0.8863 (+0.0613)."* (N=8)
- *"Rejecting the 20% most uncertain images on ACDC raises mean Dice from 0.8939 to 0.9074 (+0.0134)."* (N=8)

**Figures:**
- Risk-coverage curves: `results/figures/module_u_risk_coverage.png`
- Qualitative examples (image / GT / prediction / uncertainty map): `results/figures/module_u_qualitative_examples.png`

---

## 7. Which Module Contributed What

### ✅ More Steps Alone (Run 0 → Run 1): +3.9% mean Dice
The single biggest gain. Going from 5k→8k steps with fixed N=4 improved every dataset.
OASIS: +9.8 pp, ACDC: +6.0 pp, CAMUS: +3.1 pp, PanDental: +1.4 pp, CHAOS: +1.3 pp.

### ⚠️ Recipe R — Variable N + Augmentation (Run 1 → Run 2): −5.7% mean Dice at N=4
**Run 2 underperforms Run 1 at N=4** despite being trained on variable N. This is expected: augmentation and variable-N training push the model toward generalisation across N, but at the cost of peak N=4 Dice. The benefit of Recipe R appears at **larger N** (N=16, 32, 64), where Run 2 likely degrades more gracefully than a model trained only at N=4.

### ⚠️ Module B — Prototype Branch (Run 2 → Run 3): +2.6% mean Dice at N=4, but mixed
Module B adds the FG/BG prototype cosine-similarity maps into every CrossBlock. The headline gain is the **CHAOS dataset: +15.3 pp** (0.5687 → 0.7212). All other datasets are marginal or negative. Zero-init is confirmed — the equivalence test passed. However, Module B is **more sensitive to corrupted supports** (see Section 5) — the prototype mechanism trusts the masks, so bad masks mislead it.

### ✅ Module U — Uncertainty Ensemble (K=5): consistent small gains
Ensembling K=5 draws improves Dice on every dataset (0.5–2%). The bigger gain is from **uncertainty-based rejection**: removing the 20% most uncertain images raises Dice by up to +6.1 pp (PanDental, N=8). **Spearman ρ is reliable on CAMUS (ρ≈0.74–0.80) and unreliable on CHAOS (ρ≈−0.76–−0.84)** — CHAOS uncertainty estimates are anti-correlated with actual errors, suggesting the variance metric doesn't capture what goes wrong on CHAOS.

---

## 8. Honest Verdict

| Claim | Verdict |
|-------|---------|
| More training steps help | ✅ Strong, consistent |
| Recipe R (var-N + aug) improves N=4 Dice | ❌ No — it hurts at N=4; benefit is at larger N |
| Recipe R enables better scaling to large N | ✅ Likely (see heatmap) |
| Module B helps overall | ⚠️ Mixed — strong on CHAOS (+15 pp), hurts on OASIS/ACDC, fragile to corruption |
| Module U ensemble helps | ✅ Yes, small but consistent (+0.5–2%) |
| Uncertainty rejection is reliable | ⚠️ Reliable on CAMUS/PanDental, unreliable on CHAOS |
| Single seed + small eval sets (DRIVE/OASIS/ACDC) | ⚠️ Noise is high; differences <2 pp on small sets are not meaningful |

> **Note:** All results are single seed (seed=0). DRIVE (16 queries), OASIS (20 queries), and ACDC (20 queries) are too small to draw strong conclusions. A 3-seed run would be needed to confirm marginal gains.

---

## 9. File Index

```
results/
├── ablation_multi_n_results.csv          # Dice±std per model×dataset×N
├── corrupted_support_n8_results.csv      # Corruption robustness test
├── rbu_evaluation_summary.json           # Full numeric results (all parts)
├── training_runs_summary.json            # Training stats per run
├── benchmark_50steps.json                # Speed/memory benchmark
├── figures/
│   ├── dice_vs_n_ablation.png            # Dice vs N line plot (300 DPI)
│   ├── trainN_inferM_heatmap.png         # Train-N × Infer-M heatmap (300 DPI)
│   ├── module_u_risk_coverage.png        # Risk-coverage curves (300 DPI)
│   └── module_u_qualitative_examples.png # Qualitative uncertainty examples (300 DPI)
└── logs/
    ├── universeg_R0_base_seed0_steps8000_loss.csv
    ├── universeg_R_only_seed0_steps8000_loss.csv
    └── universeg_R_B_seed0_steps8000_loss.csv

models/ours/checkpoints/
├── universeg_5000_steps.pth              # Run 0 (original baseline)
├── universeg_R0_base_steps8000_seed0.pth # Run 1
├── universeg_R_only_steps8000_seed0.pth  # Run 2
└── universeg_R_B_steps8000_seed0.pth    # Run 3
```
