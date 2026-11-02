#!/usr/bin/env bash
set -euo pipefail

STEPS=${1:-8000}
SEED=${2:-0}

echo "=== 1. Running Unit Tests (Equivalence, Permutation Invariance, Variable S, Recipe R, Proto Grads) ==="
python tests/test_rbu.py

echo "=== 2. Running 50-Step Smoke Benchmark + Training Runs 1, 2, 3 (steps=${STEPS}, seed=${SEED}) ==="
python src/training/train_rbu.py --run_all_ablation --steps "${STEPS}" --seed "${SEED}"

echo "=== 3. Running Part 4 Evaluation + Part 5 Module U (Uncertainty) ==="
python experiments/24_eval_rbu.py --steps "${STEPS}" --seed "${SEED}" --uncertainty_k 5 --num_draws 3

echo "=== All runs, tables, and 300 DPI figures completed in results/ ==="
