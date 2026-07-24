#!/usr/bin/env bash
# Collect the weak/medium/strong/combined V_stop datasets for the retrained
# (v2, randomized-entry) pi_L1, then build the comparison dataset-stats table.
#
# Usage:
#   bash scripts/aap/run_v2_dataset_sweep.sh <path_to_new_L1_checkpoint> [path_to_L2_checkpoint]
#
# Example:
#   bash scripts/aap/run_v2_dataset_sweep.sh \
#     logs/rsl_rl/unitree_g1_29dof_safestop/2026-07-24_12-00-00/model_5000.pt
#
# Run from the repo root (same place you'd normally run collect_stoppability.py).

set -euo pipefail

L1="${1:?Usage: bash scripts/aap/run_v2_dataset_sweep.sh <L1_checkpoint> [L2_checkpoint]}"
L2="${2:-logs/rsl_rl/unitree_g1_29dof_velocity/2026-07-23_20-16-16/model_15000.pt}"

NUM_ENVS="${NUM_ENVS:-64}"
OUT_DIR="logs/aap"
RESULTS_DIR="logs/aap/results"
mkdir -p "$OUT_DIR" "$RESULTS_DIR"

echo "=================================================================="
echo "[INFO] L2 checkpoint: $L2"
echo "[INFO] L1 checkpoint: $L1"
echo "[INFO] num_envs:      $NUM_ENVS"
echo "=================================================================="

echo
echo "[1/4] Collecting weak-push dataset (2,000 samples)..."
python scripts/aap/collect_stoppability.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
  --num_envs "$NUM_ENVS" --num_samples 2000 --push_before_sample \
  --push_vx 0.3 --push_vy 0.15 \
  --output "$OUT_DIR/ds_push_weak_v2.pt"

echo
echo "[2/4] Collecting medium-push dataset (2,000 samples)..."
python scripts/aap/collect_stoppability.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
  --num_envs "$NUM_ENVS" --num_samples 2000 --push_before_sample \
  --push_vx 0.6 --push_vy 0.3 \
  --output "$OUT_DIR/ds_push_medium_v2.pt"

echo
echo "[3/4] Collecting strong-push dataset (2,000 samples)..."
python scripts/aap/collect_stoppability.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
  --num_envs "$NUM_ENVS" --num_samples 2000 --push_before_sample \
  --push_vx 0.9 --push_vy 0.4 \
  --output "$OUT_DIR/ds_push_strong_v2.pt"

echo
echo "[4/4] Collecting final combined dataset (20,000 samples)..."
python scripts/aap/collect_stoppability.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
  --num_envs "$NUM_ENVS" --num_samples 20000 --push_before_sample \
  --push_vx 0.9 --push_vy 0.4 \
  --output "$OUT_DIR/stoppability_dataset_v2.pt"

echo
echo "[INFO] Building dataset-stats comparison table..."
python scripts/aap/dataset_stats.py \
  --dataset "$OUT_DIR/ds_push_weak_v2.pt" "$OUT_DIR/ds_push_medium_v2.pt" "$OUT_DIR/ds_push_strong_v2.pt" "$OUT_DIR/stoppability_dataset_v2.pt" \
  --labels "weak push (v2)" "medium push (v2)" "strong push (v2)" "final (20k, v2)" \
  --output "$RESULTS_DIR/table_dataset_stats_v2.csv"

echo
echo "=================================================================="
echo "[DONE] All 4 datasets collected + stats table written to:"
echo "         $RESULTS_DIR/table_dataset_stats_v2.csv"
echo
echo "Datasets:"
echo "  $OUT_DIR/ds_push_weak_v2.pt"
echo "  $OUT_DIR/ds_push_medium_v2.pt"
echo "  $OUT_DIR/ds_push_strong_v2.pt"
echo "  $OUT_DIR/stoppability_dataset_v2.pt"
echo "=================================================================="
