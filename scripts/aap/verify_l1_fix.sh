#!/usr/bin/env bash
# Run the full pi_L1 verification chain after retraining with the fixed
# action_rate reward weight: standalone bank-drop fall rate -> per-step
# trajectory logging -> fall-aligned analysis -> (optional) full push sweep.
#
# Usage:
#   bash scripts/aap/verify_l1_fix.sh <L1_checkpoint> [<L2_checkpoint> <vstop.pt>]
#
# The L2 checkpoint and vstop args are optional -- if given, this also
# re-runs the full aap/hard_switch/always_l2 push sweep at the end. If
# omitted, only the L1-standalone steps run (no L2/monitor needed).
#
# Example (standalone-only, no push sweep):
#   bash scripts/aap/verify_l1_fix.sh \
#     logs/rsl_rl/unitree_g1_29dof_safestop/<new_run>/model_12600.pt
#
# Example (full chain including push sweep):
#   bash scripts/aap/verify_l1_fix.sh \
#     logs/rsl_rl/unitree_g1_29dof_safestop/<new_run>/model_12600.pt \
#     logs/rsl_rl/unitree_g1_29dof_velocity/<run>/model_15000.pt \
#     logs/aap/vstop_v2.pt
#
# Optional env overrides:
#   BANK_PATH=logs/aap/l2_state_bank.pt
#   NUM_ENVS=64
#   EPISODES=200
#   OUT_ROOT=logs/aap/verify_l1_fix

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/verify_l1_fix.sh <L1_checkpoint> [<L2_checkpoint> <vstop.pt>]}"
L2="${2:-}"
VSTOP="${3:-}"

BANK_PATH="${BANK_PATH:-logs/aap/l2_state_bank.pt}"
NUM_ENVS="${NUM_ENVS:-64}"
EPISODES="${EPISODES:-200}"
OUT_ROOT="${OUT_ROOT:-logs/aap/verify_l1_fix}"

mkdir -p "$OUT_ROOT"

echo "=================================================================="
echo "[1/3] Standalone bank-drop fall rate (no L2, no handoff, no monitor)"
echo "=================================================================="
python scripts/aap/eval_l1_standalone.py --headless \
  --checkpoint_l1 "$L1" \
  --bank_path "$BANK_PATH" \
  --num_envs "$NUM_ENVS" --episodes "$EPISODES" --max_steps 300 \
  --output_dir "$OUT_ROOT/standalone"

echo
echo "Baseline (pre-fix) fall_rate was 0.5350 -- compare against:"
cat "$OUT_ROOT/standalone/l1_standalone_summary.csv"

echo
echo "=================================================================="
echo "[2/3] Per-step trajectory logging + fall-aligned analysis"
echo "=================================================================="
python scripts/aap/log_l1_trajectories.py --headless \
  --checkpoint_l1 "$L1" \
  --bank_path "$BANK_PATH" \
  --num_trials 24 --max_log_steps 40 \
  --output_dir "$OUT_ROOT/trajectories"

python scripts/aap/analyze_l1_trajectories.py \
  --results_dir "$OUT_ROOT/trajectories"

if [ -n "$L2" ] && [ -n "$VSTOP" ]; then
  echo
  echo "=================================================================="
  echo "[3/3] Full push sweep (always_l2 / hard_switch / aap)"
  echo "=================================================================="
  PUSHES="0.8,1.2,1.6,2.0" NUM_ENVS="$NUM_ENVS" EPISODES="$EPISODES" \
    OUT_ROOT="$OUT_ROOT/push_sweep" \
    bash scripts/aap/run_push_sweep.sh "$L1" "$L2" "$VSTOP"
else
  echo
  echo "=================================================================="
  echo "[3/3] Skipped push sweep (no L2 checkpoint / vstop path given)."
  echo "  Re-run this script with them once L1's standalone fall rate looks good:"
  echo "    bash scripts/aap/verify_l1_fix.sh $L1 <L2_checkpoint> <vstop.pt>"
  echo "=================================================================="
fi

echo
echo "[DONE] Results under $OUT_ROOT"
