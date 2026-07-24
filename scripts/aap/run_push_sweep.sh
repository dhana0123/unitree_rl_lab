#!/usr/bin/env bash
# Harder push sweep: re-run eval_compare.py at several push_vx strengths and
# merge into one robustness table (paper Table D / disturbance sweep).
#
# Usage:
#   bash scripts/aap/run_push_sweep.sh <L1_checkpoint> <L2_checkpoint> <vstop.pt>
#
# Example:
#   bash scripts/aap/run_push_sweep.sh \
#     logs/rsl_rl/unitree_g1_29dof_safestop/<run>/model_5000.pt \
#     logs/rsl_rl/unitree_g1_29dof_velocity/2026-07-23_20-16-16/model_15000.pt \
#     logs/aap/vstop_v2.pt
#
# Optional env overrides:
#   PUSHES="0.8,1.2,1.6,2.0"   # comma-separated push_vx values
#   NUM_ENVS=64
#   EPISODES=100
#   PUSH_VY=0.0
#   OUT_ROOT=logs/aap/results_push_sweep

set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/run_push_sweep.sh <L1> <L2> <vstop>}"
L2="${2:?Usage: bash scripts/aap/run_push_sweep.sh <L1> <L2> <vstop>}"
VSTOP="${3:?Usage: bash scripts/aap/run_push_sweep.sh <L1> <L2> <vstop>}"

PUSHES_CSV="${PUSHES:-0.8,1.2,1.6,2.0}"
NUM_ENVS="${NUM_ENVS:-64}"
EPISODES="${EPISODES:-100}"
PUSH_VY="${PUSH_VY:-0.0}"
OUT_ROOT="${OUT_ROOT:-logs/aap/results_push_sweep}"
MERGED_CSV="$OUT_ROOT/table_d_push_sweep.csv"

mkdir -p "$OUT_ROOT"

# Reset merged table
echo "push_vx,condition,n,fall_rate,success_rate,mean_peak_jerk,std_peak_jerk" > "$MERGED_CSV"

IFS=',' read -r -a PUSH_LIST <<< "$PUSHES_CSV"

echo "=================================================================="
echo "[INFO] Push sweep"
echo "  L1:       $L1"
echo "  L2:       $L2"
echo "  V_stop:   $VSTOP"
echo "  Pushes:   ${PUSH_LIST[*]}"
echo "  num_envs: $NUM_ENVS"
echo "  episodes: $EPISODES"
echo "  output:   $OUT_ROOT"
echo "=================================================================="

i=0
for PUSH_VX in "${PUSH_LIST[@]}"; do
  i=$((i + 1))
  # sanitize folder name (dots -> p)
  TAG="vx$(echo "$PUSH_VX" | tr '.' 'p')"
  RUN_DIR="$OUT_ROOT/$TAG"

  echo
  echo "[$i/${#PUSH_LIST[@]}] Evaluating push_vx=$PUSH_VX -> $RUN_DIR"
  mkdir -p "$RUN_DIR"

  python scripts/aap/eval_compare.py --headless \
    --checkpoint_l2 "$L2" \
    --checkpoint_l1 "$L1" \
    --vstop "$VSTOP" \
    --num_envs "$NUM_ENVS" \
    --episodes "$EPISODES" \
    --push_step 80 \
    --push_vx "$PUSH_VX" \
    --push_vy "$PUSH_VY" \
    --alpha 0.7 --alpha_low 0.5 --alpha_high 0.7 \
    --conditions always_l2,hard_switch,aap \
    --output_dir "$RUN_DIR"

  SUMMARY="$RUN_DIR/table_a_summary.csv"
  if [ ! -f "$SUMMARY" ]; then
    echo "[ERROR] Missing $SUMMARY after eval_compare"
    exit 1
  fi

  # Append rows with push_vx column (skip header)
  tail -n +2 "$SUMMARY" | while IFS= read -r line; do
    [ -z "$line" ] && continue
    echo "${PUSH_VX},${line}" >> "$MERGED_CSV"
  done

  echo "[INFO] Appended results for push_vx=$PUSH_VX"
done

echo
echo "=================================================================="
echo "[DONE] Push sweep finished."
echo "  Merged table: $MERGED_CSV"
echo
echo "Preview:"
cat "$MERGED_CSV"
echo "=================================================================="
echo
echo "Optional plots per push strength:"
echo "  for d in $OUT_ROOT/vx*; do python scripts/aap/plot_results.py --results_dir \"\$d\"; done"
