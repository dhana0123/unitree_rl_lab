#!/usr/bin/env bash
# Quick calibration: try several push_vx/push_vy strengths with a SMALL sample
# count and report pos_rate for each, so you can pick a push magnitude that
# lands in a healthy 0.3-0.7 pos_rate band before running the full (slow)
# bootstrap collection in run_full_paper_pipeline.sh.
#
# Usage:
#   bash scripts/aap/calibrate_push.sh <L1_checkpoint> <L2_checkpoint> [VX_LIST]
#
# Example:
#   bash scripts/aap/calibrate_push.sh "$L1" "$L2" "0.6,0.8,1.0,1.2,1.4"
#
# Optional env overrides:
#   NUM_ENVS=64
#   CALIB_SAMPLES=1500       # small on purpose, just for a quick pos_rate read
#   PUSH_DELAY=4
#   VY_RATIO=0.4             # push_vy = push_vx * VY_RATIO

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/calibrate_push.sh <L1> <L2> [VX_LIST]}"
L2="${2:?Usage: bash scripts/aap/calibrate_push.sh <L1> <L2> [VX_LIST]}"
VX_CSV="${3:-0.6,0.8,1.0,1.2,1.4}"

NUM_ENVS="${NUM_ENVS:-64}"
CALIB_SAMPLES="${CALIB_SAMPLES:-1500}"
PUSH_DELAY="${PUSH_DELAY:-4}"
VY_RATIO="${VY_RATIO:-0.4}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "=================================================================="
echo "[INFO] Push calibration (target: pos_rate in ~0.3-0.7)"
echo "  Trying push_vx in: $VX_CSV"
echo "  push_vy = push_vx * $VY_RATIO"
echo "  samples per trial: $CALIB_SAMPLES (small, just for a quick read)"
echo "=================================================================="

IFS=',' read -r -a VX_LIST <<< "$VX_CSV"
RESULTS_FILE="$TMP_DIR/results.csv"
echo "push_vx,push_vy,pos_rate,unsafe_n,total_n" > "$RESULTS_FILE"

for VX in "${VX_LIST[@]}"; do
  VY=$(python -c "print(round(${VX} * ${VY_RATIO}, 3))")
  OUT_PT="$TMP_DIR/calib_vx${VX//./p}.pt"

  echo
  echo "------------------------------------------------------------------"
  echo "[TRIAL] push_vx=$VX  push_vy=$VY"
  echo "------------------------------------------------------------------"

  LOG_FILE="$TMP_DIR/log_vx${VX//./p}.txt"
  python scripts/aap/collect_stoppability.py --headless \
    --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
    --num_envs "$NUM_ENVS" --num_samples "$CALIB_SAMPLES" \
    --push_before_sample --push_vx "$VX" --push_vy "$VY" --push_delay "$PUSH_DELAY" \
    --sample_interval 25 --warmup_steps 30 \
    --output "$OUT_PT" 2>&1 | tee "$LOG_FILE"

  # Pull the final "Saved N samples ... pos_rate=X.XXX unsafe=Y" line.
  SUMMARY_LINE=$(grep -E "^\[INFO\] Saved" "$LOG_FILE" | tail -n 1 || true)
  POS_RATE=$(echo "$SUMMARY_LINE" | grep -oE "pos_rate=[0-9.]+" | cut -d= -f2 || echo "NA")
  UNSAFE_N=$(echo "$SUMMARY_LINE" | grep -oE "unsafe=[0-9]+" | cut -d= -f2 || echo "NA")

  echo "$VX,$VY,$POS_RATE,$UNSAFE_N,$CALIB_SAMPLES" >> "$RESULTS_FILE"
  echo "[RESULT] push_vx=$VX -> pos_rate=$POS_RATE unsafe=$UNSAFE_N/$CALIB_SAMPLES"
done

echo
echo "=================================================================="
echo "[DONE] Calibration summary"
echo "=================================================================="
column -s, -t "$RESULTS_FILE"
echo
echo "Pick the push_vx/push_vy row with pos_rate closest to the middle of"
echo "0.3-0.7 (a good mix of success and failure), then run the full pipeline:"
echo
echo "  PUSH_VX_COLLECT=<chosen_vx> PUSH_VY_COLLECT=<chosen_vy> \\"
echo "    bash scripts/aap/run_full_paper_pipeline.sh \"\$L1\" \"\$L2\" logs/aap/paper_final"
echo "=================================================================="
