#!/usr/bin/env bash
# Table B: sweep the AAP abstention band (alpha_low, alpha_high) and merge
# into one CSV, plus a hard-switch (single alpha) reference row for
# comparison. eval_compare.py's own Table B output is just a template stub;
# this actually fills it in by re-running --conditions aap at each band.
#
# Usage:
#   bash scripts/aap/run_band_sweep.sh <L1_checkpoint> <L2_checkpoint> <vstop.pt>
#
# Optional env overrides:
#   BANDS="0.4:0.6,0.5:0.7,0.6:0.8,0.3:0.8"   # colon-separated alpha_low:alpha_high pairs
#   HARD_ALPHA=0.7                             # single-threshold reference row (hard_switch)
#   PUSH_VX=0.8
#   NUM_ENVS=64
#   EPISODES=100
#   OUT_ROOT=logs/aap/results_band_sweep

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/run_band_sweep.sh <L1> <L2> <vstop>}"
L2="${2:?Usage: bash scripts/aap/run_band_sweep.sh <L1> <L2> <vstop>}"
VSTOP="${3:?Usage: bash scripts/aap/run_band_sweep.sh <L1> <L2> <vstop>}"

BANDS_CSV="${BANDS:-0.4:0.6,0.5:0.7,0.6:0.8,0.3:0.8}"
HARD_ALPHA="${HARD_ALPHA:-0.7}"
PUSH_VX="${PUSH_VX:-0.8}"
NUM_ENVS="${NUM_ENVS:-64}"
EPISODES="${EPISODES:-100}"
OUT_ROOT="${OUT_ROOT:-logs/aap/results_band_sweep}"
MERGED_CSV="$OUT_ROOT/table_b_band_sweep.csv"

mkdir -p "$OUT_ROOT"
echo "alpha_low,alpha_high,condition,n,fall_rate,success_rate,mean_peak_jerk,std_peak_jerk" > "$MERGED_CSV"

echo "=================================================================="
echo "[INFO] Band sweep (Table B)  push_vx=$PUSH_VX  episodes=$EPISODES"
echo "  Bands: $BANDS_CSV"
echo "  hard_switch reference alpha: $HARD_ALPHA"
echo "=================================================================="

# Hard-switch reference row (single threshold, no graduated band).
echo
echo "[REF] hard_switch @ alpha=$HARD_ALPHA"
RUN_DIR="$OUT_ROOT/hard_alpha${HARD_ALPHA//./p}"
mkdir -p "$RUN_DIR"
python scripts/aap/eval_compare.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" --vstop "$VSTOP" \
  --num_envs "$NUM_ENVS" --episodes "$EPISODES" \
  --push_step 80 --push_vx "$PUSH_VX" \
  --alpha "$HARD_ALPHA" \
  --conditions hard_switch \
  --output_dir "$RUN_DIR"
tail -n +2 "$RUN_DIR/table_a_summary.csv" | while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "${HARD_ALPHA},${HARD_ALPHA},${line}" >> "$MERGED_CSV"
done

IFS=',' read -r -a BAND_LIST <<< "$BANDS_CSV"
i=0
for BAND in "${BAND_LIST[@]}"; do
  i=$((i + 1))
  ALPHA_LOW="${BAND%%:*}"
  ALPHA_HIGH="${BAND##*:}"
  TAG="band${ALPHA_LOW//./p}_${ALPHA_HIGH//./p}"
  RUN_DIR="$OUT_ROOT/$TAG"

  echo
  echo "[$i/${#BAND_LIST[@]}] aap @ alpha_low=$ALPHA_LOW alpha_high=$ALPHA_HIGH -> $RUN_DIR"
  mkdir -p "$RUN_DIR"

  python scripts/aap/eval_compare.py --headless \
    --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" --vstop "$VSTOP" \
    --num_envs "$NUM_ENVS" --episodes "$EPISODES" \
    --push_step 80 --push_vx "$PUSH_VX" \
    --alpha_low "$ALPHA_LOW" --alpha_high "$ALPHA_HIGH" \
    --conditions aap \
    --output_dir "$RUN_DIR"

  SUMMARY="$RUN_DIR/table_a_summary.csv"
  if [ ! -f "$SUMMARY" ]; then
    echo "[ERROR] Missing $SUMMARY after eval_compare"
    exit 1
  fi
  tail -n +2 "$SUMMARY" | while IFS= read -r line; do
    [ -z "$line" ] && continue
    echo "${ALPHA_LOW},${ALPHA_HIGH},${line}" >> "$MERGED_CSV"
  done
done

echo
echo "=================================================================="
echo "[DONE] Band sweep finished. Merged table: $MERGED_CSV"
echo
cat "$MERGED_CSV"
echo "=================================================================="
