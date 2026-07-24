#!/usr/bin/env bash
# Fully automatic stoppability dataset collection, single pass:
#   1) Adaptive (bisection) search for a push_vx that produces a healthy
#      fraction of REAL failures -- few small/fast probes, no manual eyeballing.
#   2) One full collection pass at that push level that keeps EVERY failure
#      and subsamples successes (--success_keep_prob), so the saved dataset
#      is dominated by failure/near-boundary states without needing a
#      separate bootstrap V_stop + refine pass.
#   3) Train the final V_stop monitor directly on that dataset.
#
# Usage:
#   bash scripts/aap/auto_collect_stoppability.sh <L1_checkpoint> <L2_checkpoint> [OUT_DIR]
#
# Optional env overrides:
#   NUM_ENVS=64
#   PROBE_SAMPLES=800          # small/fast trial size used only for the search
#   PROBE_ITERS=6              # max bisection iterations (usually converges in 3-5)
#   TARGET_FAIL_LOW=0.15       # target band for RAW (unfiltered) failure rate
#   TARGET_FAIL_HIGH=0.45
#   VX_MIN=0.3
#   VX_MAX=3.0
#   VY_RATIO=0.4               # push_vy = push_vx * VY_RATIO
#   PUSH_DELAY=4
#   FINAL_SAMPLES=15000        # size of the real dataset written to disk
#   SUCCESS_KEEP_PROB=0.3      # fraction of successes kept in the final pass
#
# Output: $OUT_DIR/stoppability_dataset.pt, $OUT_DIR/vstop_final.pt,
#         $OUT_DIR/table_c_monitor.csv

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/auto_collect_stoppability.sh <L1> <L2> [OUT_DIR]}"
L2="${2:?Usage: bash scripts/aap/auto_collect_stoppability.sh <L1> <L2> [OUT_DIR]}"
OUT="${3:-logs/aap/paper_final}"

NUM_ENVS="${NUM_ENVS:-64}"
PROBE_SAMPLES="${PROBE_SAMPLES:-800}"
PROBE_ITERS="${PROBE_ITERS:-6}"
TARGET_FAIL_LOW="${TARGET_FAIL_LOW:-0.15}"
TARGET_FAIL_HIGH="${TARGET_FAIL_HIGH:-0.45}"
VX_MIN="${VX_MIN:-0.3}"
VX_MAX="${VX_MAX:-3.0}"
VY_RATIO="${VY_RATIO:-0.4}"
PUSH_DELAY="${PUSH_DELAY:-4}"
FINAL_SAMPLES="${FINAL_SAMPLES:-15000}"
SUCCESS_KEEP_PROB="${SUCCESS_KEEP_PROB:-0.3}"

mkdir -p "$OUT"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

for f in "$L1" "$L2"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Checkpoint not found: $f"
    exit 1
  fi
done

get_pos_rate() {
  # $1 = log file -> prints pos_rate float (or "NA")
  local line
  line=$(grep -E "^\[INFO\] Saved" "$1" | tail -n 1 || true)
  echo "$line" | grep -oE "pos_rate=[0-9.]+" | cut -d= -f2 || echo "NA"
}

run_probe() {
  # $1 = push_vx -> writes log, prints raw_fail_rate float via stdout (last line)
  local vx="$1"
  local vy
  vy=$(python -c "print(round(${vx} * ${VY_RATIO}, 3))")
  local tag="probe_$(echo "$vx" | tr '.' 'p')"
  local log_file="$TMP_DIR/${tag}.log"
  local out_pt="$TMP_DIR/${tag}.pt"

  echo "[PROBE] push_vx=$vx push_vy=$vy (n=$PROBE_SAMPLES)" >&2
  python scripts/aap/collect_stoppability.py --headless \
    --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
    --num_envs "$NUM_ENVS" --num_samples "$PROBE_SAMPLES" \
    --push_before_sample --push_vx "$vx" --push_vy "$vy" --push_delay "$PUSH_DELAY" \
    --sample_interval 25 --warmup_steps 30 \
    --output "$out_pt" > "$log_file" 2>&1

  local pos_rate
  pos_rate=$(get_pos_rate "$log_file")
  if [ "$pos_rate" = "NA" ]; then
    echo "[WARN] Could not parse pos_rate from $log_file, treating as 0.5" >&2
    pos_rate="0.5"
  fi
  python -c "print(round(1.0 - ${pos_rate}, 4))"
}

echo "=================================================================="
echo "[SEARCH] Adaptive push_vx search for raw failure rate in [$TARGET_FAIL_LOW, $TARGET_FAIL_HIGH]"
echo "=================================================================="

LOW="$VX_MIN"
HIGH="$VX_MAX"
BEST_VX=""
BEST_DIST="999"
RESULTS_CSV="$TMP_DIR/search_results.csv"
echo "push_vx,raw_fail_rate" > "$RESULTS_CSV"

TARGET_MID=$(python -c "print((${TARGET_FAIL_LOW} + ${TARGET_FAIL_HIGH}) / 2.0)")
FOUND_VX=""

for i in $(seq 1 "$PROBE_ITERS"); do
  MID=$(python -c "print(round((${LOW} + ${HIGH}) / 2.0, 3))")
  RAW_FAIL=$(run_probe "$MID")
  echo "$MID,$RAW_FAIL" >> "$RESULTS_CSV"
  echo "  [$i/$PROBE_ITERS] push_vx=$MID -> raw_fail_rate=$RAW_FAIL"

  DIST=$(python -c "print(abs(${RAW_FAIL} - ${TARGET_MID}))")
  IS_BEST=$(python -c "print(1 if ${DIST} < ${BEST_DIST} else 0)")
  if [ "$IS_BEST" = "1" ]; then
    BEST_VX="$MID"
    BEST_DIST="$DIST"
  fi

  IN_BAND=$(python -c "print(1 if ${TARGET_FAIL_LOW} <= ${RAW_FAIL} <= ${TARGET_FAIL_HIGH} else 0)")
  if [ "$IN_BAND" = "1" ]; then
    echo "  [CONVERGED] push_vx=$MID lands in target band."
    FOUND_VX="$MID"
    break
  fi

  TOO_EASY=$(python -c "print(1 if ${RAW_FAIL} < ${TARGET_FAIL_LOW} else 0)")
  if [ "$TOO_EASY" = "1" ]; then
    LOW="$MID"   # push wasn't hard enough -> search upper half
  else
    HIGH="$MID"  # push was too hard -> search lower half
  fi
done

if [ -z "$FOUND_VX" ]; then
  FOUND_VX="$BEST_VX"
  echo "  [INFO] Did not converge exactly in $PROBE_ITERS iterations; using closest probe: push_vx=$FOUND_VX"
fi

FOUND_VY=$(python -c "print(round(${FOUND_VX} * ${VY_RATIO}, 3))")

echo
echo "=================================================================="
echo "[CHOSEN] push_vx=$FOUND_VX  push_vy=$FOUND_VY"
echo "=================================================================="
column -s, -t "$RESULTS_CSV" || cat "$RESULTS_CSV"

DATASET="$OUT/stoppability_dataset.pt"
VSTOP_FINAL="$OUT/vstop_final.pt"

echo
echo "=================================================================="
echo "[COLLECT] Final dataset: keep ALL failures + ${SUCCESS_KEEP_PROB} of successes (n=$FINAL_SAMPLES)"
echo "=================================================================="
python scripts/aap/collect_stoppability.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
  --num_envs "$NUM_ENVS" --num_samples "$FINAL_SAMPLES" \
  --push_before_sample --push_vx "$FOUND_VX" --push_vy "$FOUND_VY" --push_delay "$PUSH_DELAY" \
  --success_keep_prob "$SUCCESS_KEEP_PROB" \
  --sample_interval 25 --warmup_steps 30 \
  --output "$DATASET"

if [ ! -f "$DATASET" ]; then
  echo "[ERROR] Dataset was not created at $DATASET"
  exit 1
fi

echo
echo "=================================================================="
echo "[STATS] Dataset stats"
echo "=================================================================="
python scripts/aap/dataset_stats.py \
  --dataset "$DATASET" \
  --labels "auto (push_vx=$FOUND_VX, success_keep_prob=$SUCCESS_KEEP_PROB)" \
  --output "$OUT/table_dataset_stats.csv"

echo
echo "=================================================================="
echo "[TRAIN] Final V_stop monitor"
echo "=================================================================="
python scripts/aap/train_vstop.py \
  --dataset "$DATASET" \
  --output "$VSTOP_FINAL" \
  --table_output "$OUT/table_c_monitor.csv" \
  --split_name "final"

if [ ! -f "$VSTOP_FINAL" ]; then
  echo "[ERROR] V_stop monitor was not created at $VSTOP_FINAL"
  exit 1
fi

echo
echo "=================================================================="
echo "[DONE] Dataset + V_stop ready"
echo "  push_vx used:  $FOUND_VX (push_vy=$FOUND_VY)"
echo "  Dataset:       $DATASET"
echo "  V_stop:        $VSTOP_FINAL"
echo "  Dataset stats: $OUT/table_dataset_stats.csv"
echo "  Table C:       $OUT/table_c_monitor.csv"
echo "=================================================================="
