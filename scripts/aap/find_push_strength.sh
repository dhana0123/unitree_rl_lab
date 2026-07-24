#!/usr/bin/env bash
# STEP A: Auto-find push strength only.
# Adaptive (bisection) search for a push_vx that makes L1 fail a healthy
# fraction of the time (default target: 15-45% raw failure rate) using small,
# fast probes. Does NOT collect the real dataset or train anything -- just
# finds the right difficulty and writes it to a file for the next step.
#
# Usage:
#   bash scripts/aap/find_push_strength.sh <L1_checkpoint> <L2_checkpoint> [OUT_DIR]
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
#
# Output: $OUT_DIR/chosen_push.env  (sourceable: PUSH_VX=... / PUSH_VY=...)
# Also prints the chosen values to stdout at the end.

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/find_push_strength.sh <L1> <L2> [OUT_DIR]}"
L2="${2:?Usage: bash scripts/aap/find_push_strength.sh <L1> <L2> [OUT_DIR]}"
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

CHOSEN_FILE="$OUT/chosen_push.env"
{
  echo "PUSH_VX=$FOUND_VX"
  echo "PUSH_VY=$FOUND_VY"
} > "$CHOSEN_FILE"

echo
echo "[SAVED] Chosen push written to $CHOSEN_FILE"
echo "Next step:"
echo "  bash scripts/aap/collect_and_train_vstop.sh \"$L1\" \"$L2\" $FOUND_VX $FOUND_VY \"$OUT\""
echo "  (or omit the push values to auto-read them from $CHOSEN_FILE)"
