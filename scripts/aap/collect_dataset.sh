#!/usr/bin/env bash
# Collect the stoppability dataset ONLY (no stats, no training).
# Keeps EVERY failure and subsamples successes (--success_keep_prob) so the
# saved dataset is rich in failure/near-boundary states.
#
# Usage:
#   bash scripts/aap/collect_dataset.sh <L1> <L2> [PUSH_VX] [PUSH_VY] [OUT_DIR]
#
# If PUSH_VX/PUSH_VY are omitted, they're read from $OUT_DIR/chosen_push.env
# (the file written by scripts/aap/find_push_strength.sh).
#
# Optional env overrides:
#   NUM_ENVS=64
#   PUSH_DELAY=4
#   FINAL_SAMPLES=15000        # size of the dataset written to disk
#   SUCCESS_KEEP_PROB=0.3      # fraction of successes kept
#
# Output: $OUT_DIR/stoppability_dataset.pt

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/collect_dataset.sh <L1> <L2> [PUSH_VX] [PUSH_VY] [OUT_DIR]}"
L2="${2:?Usage: bash scripts/aap/collect_dataset.sh <L1> <L2> [PUSH_VX] [PUSH_VY] [OUT_DIR]}"
PUSH_VX_ARG="${3:-}"
PUSH_VY_ARG="${4:-}"
OUT="${5:-logs/aap/paper_final}"

mkdir -p "$OUT"

for f in "$L1" "$L2"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Checkpoint not found: $f"
    exit 1
  fi
done

if [ -z "$PUSH_VX_ARG" ] || [ -z "$PUSH_VY_ARG" ]; then
  CHOSEN_FILE="$OUT/chosen_push.env"
  if [ ! -f "$CHOSEN_FILE" ]; then
    echo "[ERROR] No PUSH_VX/PUSH_VY given and $CHOSEN_FILE not found."
    echo "        Run scripts/aap/find_push_strength.sh first, or pass push values explicitly."
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$CHOSEN_FILE"
  echo "[INFO] Read push from $CHOSEN_FILE: PUSH_VX=$PUSH_VX PUSH_VY=$PUSH_VY"
else
  PUSH_VX="$PUSH_VX_ARG"
  PUSH_VY="$PUSH_VY_ARG"
fi

NUM_ENVS="${NUM_ENVS:-64}"
PUSH_DELAY="${PUSH_DELAY:-4}"
FINAL_SAMPLES="${FINAL_SAMPLES:-15000}"
SUCCESS_KEEP_PROB="${SUCCESS_KEEP_PROB:-0.3}"

DATASET="$OUT/stoppability_dataset.pt"

echo "=================================================================="
echo "[COLLECT] push_vx=$PUSH_VX push_vy=$PUSH_VY"
echo "  keep ALL failures + ${SUCCESS_KEEP_PROB} of successes (n=$FINAL_SAMPLES)"
echo "=================================================================="
python scripts/aap/collect_stoppability.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
  --num_envs "$NUM_ENVS" --num_samples "$FINAL_SAMPLES" \
  --push_before_sample --push_vx "$PUSH_VX" --push_vy "$PUSH_VY" --push_delay "$PUSH_DELAY" \
  --success_keep_prob "$SUCCESS_KEEP_PROB" \
  --sample_interval 25 --warmup_steps 30 \
  --output "$DATASET"

if [ ! -f "$DATASET" ]; then
  echo "[ERROR] Dataset was not created at $DATASET"
  exit 1
fi

echo
echo "[DONE] Dataset saved: $DATASET"
