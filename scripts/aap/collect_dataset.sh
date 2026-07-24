#!/usr/bin/env bash
# Collect the stoppability dataset.
#
# No push-strength search needed: each push's magnitude is randomized per
# env within [PUSH_VX_MIN, PUSH_VX_MAX] (and vy similarly), so a single pass
# naturally spans gentle pushes (mostly survived) to hard pushes (mostly
# failed) instead of everyone getting hit with the same fixed magnitude.
# Every failure is kept; successes are subsampled (--success_keep_prob) so
# the saved dataset isn't drowned out by easy survivals.
#
# Usage:
#   bash scripts/aap/collect_dataset.sh <L1> <L2> [OUT_DIR]
#
# Optional env overrides:
#   NUM_ENVS=64
#   PUSH_VX_MIN=0.4   PUSH_VX_MAX=2.4
#   PUSH_VY_MIN=0.15  PUSH_VY_MAX=0.9
#   PUSH_DELAY=4
#   NUM_SAMPLES=15000          # size of the dataset written to disk
#   SUCCESS_KEEP_PROB=0.3      # fraction of successes kept
#
# Output: $OUT_DIR/stoppability_dataset.pt

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/collect_dataset.sh <L1> <L2> [OUT_DIR]}"
L2="${2:?Usage: bash scripts/aap/collect_dataset.sh <L1> <L2> [OUT_DIR]}"
OUT="${3:-logs/aap/paper_final}"

mkdir -p "$OUT"

for f in "$L1" "$L2"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Checkpoint not found: $f"
    exit 1
  fi
done

NUM_ENVS="${NUM_ENVS:-64}"
PUSH_VX_MIN="${PUSH_VX_MIN:-0.4}"
PUSH_VX_MAX="${PUSH_VX_MAX:-2.4}"
PUSH_VY_MIN="${PUSH_VY_MIN:-0.15}"
PUSH_VY_MAX="${PUSH_VY_MAX:-0.9}"
PUSH_DELAY="${PUSH_DELAY:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-15000}"
SUCCESS_KEEP_PROB="${SUCCESS_KEEP_PROB:-0.3}"

DATASET="$OUT/stoppability_dataset.pt"

echo "=================================================================="
echo "[COLLECT] push_vx in [$PUSH_VX_MIN, $PUSH_VX_MAX], push_vy in [$PUSH_VY_MIN, $PUSH_VY_MAX]"
echo "  keep ALL failures + ${SUCCESS_KEEP_PROB} of successes (n=$NUM_SAMPLES)"
echo "=================================================================="
python scripts/aap/collect_stoppability.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
  --num_envs "$NUM_ENVS" --num_samples "$NUM_SAMPLES" \
  --push_before_sample \
  --push_vx_min "$PUSH_VX_MIN" --push_vx_max "$PUSH_VX_MAX" \
  --push_vy_min "$PUSH_VY_MIN" --push_vy_max "$PUSH_VY_MAX" \
  --push_delay "$PUSH_DELAY" \
  --success_keep_prob "$SUCCESS_KEEP_PROB" \
  --sample_interval 25 --warmup_steps 30 \
  --output "$DATASET"

if [ ! -f "$DATASET" ]; then
  echo "[ERROR] Dataset was not created at $DATASET"
  exit 1
fi

echo
echo "[DONE] Dataset saved: $DATASET"
