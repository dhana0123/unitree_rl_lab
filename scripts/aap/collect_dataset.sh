#!/usr/bin/env bash
# Collect the stoppability dataset.
#
# No push-strength search needed: each push's magnitude is randomized per
# env within [PUSH_VX_MIN, PUSH_VX_MAX] (and vy similarly), so a single pass
# naturally spans gentle pushes (mostly survived) to hard pushes (mostly
# failed) instead of everyone getting hit with the same fixed magnitude.
#
# Unsafe (failure) and safe (success) samples are enforced towards a target
# CUMULATIVE mix via a running quota (default: 90% unsafe / 10% safe of the
# final dataset) -- NOT an independent per-batch/per-sample coin flip. Since
# failures are naturally rare in the raw rollout stream, this means every
# unsafe sample is kept until its (large) quota fills, while safe samples
# stop being added once their (small) quota fills -- so the running
# composition converges to the target ratio even if most batches are
# individually dominated by successes.
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
#   SUCCESS_KEEP_PROB=0.1      # target share of the FINAL dataset that is safe
#   FAILURE_KEEP_PROB=0.9      # target share of the FINAL dataset that is unsafe
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
SUCCESS_KEEP_PROB="${SUCCESS_KEEP_PROB:-0.1}"
FAILURE_KEEP_PROB="${FAILURE_KEEP_PROB:-0.9}"

DATASET="$OUT/stoppability_dataset.pt"

UNSAFE_PCT=$(python -c "print(round(${FAILURE_KEEP_PROB} * 100, 1))")
SAFE_PCT=$(python -c "print(round(${SUCCESS_KEEP_PROB} * 100, 1))")

echo "=================================================================="
echo "[TARGET] Forcing dataset composition via running quota: ${UNSAFE_PCT}% unsafe / ${SAFE_PCT}% safe"
echo "  (cumulative over the whole run, not per batch -- override with"
echo "   FAILURE_KEEP_PROB=x.xx / SUCCESS_KEEP_PROB=x.xx env vars)"
echo "=================================================================="
echo "[COLLECT] push_vx in [$PUSH_VX_MIN, $PUSH_VX_MAX], push_vy in [$PUSH_VY_MIN, $PUSH_VY_MAX]"
echo "  n=$NUM_SAMPLES samples -> $DATASET"
echo "=================================================================="
python scripts/aap/collect_stoppability.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" \
  --num_envs "$NUM_ENVS" --num_samples "$NUM_SAMPLES" \
  --push_before_sample \
  --push_vx_min "$PUSH_VX_MIN" --push_vx_max "$PUSH_VX_MAX" \
  --push_vy_min "$PUSH_VY_MIN" --push_vy_max "$PUSH_VY_MAX" \
  --push_delay "$PUSH_DELAY" \
  --success_keep_prob "$SUCCESS_KEEP_PROB" \
  --failure_keep_prob "$FAILURE_KEEP_PROB" \
  --sample_interval 25 --warmup_steps 30 \
  --output "$DATASET"

if [ ! -f "$DATASET" ]; then
  echo "[ERROR] Dataset was not created at $DATASET"
  exit 1
fi

echo
echo "[DONE] Dataset saved: $DATASET"
