#!/usr/bin/env bash
# Full paper pipeline, start to finish:
#   1) Auto-collect stoppability dataset (adaptive push search, single pass,
#      keeps every failure + subsampled successes) + train final V_stop
#   2) Table A (main condition comparison, hard push)
#   3) Table A plots
#   4) Table D (push-strength sweep) + plots per push level
#   5) Table B (abstention-band sweep, hard push)
#
# Usage:
#   bash scripts/aap/run_full_paper_pipeline.sh <L1_checkpoint> <L2_checkpoint> [OUT_DIR]
#
# Example:
#   bash scripts/aap/run_full_paper_pipeline.sh \
#     logs/rsl_rl/unitree_g1_29dof_safestop/2026-07-24_.../model_12300.pt \
#     logs/rsl_rl/unitree_g1_29dof_velocity/2026-07-23_.../model_15000.pt \
#     logs/aap/paper_final
#
# Optional env overrides (all have sane defaults):
#   NUM_ENVS=64
#   PROBE_SAMPLES=800  PROBE_ITERS=6  TARGET_FAIL_LOW=0.15  TARGET_FAIL_HIGH=0.45
#   VX_MIN=0.3  VX_MAX=3.0  VY_RATIO=0.4  PUSH_DELAY=4
#   FINAL_SAMPLES=15000            SUCCESS_KEEP_PROB=0.3
#   MAIN_PUSH_VX=2.0               MAIN_EPISODES=200
#   SWEEP_PUSHES="0.8,1.2,1.6,2.0,2.4,2.8"   SWEEP_EPISODES=150
#   BAND_PUSH_VX=2.0               BAND_EPISODES=200
#   BANDS="0.4:0.6,0.5:0.7,0.6:0.8,0.3:0.8"  HARD_ALPHA=0.7
#
# Dataset collection is fully automatic (scripts/aap/auto_collect_stoppability.sh
# does an adaptive bisection search for the right push strength, then a single
# collection pass) -- no manual calibration or two-pass refine needed.
#
# Fails fast on the first error (set -e) so you won't burn GPU time on a
# broken later stage without noticing.

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/run_full_paper_pipeline.sh <L1> <L2> [OUT_DIR]}"
L2="${2:?Usage: bash scripts/aap/run_full_paper_pipeline.sh <L1> <L2> [OUT_DIR]}"
OUT="${3:-logs/aap/paper_final}"

NUM_ENVS="${NUM_ENVS:-64}"

MAIN_PUSH_VX="${MAIN_PUSH_VX:-2.0}"
MAIN_EPISODES="${MAIN_EPISODES:-200}"

SWEEP_PUSHES="${SWEEP_PUSHES:-0.8,1.2,1.6,2.0,2.4,2.8}"
SWEEP_EPISODES="${SWEEP_EPISODES:-150}"

BAND_PUSH_VX="${BAND_PUSH_VX:-2.0}"
BAND_EPISODES="${BAND_EPISODES:-200}"
BANDS="${BANDS:-0.4:0.6,0.5:0.7,0.6:0.8,0.3:0.8}"
HARD_ALPHA="${HARD_ALPHA:-0.7}"

DATASET="$OUT/stoppability_dataset.pt"
VSTOP_FINAL="$OUT/vstop_final.pt"

mkdir -p "$OUT"

echo "=================================================================="
echo "[CHECK] Verifying checkpoints exist"
echo "=================================================================="
for f in "$L1" "$L2"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Checkpoint not found: $f"
    exit 1
  fi
done
echo "  L1: $L1"
echo "  L2: $L2"
echo "  OUT: $OUT"

echo
echo "=================================================================="
echo "[1a/5] Auto-find push strength"
echo "=================================================================="
bash scripts/aap/find_push_strength.sh "$L1" "$L2" "$OUT"

echo
echo "=================================================================="
echo "[1b/5] Collect dataset"
echo "=================================================================="
bash scripts/aap/collect_dataset.sh "$L1" "$L2" "" "" "$OUT"

echo
echo "=================================================================="
echo "[1c/5] Dataset stats table"
echo "=================================================================="
python scripts/aap/dataset_stats.py \
  --dataset "$DATASET" \
  --labels "final" \
  --output "$OUT/table_dataset_stats.csv"

echo
echo "=================================================================="
echo "[1d/5] Train V_stop + Table C"
echo "=================================================================="
python scripts/aap/train_vstop.py \
  --dataset "$DATASET" \
  --output "$VSTOP_FINAL" \
  --table_output "$OUT/table_c_monitor.csv" \
  --split_name "final"

if [ ! -f "$DATASET" ] || [ ! -f "$VSTOP_FINAL" ]; then
  echo "[ERROR] Expected $DATASET and $VSTOP_FINAL after dataset collection"
  exit 1
fi

echo
echo "=================================================================="
echo "[2/5] Table A: main condition comparison (push_vx=$MAIN_PUSH_VX, episodes=$MAIN_EPISODES)"
echo "=================================================================="
mkdir -p "$OUT/results_main"
python scripts/aap/eval_compare.py --headless \
  --checkpoint_l2 "$L2" --checkpoint_l1 "$L1" --vstop "$VSTOP_FINAL" \
  --num_envs "$NUM_ENVS" --episodes "$MAIN_EPISODES" \
  --push_step 80 --push_vx "$MAIN_PUSH_VX" --push_vy 0.0 \
  --alpha 0.7 --alpha_low 0.5 --alpha_high 0.7 \
  --conditions always_l2,always_l1,hard_switch,aap \
  --output_dir "$OUT/results_main"

if [ ! -f "$OUT/results_main/table_a_summary.csv" ]; then
  echo "[ERROR] Missing table_a_summary.csv after main eval"
  exit 1
fi

echo
echo "=================================================================="
echo "[3/5] Table A plots"
echo "=================================================================="
python scripts/aap/plot_results.py --results_dir "$OUT/results_main" --dpi 300

echo
echo "=================================================================="
echo "[4/5] Table D: push-strength sweep ($SWEEP_PUSHES, episodes=$SWEEP_EPISODES)"
echo "=================================================================="
PUSHES="$SWEEP_PUSHES" NUM_ENVS="$NUM_ENVS" EPISODES="$SWEEP_EPISODES" \
  OUT_ROOT="$OUT/results_push_sweep" \
  bash scripts/aap/run_push_sweep.sh "$L1" "$L2" "$VSTOP_FINAL"

for d in "$OUT"/results_push_sweep/vx*; do
  [ -d "$d" ] || continue
  python scripts/aap/plot_results.py --results_dir "$d" --dpi 300
done

echo
echo "=================================================================="
echo "[5/5] Table B: abstention-band sweep (push_vx=$BAND_PUSH_VX, episodes=$BAND_EPISODES)"
echo "=================================================================="
BANDS="$BANDS" HARD_ALPHA="$HARD_ALPHA" PUSH_VX="$BAND_PUSH_VX" \
  NUM_ENVS="$NUM_ENVS" EPISODES="$BAND_EPISODES" \
  OUT_ROOT="$OUT/results_band_sweep" \
  bash scripts/aap/run_band_sweep.sh "$L1" "$L2" "$VSTOP_FINAL"

echo
echo "=================================================================="
echo "[DONE] Full pipeline finished."
echo "=================================================================="
echo "Artifacts in $OUT:"
echo "  Dataset:          $DATASET"
echo "  V_stop final:     $VSTOP_FINAL"
echo "  Table (dataset):  $OUT/table_dataset_stats.csv"
echo "  Table C:          $OUT/table_c_monitor.csv"
echo "  Table A:          $OUT/results_main/table_a_summary.csv"
echo "  Table A plots:    $OUT/results_main/*.png"
echo "  Table D:          $OUT/results_push_sweep/table_d_push_sweep.csv"
echo "  Table D plots:    $OUT/results_push_sweep/vx*/*.png"
echo "  Table B:          $OUT/results_band_sweep/table_b_band_sweep.csv"
echo "=================================================================="
