#!/usr/bin/env bash
# Convenience wrapper: runs the fully separate steps back to back.
#   1) scripts/aap/find_push_strength.sh   (auto-find push strength)
#   2) scripts/aap/collect_dataset.sh      (collect dataset)
#   3) scripts/aap/dataset_stats.py        (dataset stats table)
#   4) scripts/aap/train_vstop.py          (train V_stop + Table C)
#
# Usage:
#   bash scripts/aap/auto_collect_stoppability.sh <L1_checkpoint> <L2_checkpoint> [OUT_DIR]
#
# Each step above can also be run on its own -- see that script's header.
#
# Output: $OUT_DIR/stoppability_dataset.pt, $OUT_DIR/vstop_final.pt,
#         $OUT_DIR/table_dataset_stats.csv, $OUT_DIR/table_c_monitor.csv

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

L1="${1:?Usage: bash scripts/aap/auto_collect_stoppability.sh <L1> <L2> [OUT_DIR]}"
L2="${2:?Usage: bash scripts/aap/auto_collect_stoppability.sh <L1> <L2> [OUT_DIR]}"
OUT="${3:-logs/aap/paper_final}"
DATASET="$OUT/stoppability_dataset.pt"

bash scripts/aap/find_push_strength.sh "$L1" "$L2" "$OUT"

bash scripts/aap/collect_dataset.sh "$L1" "$L2" "" "" "$OUT"

python scripts/aap/dataset_stats.py \
  --dataset "$DATASET" \
  --labels "final" \
  --output "$OUT/table_dataset_stats.csv"

python scripts/aap/train_vstop.py \
  --dataset "$DATASET" \
  --output "$OUT/vstop_final.pt" \
  --table_output "$OUT/table_c_monitor.csv" \
  --split_name "final"
