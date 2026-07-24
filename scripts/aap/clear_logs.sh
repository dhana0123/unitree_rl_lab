#!/usr/bin/env bash
# Clear out logs/aap/ (eval results, trajectory logs, figures, etc.) between
# experiment sweeps.
#
# By default this PROTECTS the expensive-to-regenerate artifacts (the D_L2
# state bank, the stoppability dataset, and trained V_stop monitors) and
# only clears cheap/regenerable stuff (eval results, trajectory logs,
# figures, push-sweep tables, traces). Pass --all to wipe everything
# including the protected artifacts too.
#
# Usage:
#   bash scripts/aap/clear_logs.sh              # dry-run preview
#   bash scripts/aap/clear_logs.sh -y            # actually delete (protected)
#   bash scripts/aap/clear_logs.sh --all -y      # actually delete everything
#   bash scripts/aap/clear_logs.sh --path logs/aap/results_push_sweep -y
#
# Flags:
#   -y, --yes       Skip the confirmation prompt and actually delete.
#   --all           Also delete protected artifacts (l2_state_bank*.pt,
#                   stoppability_dataset*.pt, vstop*.pt).
#   --path DIR      Target a directory other than logs/aap (default: logs/aap).
#   -n, --dry-run   Explicitly force dry-run (default behavior anyway without -y).

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

TARGET_DIR="logs/aap"
DELETE_ALL=false
CONFIRMED=false
DRY_RUN=true

while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes) CONFIRMED=true; DRY_RUN=false; shift ;;
    --all) DELETE_ALL=true; shift ;;
    --path) TARGET_DIR="${2:?--path requires a directory}"; shift 2 ;;
    -n|--dry-run) DRY_RUN=true; CONFIRMED=false; shift ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ ! -d "$TARGET_DIR" ]; then
  echo "[INFO] $TARGET_DIR does not exist, nothing to clear."
  exit 0
fi

# Protected (expensive to regenerate: needs real Isaac Sim rollouts / training).
PROTECT_PATTERNS=(
  "l2_state_bank*.pt"
  "stoppability_dataset*.pt"
  "vstop*.pt"
)

is_protected() {
  local name
  name="$(basename "$1")"
  if [ "$DELETE_ALL" = true ]; then
    return 1
  fi
  for pat in "${PROTECT_PATTERNS[@]}"; do
    # shellcheck disable=SC2053
    if [[ "$name" == $pat ]]; then
      return 0
    fi
  done
  return 1
}

to_delete=()
to_keep=()
shopt -s nullglob dotglob
for entry in "$TARGET_DIR"/*; do
  if is_protected "$entry"; then
    to_keep+=("$entry")
  else
    to_delete+=("$entry")
  fi
done
shopt -u nullglob dotglob

echo "=================================================================="
echo "[INFO] Target directory: $TARGET_DIR"
if [ "$DELETE_ALL" = true ]; then
  echo "[INFO] Mode: --all (protected artifacts will ALSO be deleted)"
else
  echo "[INFO] Mode: protected (bank/dataset/vstop files kept; pass --all to wipe those too)"
fi
echo "=================================================================="

if [ ${#to_delete[@]} -eq 0 ]; then
  echo "[INFO] Nothing to delete."
else
  echo "Will delete:"
  for e in "${to_delete[@]}"; do
    size="$(du -sh "$e" 2>/dev/null | cut -f1)"
    echo "  - $e ($size)"
  done
fi

if [ ${#to_keep[@]} -gt 0 ]; then
  echo
  echo "Will KEEP (protected; use --all to include):"
  for e in "${to_keep[@]}"; do
    echo "  - $e"
  done
fi

echo

if [ ${#to_delete[@]} -eq 0 ]; then
  exit 0
fi

if [ "$DRY_RUN" = true ] && [ "$CONFIRMED" = false ]; then
  echo "[DRY RUN] Nothing deleted. Re-run with -y to actually delete the list above."
  exit 0
fi

if [ "$CONFIRMED" = false ]; then
  read -r -p "Delete the ${#to_delete[@]} item(s) listed above? [y/N] " reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "[INFO] Aborted."; exit 0 ;;
  esac
fi

for e in "${to_delete[@]}"; do
  rm -rf -- "$e"
  echo "[DELETED] $e"
done

echo "[DONE] Cleared $TARGET_DIR (${#to_delete[@]} item(s) removed, ${#to_keep[@]} kept)."
