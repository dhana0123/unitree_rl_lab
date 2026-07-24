#!/usr/bin/env bash
# Stage everything useful under logs/ for git, while keeping the repo/LFS
# quota sane:
#   - force-adds every non-checkpoint file under logs/ (CSVs, YAML configs,
#     figures, V_stop models, datasets, etc.) -- these are normally excluded
#     by .gitignore's `**/logs/*`.
#   - for each RL training run (logs/rsl_rl/<experiment>/<timestamp>/), force
#     -adds ONLY the latest (highest-iteration) model_<N>.pt checkpoint, not
#     every intermediate save-interval checkpoint. This is what keeps this
#     safe to run frequently without blowing up your Git LFS budget.
#
# Usage:
#   bash scripts/aap/git_add_logs.sh
#
# Run this any time you want to snapshot current progress into git staging.
# It only stages files (git add); you still need to `git commit` yourself.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "[INFO] Staging all non-checkpoint files under logs/ ..."
non_ckpt_count=0
while IFS= read -r -d '' f; do
    git add -f -- "$f"
    non_ckpt_count=$((non_ckpt_count + 1))
done < <(find logs -type f ! -name 'model_*.pt' -print0 2>/dev/null)
echo "[INFO] Staged $non_ckpt_count non-checkpoint file(s)."

echo
echo "[INFO] Finding latest checkpoint per training run under logs/rsl_rl/ ..."
ckpt_count=0
while IFS= read -r -d '' run_dir; do
    latest=""
    latest_iter=-1
    for f in "$run_dir"/model_*.pt; do
        [ -e "$f" ] || continue
        base="$(basename "$f")"
        iter="${base#model_}"
        iter="${iter%.pt}"
        if [[ "$iter" =~ ^[0-9]+$ ]] && [ "$iter" -gt "$latest_iter" ]; then
            latest_iter="$iter"
            latest="$f"
        fi
    done
    if [ -n "$latest" ]; then
        git add -f -- "$latest"
        echo "  + $latest  (iteration $latest_iter)"
        ckpt_count=$((ckpt_count + 1))
    fi
done < <(find logs/rsl_rl -mindepth 2 -maxdepth 2 -type d -print0 2>/dev/null)
echo "[INFO] Staged $ckpt_count latest checkpoint(s) (skipped older intermediate checkpoints)."

echo
echo "=================================================================="
echo "[DONE] Staged $((non_ckpt_count + ckpt_count)) file(s) total."
echo "Review before committing:"
echo "  git status --short"
echo "Then commit with:"
echo "  git commit -m \"Update AAP logs/checkpoints\""
echo "=================================================================="
