"""Fall-aligned analysis of scripts/aap/log_l1_trajectories.py output (pure
CSV analysis, no Isaac Sim / GPU needed -- safe to run anywhere, including
off-machine).

The raw per-trial trajectories from log_l1_trajectories.py fall at different
absolute steps (e.g. 10, 12, 13, ..., 20), which smears out the common
precursor pattern when plotted against absolute step. This script instead
aligns every FELL trial to t=0 at its own fall step (t<0 = steps before
falling) and averages across trials, so you can directly see:

  - whether action_rate / joint_deviation effort ramps up *before* tilt
    becomes unrecoverable (proactive) or only *at/after* the fall is already
    detected (reactive/too-late) -- exactly what's ambiguous from the raw
    per-step plot.
  - the exact step at which the tilt penalty (flat_orientation_l2) starts to
    outweigh the movement penalties (action_rate + joint_deviation_*), using
    the real per-step reward values already logged, not a re-derived
    approximation.

Example:
  python scripts/aap/analyze_l1_trajectories.py --results_dir logs/aap/l1_trajectories
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


FILENAME_RE = re.compile(r"traj_env(\d+)_(fell|ok)_fallstep(-?\d+)\.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Fall-aligned analysis of pi_L1 standalone trajectory logs.")
    p.add_argument("--results_dir", type=str, default="logs/aap/l1_trajectories")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--align_window", type=int, default=15, help="Steps before fall to include (t in [-window, 0]).")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def _load_trial(path: Path) -> tuple[int, str, int, list[dict]]:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    env_id, tag, fall_step = int(m.group(1)), m.group(2), int(m.group(3))
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return env_id, tag, fall_step, rows


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(results_dir.glob("traj_env*_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No traj_env*.csv files found in {results_dir}")

    fell_trials = []
    ok_trials = []
    for p in paths:
        env_id, tag, fall_step, rows = _load_trial(p)
        if tag == "fell":
            fell_trials.append((env_id, fall_step, rows))
        else:
            ok_trials.append((env_id, fall_step, rows))

    print(f"[INFO] Loaded {len(fell_trials)} FELL trials and {len(ok_trials)} survived trials from {results_dir}")

    # Figure out numeric columns (everything except step/fell_so_far, which are ints we don't average).
    sample_row = fell_trials[0][2][0] if fell_trials else ok_trials[0][2][0]
    numeric_cols = [c for c in sample_row.keys() if c not in ("step", "fell_so_far")]

    # --- Fall-aligned aggregation for FELL trials ---
    # bucket[t][col] -> list of values across trials
    window = args.align_window
    buckets: dict[int, dict[str, list[float]]] = {t: defaultdict(list) for t in range(-window, 1)}
    for env_id, fall_step, rows in fell_trials:
        for row in rows:
            step = int(row["step"])
            t = step - fall_step
            if -window <= t <= 0:
                for col in numeric_cols:
                    try:
                        buckets[t][col].append(float(row[col]))
                    except (ValueError, KeyError):
                        pass

    agg_rows = []
    for t in range(-window, 1):
        n = len(buckets[t].get("tilt_rad", []))
        row = {"t_before_fall": t, "n_trials": n}
        for col in numeric_cols:
            vals = buckets[t].get(col, [])
            row[col] = float(np.mean(vals)) if vals else float("nan")
        agg_rows.append(row)

    # Write aggregated CSV.
    agg_csv = out_dir / "fall_aligned_means.csv"
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"[INFO] Wrote {agg_csv}")

    # --- Print a compact table of the key quantities for a quick read. ---
    key_reward_cols = [c for c in ["rew_alive", "rew_flat_orientation_l2", "rew_action_rate",
                                    "rew_joint_deviation_legs", "rew_joint_deviation_waists",
                                    "rew_joint_deviation_arms", "rew_base_height"] if c in numeric_cols]
    header = ["t", "n", "tilt", "action_rate_raw", "dev_legs_hip"] + key_reward_cols
    print()
    print(" | ".join(f"{h:>16}" for h in header))
    print("-" * (19 * len(header)))
    for row in agg_rows:
        vals = [f"{row['t_before_fall']:>16d}", f"{row['n_trials']:>16d}",
                f"{row.get('tilt_rad', float('nan')):>16.3f}",
                f"{row.get('action_rate_raw', float('nan')):>16.3f}",
                f"{row.get('dev_legs_hip', float('nan')):>16.3f}"]
        for c in key_reward_cols:
            vals.append(f"{row.get(c, float('nan')):>16.4f}")
        print(" | ".join(vals))

    # Identify the crossover step: first t where |rew_flat_orientation_l2| exceeds
    # the combined movement-penalty terms (action_rate + joint_deviation_legs + waists).
    crossover_t = None
    for row in agg_rows:
        tilt_pen = -row.get("rew_flat_orientation_l2", 0.0)  # penalties are negative; flip sign for comparison
        move_pen = -(row.get("rew_action_rate", 0.0) + row.get("rew_joint_deviation_legs", 0.0)
                     + row.get("rew_joint_deviation_waists", 0.0))
        if tilt_pen > move_pen and crossover_t is None and not np.isnan(tilt_pen):
            crossover_t = row["t_before_fall"]
            break
    if crossover_t is not None:
        print(f"\n[INFO] Tilt penalty (flat_orientation_l2) first exceeds movement penalties "
              f"(action_rate + joint_deviation_legs/waists) at t={crossover_t} steps before fall.")
    action_rate_ramp_t = None
    baseline = np.nanmean([agg_rows[i]["action_rate_raw"] for i in range(min(5, len(agg_rows)))])
    for row in agg_rows:
        if row["action_rate_raw"] > 3 * max(baseline, 1e-6):
            action_rate_ramp_t = row["t_before_fall"]
            break
    if action_rate_ramp_t is not None:
        print(f"[INFO] action_rate_raw first ramps to >3x its far-from-fall baseline at "
              f"t={action_rate_ramp_t} steps before fall.")
    if crossover_t is not None and action_rate_ramp_t is not None:
        if action_rate_ramp_t <= crossover_t:
            print("[CONCLUSION] The policy starts reacting AT/BEFORE the point where the reward math "
                  "itself would justify it -> the delay is not simply 'reward doesn't pay for it yet'; "
                  "the policy/value function likely hasn't learned an effective correction in time "
                  "(training/coverage issue) even though acting earlier would already be rewarded.")
        else:
            print("[CONCLUSION] The policy only starts reacting AFTER the reward math would already "
                  "justify correcting -> real reward-shaping delay: action_rate/joint_deviation weights "
                  "are making early, cheap correction look worse than waiting, until it's too late. "
                  "Try lowering action_rate (e.g. -0.2 -> -0.05/-0.1) and/or joint_deviation_legs/waists "
                  "and re-run the standalone bank eval.")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [r["t_before_fall"] for r in agg_rows]
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        ax = axes[0]
        ax.plot(ts, [r.get("tilt_rad", np.nan) for r in agg_rows], label="tilt (rad)", color="#d62728")
        ax.plot(ts, [r.get("action_rate_raw", np.nan) / 20.0 for r in agg_rows],
                label="action_rate_raw / 20", color="#1f77b4")
        ax.plot(ts, [r.get("dev_legs_hip", np.nan) for r in agg_rows],
                label="hip roll/yaw deviation", color="#ff7f0e")
        ax.axvline(0, color="black", linestyle=":", alpha=0.6)
        ax.set_ylabel("Value")
        ax.set_title(f"Fall-aligned mean kinematics (n={len(fell_trials)} fallen trials, t=0 is fall detection)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        ax2 = axes[1]
        for c, color in zip(key_reward_cols, ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]):
            ax2.plot(ts, [r.get(c, np.nan) for r in agg_rows], label=c.replace("rew_", ""), color=color)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.axvline(0, color="black", linestyle=":", alpha=0.6)
        ax2.set_xlabel("Steps before fall detection (t=0)")
        ax2.set_ylabel("Per-step weighted reward")
        ax2.set_title("Fall-aligned mean reward terms")
        ax2.legend(fontsize=8, loc="lower left")
        ax2.grid(True, linestyle="--", alpha=0.4)

        fig.tight_layout()
        fig_path = out_dir / "fig_fall_aligned.png"
        fig.savefig(fig_path, dpi=args.dpi)
        plt.close(fig)
        print(f"\n[INFO] Wrote {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Plotting failed ({exc}); CSV is still written.")


if __name__ == "__main__":
    main()
