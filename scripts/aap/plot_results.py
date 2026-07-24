"""Plot a gallery of paper figures from eval_compare.py outputs (no Isaac needed).

Generates bar / box / violin / KDE / histogram / scatter / trace-comparison
plots for the Always-L2 / Hard-switch / AAP conditions so you can review the
whole gallery and pick the figures that best support the paper.

Example:
  python scripts/aap/plot_results.py --results_dir logs/aap/results
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (no display needed on the server)

import matplotlib.pyplot as plt
import numpy as np
import torch

# Consistent color per condition across every figure in the gallery.
COND_COLORS = {
    "always_l2": "#d62728",  # red   - no safety layer (baseline)
    "hard_switch": "#ff7f0e",  # orange - PRISM-style hard switch
    "aap": "#2ca02c",  # green - proposed graduated abstention
    "always_l1": "#1f77b4",  # blue  - pure fallback baseline (if present)
}
_FALLBACK_CYCLE = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _color_for(cond: str, idx: int) -> str:
    return COND_COLORS.get(cond, _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="logs/aap/results")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def _read_trials(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "condition": r["condition"],
                    "fell": int(r["fell"]),
                    "success": int(r["success"]),
                    "peak_jerk": float(r["peak_jerk"]),
                    "final_height": float(r["final_height"]),
                    "mean_v_stop": float(r["mean_v_stop"]),
                }
            )
    return rows


def _kde(data: list[float], n_points: int = 200, bw: float | None = None):
    """Minimal Gaussian KDE (no scipy/seaborn dependency)."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return np.array([]), np.array([])
    std = arr.std()
    if bw is None:
        bw = 1.06 * std * n ** (-1 / 5) if std > 1e-9 else 0.05
        bw = max(bw, 1e-6)
    xs = np.linspace(arr.min() - 3 * bw, arr.max() + 3 * bw, n_points)
    diffs = (xs[:, None] - arr[None, :]) / bw
    dens = np.exp(-0.5 * diffs**2).sum(axis=1) / (n * bw * math.sqrt(2 * math.pi))
    return xs, dens


def _save(fig, out_dir: Path, name: str, dpi: int):
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


def plot_success_fall_bar(by_cond, conditions, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(6, 4))
    success = [sum(r["success"] for r in by_cond[c]) / max(1, len(by_cond[c])) * 100 for c in conditions]
    fall = [sum(r["fell"] for r in by_cond[c]) / max(1, len(by_cond[c])) * 100 for c in conditions]
    x = range(len(conditions))
    ax.bar([i - 0.2 for i in x], success, width=0.4, label="Success %", color="#2ca02c", alpha=0.85)
    ax.bar([i + 0.2 for i in x], fall, width=0.4, label="Fall %", color="#d62728", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=15)
    ax.set_ylabel("Percent")
    ax.legend()
    ax.set_title("Table A — Success / Fall by condition")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig_success_fall_bar.png", dpi)


def plot_jerk_box(by_cond, conditions, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [[r["peak_jerk"] for r in by_cond[c]] for c in conditions]
    bp = ax.boxplot(data, tick_labels=conditions, patch_artist=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_color_for(conditions[i], i))
        box.set_alpha(0.6)
    ax.set_ylabel("Peak action jerk at handoff")
    ax.set_title("Handoff discontinuity — box plot")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig_jerk_boxplot.png", dpi)


def plot_jerk_violin(by_cond, conditions, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [[r["peak_jerk"] for r in by_cond[c]] for c in conditions]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(_color_for(conditions[i], i))
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions, rotation=15)
    ax.set_ylabel("Peak action jerk at handoff")
    ax.set_title("Handoff discontinuity — violin plot")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig_jerk_violin.png", dpi)


def plot_jerk_kde(by_cond, conditions, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(conditions):
        xs, dens = _kde([r["peak_jerk"] for r in by_cond[c]])
        if len(xs) == 0:
            continue
        color = _color_for(c, i)
        ax.plot(xs, dens, color=color, label=c)
        ax.fill_between(xs, dens, alpha=0.25, color=color)
    ax.set_xlabel("Peak action jerk at handoff")
    ax.set_ylabel("Density")
    ax.set_title("Handoff discontinuity — KDE")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig_jerk_kde.png", dpi)


def plot_jerk_hist(by_cond, conditions, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(conditions):
        vals = [r["peak_jerk"] for r in by_cond[c]]
        if not vals:
            continue
        ax.hist(vals, bins=20, alpha=0.45, label=c, color=_color_for(c, i), density=True)
    ax.set_xlabel("Peak action jerk at handoff")
    ax.set_ylabel("Density")
    ax.set_title("Handoff discontinuity — histogram")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig_jerk_hist.png", dpi)


def plot_height_kde(by_cond, conditions, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(conditions):
        xs, dens = _kde([r["final_height"] for r in by_cond[c]])
        if len(xs) == 0:
            continue
        color = _color_for(c, i)
        ax.plot(xs, dens, color=color, label=c)
        ax.fill_between(xs, dens, alpha=0.25, color=color)
    ax.set_xlabel("Final base height (m)")
    ax.set_ylabel("Density")
    ax.set_title("Recovered height distribution — KDE")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig_height_kde.png", dpi)


def plot_height_box(by_cond, conditions, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [[r["final_height"] for r in by_cond[c]] for c in conditions]
    bp = ax.boxplot(data, tick_labels=conditions, patch_artist=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_color_for(conditions[i], i))
        box.set_alpha(0.6)
    ax.set_ylabel("Final base height (m)")
    ax.set_title("Recovered height distribution — box plot")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig_height_box.png", dpi)


def plot_scatter_height_vs_jerk(rows, conditions, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, c in enumerate(conditions):
        subset = [r for r in rows if r["condition"] == c]
        color = _color_for(c, i)
        ok = [r for r in subset if r["success"]]
        bad = [r for r in subset if not r["success"]]
        if ok:
            ax.scatter([r["peak_jerk"] for r in ok], [r["final_height"] for r in ok],
                       color=color, marker="o", alpha=0.6, label=f"{c} (success)")
        if bad:
            ax.scatter([r["peak_jerk"] for r in bad], [r["final_height"] for r in bad],
                       color=color, marker="x", alpha=0.8, label=f"{c} (fail)")
    ax.set_xlabel("Peak action jerk at handoff")
    ax.set_ylabel("Final base height (m)")
    ax.set_title("Handoff smoothness vs. recovery outcome")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig_scatter_height_vs_jerk.png", dpi)


def plot_scatter_vstop_vs_jerk(rows, conditions, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, c in enumerate(conditions):
        subset = [r for r in rows if r["condition"] == c]
        color = _color_for(c, i)
        ok = [r for r in subset if r["success"]]
        bad = [r for r in subset if not r["success"]]
        if ok:
            ax.scatter([r["mean_v_stop"] for r in ok], [r["peak_jerk"] for r in ok],
                       color=color, marker="o", alpha=0.6, label=f"{c} (success)")
        if bad:
            ax.scatter([r["mean_v_stop"] for r in bad], [r["peak_jerk"] for r in bad],
                       color=color, marker="x", alpha=0.8, label=f"{c} (fail)")
    ax.set_xlabel(r"Mean $\hat{V}_{stop}$ over episode")
    ax.set_ylabel("Peak action jerk at handoff")
    ax.set_title(r"Monitor confidence vs. handoff jerk")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig_scatter_vstop_vs_jerk.png", dpi)


def plot_traces(results_dir, conditions, out_dir, dpi):
    traces = {}
    for cond in conditions:
        trace_path = results_dir / f"trace_{cond}.pt"
        if not trace_path.exists():
            continue
        traces[cond] = torch.load(trace_path, map_location="cpu", weights_only=False)

    # Individual per-condition traces.
    for i, (cond, tr) in enumerate(traces.items()):
        fig, ax = plt.subplots(figsize=(8, 3))
        color = _color_for(cond, i)
        ax.plot(tr["v_stop"].numpy(), label=r"$\hat{V}_{stop}$", color=color)
        if "w_l2" in tr:
            ax.plot(tr["w_l2"].numpy(), label=r"$w_{L2}$", color="black", linestyle=":")
        ax.axhline(0.7, color="r", linestyle="--", alpha=0.6, label=r"$\alpha_{high}$")
        ax.axhline(0.5, color="orange", linestyle="--", alpha=0.6, label=r"$\alpha_{low}$")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(f"Safety confidence trace — {cond}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        _save(fig, out_dir, f"fig_trace_{cond}.png", dpi)

    # Combined comparison: V_stop across all conditions on one axis.
    if len(traces) > 1:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for i, (cond, tr) in enumerate(traces.items()):
            ax.plot(tr["v_stop"].numpy(), label=cond, color=_color_for(cond, i))
        ax.axhline(0.7, color="r", linestyle="--", alpha=0.5, label=r"$\alpha_{high}$")
        ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label=r"$\alpha_{low}$")
        ax.set_xlabel("Step")
        ax.set_ylabel(r"$\hat{V}_{stop}$")
        ax.set_title(r"$\hat{V}_{stop}$ comparison across conditions")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        _save(fig, out_dir, "fig_trace_comparison_vstop.png", dpi)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    trials_path = results_dir / "table_a_trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing {trials_path}. Run eval_compare.py first.")

    rows = _read_trials(trials_path)
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    conditions = list(by_cond.keys())

    plot_success_fall_bar(by_cond, conditions, out_dir, args.dpi)
    plot_jerk_box(by_cond, conditions, out_dir, args.dpi)
    plot_jerk_violin(by_cond, conditions, out_dir, args.dpi)
    plot_jerk_kde(by_cond, conditions, out_dir, args.dpi)
    plot_jerk_hist(by_cond, conditions, out_dir, args.dpi)
    plot_height_kde(by_cond, conditions, out_dir, args.dpi)
    plot_height_box(by_cond, conditions, out_dir, args.dpi)
    plot_scatter_height_vs_jerk(rows, conditions, out_dir, args.dpi)
    plot_scatter_vstop_vs_jerk(rows, conditions, out_dir, args.dpi)
    plot_traces(results_dir, conditions, out_dir, args.dpi)

    print(f"\n[INFO] Full gallery written to {out_dir}")
    print("[INFO] Recommended picks for an IEEE paper (space-constrained):")
    print("  - fig_success_fall_bar.png      -> main headline result (Table A companion)")
    print("  - fig_jerk_violin.png            -> handoff smoothness (violin > box, shows full shape)")
    print("  - fig_jerk_kde.png               -> alt to violin if you prefer continuous curves")
    print("  - fig_trace_comparison_vstop.png -> qualitative example of graduated vs hard handoff")
    print("  - fig_scatter_vstop_vs_jerk.png  -> ties monitor confidence to controller behavior")


if __name__ == "__main__":
    main()
