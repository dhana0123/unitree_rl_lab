"""Plot paper figures from eval_compare.py outputs (no Isaac needed).

Example:
  python scripts/aap/plot_results.py --results_dir logs/aap/results
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="logs/aap/results")
    p.add_argument("--output_dir", type=str, default=None)
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
                }
            )
    return rows


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

    # Success / fall bars
    fig, ax = plt.subplots(figsize=(6, 4))
    success = [sum(r["success"] for r in by_cond[c]) / max(1, len(by_cond[c])) * 100 for c in conditions]
    fall = [sum(r["fell"] for r in by_cond[c]) / max(1, len(by_cond[c])) * 100 for c in conditions]
    x = range(len(conditions))
    ax.bar([i - 0.2 for i in x], success, width=0.4, label="Success %")
    ax.bar([i + 0.2 for i in x], fall, width=0.4, label="Fall %")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=15)
    ax.set_ylabel("Percent")
    ax.legend()
    ax.set_title("Table A — Success / Fall by condition")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_success_fall.png", dpi=200)
    plt.close(fig)

    # Jerk boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [[r["peak_jerk"] for r in by_cond[c]] for c in conditions]
    ax.boxplot(data, tick_labels=conditions)
    ax.set_ylabel("Peak action jerk at handoff")
    ax.set_title("Handoff discontinuity (Hard vs AAP)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_jerk_boxplot.png", dpi=200)
    plt.close(fig)

    # V_stop / w traces if present
    for cond in conditions:
        trace_path = results_dir / f"trace_{cond}.pt"
        if not trace_path.exists():
            continue
        tr = torch.load(trace_path, map_location="cpu", weights_only=False)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(tr["v_stop"].numpy(), label=r"$\hat{V}_{stop}$")
        if "w_l2" in tr:
            ax.plot(tr["w_l2"].numpy(), label=r"$w_{L2}$")
        ax.axhline(0.7, color="r", linestyle="--", alpha=0.6, label=r"$\alpha_{high}$")
        ax.axhline(0.5, color="orange", linestyle="--", alpha=0.6, label=r"$\alpha_{low}$")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(f"Safety confidence trace — {cond}")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_trace_{cond}.png", dpi=200)
        plt.close(fig)

    print(f"[INFO] Figures written to {out_dir}")


if __name__ == "__main__":
    main()
