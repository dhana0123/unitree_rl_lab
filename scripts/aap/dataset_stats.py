"""Generate the AAP dataset statistics table for the paper (no Isaac needed).

Reads the .pt file(s) saved by collect_stoppability.py and prints / writes a
CSV table analogous to PRISM's "Progressive Stoppability Monitor Refinement"
table, but for a single-pass (or multi-pass appended) dataset.

Example:
  python scripts/aap/dataset_stats.py --dataset logs/aap/stoppability_dataset.pt \\
    --output logs/aap/results/table_dataset_stats.csv

Multiple datasets (e.g. before/after a --vstop refine pass) can be compared:
  python scripts/aap/dataset_stats.py \\
    --dataset logs/aap/stoppability_dataset_v1.pt logs/aap/stoppability_dataset.pt \\
    --labels "initial pass" "refined pass" \\
    --output logs/aap/results/table_dataset_stats.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Print/export AAP stoppability dataset stats table.")
    p.add_argument("--dataset", type=str, nargs="+", required=True, help="One or more .pt dataset files")
    p.add_argument("--labels", type=str, nargs="*", default=None, help="Row labels (default: filenames)")
    p.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    return p.parse_args()


def _to_policy_tensor(obs):
    """Extract the flat policy tensor if obs is a dict/TensorDict with a 'policy' key."""
    if torch.is_tensor(obs):
        return obs
    try:
        return obs["policy"]
    except (KeyError, TypeError, IndexError):
        return obs


def _row_for(path: str, label: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    obs = _to_policy_tensor(data["obs"])
    labels = data["labels"].float().view(-1)
    meta = data.get("metadata", {})

    n = len(labels)
    n_safe = int((labels >= 0.5).sum().item())
    n_unsafe = int((labels < 0.5).sum().item())
    unsafe_ratio = n_unsafe / max(1, n)

    return {
        "dataset": label,
        "total_samples": n,
        "obs_dim": int(obs.shape[-1]) if n else 0,
        "safe_n": n_safe,
        "unsafe_n": n_unsafe,
        "unsafe_ratio_pct": round(unsafe_ratio * 100, 2),
        "num_envs": meta.get("num_envs", ""),
        "fallback_horizon": meta.get("fallback_horizon", ""),
        "push_vx": meta.get("push_vx", ""),
        "push_vy": meta.get("push_vy", ""),
        "push_delay": meta.get("push_delay", ""),
        "sample_interval": meta.get("sample_interval", ""),
        "vstop_filter": meta.get("vstop_filter", ""),
        "checkpoint_l2": Path(str(meta.get("checkpoint_l2", ""))).name,
        "checkpoint_l1": Path(str(meta.get("checkpoint_l1", ""))).name,
        "collected_at": meta.get("collected_at", ""),
    }


def main():
    args = parse_args()
    labels = args.labels or [Path(p).stem for p in args.dataset]
    if len(labels) != len(args.dataset):
        raise ValueError("--labels count must match --dataset count")

    rows = [_row_for(p, lbl) for p, lbl in zip(args.dataset, labels)]

    fieldnames = list(rows[0].keys())
    col_widths = {f: max(len(f), *(len(str(r[f])) for r in rows)) for f in fieldnames}

    header = " | ".join(f.ljust(col_widths[f]) for f in fieldnames)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[f]).ljust(col_widths[f]) for f in fieldnames))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[INFO] Wrote {out}")


if __name__ == "__main__":
    main()
