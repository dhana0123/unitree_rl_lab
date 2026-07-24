"""Convert a stoppability dataset (.pt) to a flat CSV file (no Isaac needed).

Each row is one collected sample: the label column ("safe"/"unsafe" + a
binary 0/1 column) followed by one column per observation feature
(obs_0, obs_1, ...). Only samples with a valid safe/unsafe label are written
(rows with a NaN/missing label, if any, are dropped).

Example:
  python scripts/aap/dataset_to_csv.py \\
    --dataset logs/aap/paper_final/stoppability_dataset.pt \\
    --output logs/aap/paper_final/stoppability_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Convert AAP stoppability .pt dataset to .csv")
    p.add_argument("--dataset", type=str, required=True, help="Path to .pt dataset from collect_stoppability.py")
    p.add_argument("--output", type=str, required=True, help="Output .csv path")
    return p.parse_args()


def _to_policy_tensor(obs):
    """Extract the flat policy tensor if obs is a dict/TensorDict with a 'policy' key."""
    if torch.is_tensor(obs):
        return obs
    try:
        return obs["policy"]
    except (KeyError, TypeError, IndexError):
        return obs


def main():
    args = parse_args()

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    obs = _to_policy_tensor(data["obs"]).float()
    labels = data["labels"].float().view(-1)

    valid = torch.isfinite(labels)
    n_dropped = int((~valid).sum().item())
    obs = obs[valid]
    labels = labels[valid]

    n = obs.shape[0]
    obs_dim = obs.shape[-1] if n else 0
    n_safe = int((labels >= 0.5).sum().item())
    n_unsafe = int((labels < 0.5).sum().item())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["label", "label_binary"] + [f"obs_{i}" for i in range(obs_dim)]
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(n):
            is_safe = labels[i].item() >= 0.5
            row = ["safe" if is_safe else "unsafe", int(is_safe)] + obs[i].tolist()
            writer.writerow(row)

    print(f"[INFO] Wrote {n} rows ({n_safe} safe, {n_unsafe} unsafe) to {out}")
    if n_dropped:
        print(f"[INFO] Dropped {n_dropped} rows with invalid/missing labels")


if __name__ == "__main__":
    main()
