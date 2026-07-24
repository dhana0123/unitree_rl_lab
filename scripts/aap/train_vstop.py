"""Train V_stop monitor from a collected stoppability dataset (no Isaac needed).

Example:
  python scripts/aap/train_vstop.py --dataset logs/aap/stoppability_dataset.pt --output logs/aap/vstop.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from vstop_model import StoppabilityMonitor, to_policy_tensor


def parse_args():
    p = argparse.ArgumentParser(description="Train AAP stoppability monitor V_stop.")
    p.add_argument("--dataset", type=str, required=True, help="Path to .pt dataset from collect_stoppability.py")
    p.add_argument("--output", type=str, default="logs/aap/vstop.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--table_output",
        type=str,
        default=None,
        help="Optional CSV path for Table C (monitor quality). "
        "Defaults to <output_dir>/results/table_c_monitor.csv",
    )
    p.add_argument("--split_name", type=str, default="validation", help="Row label for Table C")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    obs = to_policy_tensor(data["obs"])
    if not torch.is_tensor(obs):
        raise TypeError(
            f"Could not extract a plain policy tensor from dataset['obs'] (got {type(obs)}). "
            "This dataset was likely collected before the TensorDict-unwrap fix in "
            "collect_stoppability.py — re-run collection to regenerate it."
        )
    obs = obs.float()
    labels = data["labels"].float()
    if labels.ndim > 1:
        labels = labels.view(-1)

    if obs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"obs/labels count mismatch: obs has {obs.shape[0]} rows but labels has {labels.shape[0]}. "
            "This dataset is likely corrupted (e.g. obs saved as a TensorDict with batch_size=(N,) "
            "instead of a (N, obs_dim) tensor) — re-run collect_stoppability.py to regenerate it."
        )

    obs_dim = obs.shape[-1]
    print(f"[INFO] Loaded {len(obs)} samples, obs_dim={obs_dim}, pos_rate={labels.mean().item():.3f}")

    dataset = TensorDataset(obs, labels)
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    pos = labels.sum().clamp(min=1.0)
    neg = (len(labels) - pos).clamp(min=1.0)
    w_pos = (neg / pos).item()
    w_neg = 1.0

    model = StoppabilityMonitor(obs_dim).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCELoss(reduction="none")

    best_val = float("inf")
    best_state = None
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            pred = model(xb)
            loss_vec = bce(pred, yb)
            weights = torch.where(yb > 0.5, w_pos, w_neg)
            loss = (loss_vec * weights).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= max(1, n_train)

        model.eval()
        val_loss = 0.0
        correct = 0
        safe_correct = unsafe_correct = 0
        safe_total = unsafe_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(args.device), yb.to(args.device)
                pred = model(xb)
                loss_vec = bce(pred, yb)
                weights = torch.where(yb > 0.5, w_pos, w_neg)
                val_loss += (loss_vec * weights).mean().item() * len(xb)
                pred_bin = (pred >= 0.5).float()
                correct += (pred_bin == yb).sum().item()
                safe_mask = yb > 0.5
                unsafe_mask = ~safe_mask
                safe_total += int(safe_mask.sum().item())
                unsafe_total += int(unsafe_mask.sum().item())
                safe_correct += int(((pred_bin == yb) & safe_mask).sum().item())
                unsafe_correct += int(((pred_bin == yb) & unsafe_mask).sum().item())
        val_loss /= max(1, n_val)
        acc = correct / max(1, n_val)
        safe_acc = safe_correct / max(1, safe_total)
        unsafe_acc = unsafe_correct / max(1, unsafe_total)
        print(
            f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={acc:.3f}  safe_acc={safe_acc:.3f}  unsafe_acc={unsafe_acc:.3f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "acc": acc,
                "safe_acc": safe_acc,
                "unsafe_acc": unsafe_acc,
                "n_val": n_val,
                "n_safe": safe_total,
                "n_unsafe": unsafe_total,
            }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "obs_dim": obs_dim,
            "hidden": (256, 128),
            "val_loss": best_val,
        },
        out,
    )
    print(f"[INFO] Saved V_stop monitor to {out}")

    # Table C: monitor quality (best-epoch validation metrics).
    table_path = Path(args.table_output) if args.table_output else out.parent / "results" / "table_c_monitor.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "split": args.split_name,
        "dataset": Path(args.dataset).name,
        "safe_acc": round(best_metrics["safe_acc"], 4),
        "unsafe_acc": round(best_metrics["unsafe_acc"], 4),
        "overall_acc": round(best_metrics["acc"], 4),
        "val_loss": round(best_metrics["val_loss"], 4),
        "best_epoch": best_metrics["epoch"],
        "n_val": best_metrics["n_val"],
        "n_safe": best_metrics["n_safe"],
        "n_unsafe": best_metrics["n_unsafe"],
    }
    write_header = not table_path.exists()
    with table_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[INFO] Wrote/updated Table C at {table_path}")
    print(
        f"[INFO] Table C row -> safe_acc={row['safe_acc']}  unsafe_acc={row['unsafe_acc']}  "
        f"overall_acc={row['overall_acc']}"
    )


if __name__ == "__main__":
    main()
