"""Shared stoppability monitor network for AAP."""

from __future__ import annotations

import torch
import torch.nn as nn


def to_policy_tensor(obs) -> torch.Tensor:
    """Extract the flat policy observation tensor.

    Newer Isaac Lab / rsl-rl (>=2.3) versions return observations as a
    dict-like / TensorDict object with a "policy" key (plus e.g. "critic"),
    rather than a bare tensor. Downstream code (dataset storage, V_stop
    inference) needs a plain (N, obs_dim) tensor, so unwrap it here.
    """
    if torch.is_tensor(obs):
        return obs
    try:
        return obs["policy"]
    except (KeyError, TypeError, IndexError):
        return obs


class StoppabilityMonitor(nn.Module):
    """Predicts V_stop(x) in [0, 1]: P(π_L1 can safely reach MRC from x)."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return stoppability scores shaped (N,)."""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs).squeeze(-1)


def load_vstop(path: str, obs_dim: int | None = None, device: str | torch.device = "cpu") -> StoppabilityMonitor:
    """Load a checkpoint saved by train_vstop.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if obs_dim is None:
        obs_dim = int(ckpt["obs_dim"])
    model = StoppabilityMonitor(obs_dim, hidden=tuple(ckpt.get("hidden", (256, 128))))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
