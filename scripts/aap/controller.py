"""Graduated abstention controller (AAP) vs PRISM-style hard switch."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ControllerOutput:
    actions: torch.Tensor
    v_stop: torch.Tensor
    w_l2: torch.Tensor  # 1 = full L2 authority, 0 = full L1


def prism_hard_switch(
    actions_l2: torch.Tensor,
    actions_l1: torch.Tensor,
    v_stop: torch.Tensor,
    alpha: float = 0.7,
) -> ControllerOutput:
    """PRISM-style binary switch: keep L2 iff V_stop >= alpha."""
    if v_stop.ndim == 0:
        v_stop = v_stop.unsqueeze(0)
    keep_l2 = (v_stop >= alpha).to(dtype=actions_l2.dtype).view(-1, *([1] * (actions_l2.ndim - 1)))
    actions = keep_l2 * actions_l2 + (1.0 - keep_l2) * actions_l1
    w = keep_l2.view(-1)
    return ControllerOutput(actions=actions, v_stop=v_stop, w_l2=w)


def aap_graduated(
    actions_l2: torch.Tensor,
    actions_l1: torch.Tensor,
    v_stop: torch.Tensor,
    alpha_low: float = 0.5,
    alpha_high: float = 0.7,
) -> ControllerOutput:
    """AAP graduated abstention band (Eq. 5-6, memoryless).

    High V_stop  -> trust π_L2 (still stoppable)
    Low V_stop   -> use π_L1 (must fall back)
    Middle band  -> convex blend (smooth handoff)
    """
    if v_stop.ndim == 0:
        v_stop = v_stop.unsqueeze(0)
    if alpha_high <= alpha_low:
        raise ValueError("alpha_high must be > alpha_low")

    # w in [0, 1]: 1 = full L2, 0 = full L1
    w = (v_stop - alpha_low) / (alpha_high - alpha_low)
    w = torch.clamp(w, 0.0, 1.0)

    w_view = w.view(-1, *([1] * (actions_l2.ndim - 1)))
    actions = w_view * actions_l2 + (1.0 - w_view) * actions_l1
    return ControllerOutput(actions=actions, v_stop=v_stop, w_l2=w)


def select_controller(
    mode: str,
    actions_l2: torch.Tensor,
    actions_l1: torch.Tensor,
    v_stop: torch.Tensor,
    alpha: float = 0.7,
    alpha_low: float = 0.5,
    alpha_high: float = 0.7,
) -> ControllerOutput:
    """Dispatch by evaluation condition name. All branches are memoryless
    (depend only on the current step's V_stop), matching Eq. 5-7 exactly.
    """
    mode = mode.lower()
    if mode in {"always_l2", "l2"}:
        w = torch.ones_like(v_stop if v_stop.ndim > 0 else v_stop.unsqueeze(0))
        return ControllerOutput(actions=actions_l2, v_stop=v_stop, w_l2=w)
    if mode in {"always_l1", "l1"}:
        w = torch.zeros_like(v_stop if v_stop.ndim > 0 else v_stop.unsqueeze(0))
        return ControllerOutput(actions=actions_l1, v_stop=v_stop, w_l2=w)
    if mode in {"hard", "hard_switch", "prism"}:
        return prism_hard_switch(actions_l2, actions_l1, v_stop, alpha=alpha)
    if mode in {"aap", "aap_full", "graduated"}:
        return aap_graduated(
            actions_l2,
            actions_l1,
            v_stop,
            alpha_low=alpha_low,
            alpha_high=alpha_high,
        )
    raise ValueError(f"Unknown controller mode: {mode}")
