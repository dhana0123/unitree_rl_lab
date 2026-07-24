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
    w_prev: torch.Tensor | None = None,
    ema_beta: float = 0.0,
) -> ControllerOutput:
    """AAP graduated abstention band.

    High V_stop  -> trust π_L2 (still stoppable)
    Low V_stop   -> use π_L1 (must fall back)
    Middle band  -> convex blend (smooth handoff)

    ``w_prev``/``ema_beta`` add optional temporal smoothing on the blend
    weight: ``w = ema_beta * w_prev + (1 - ema_beta) * w_raw``. Without this,
    a noisy V_stop estimate hovering near the band edges causes ``w`` (and
    hence the commanded action) to flicker step-to-step, which showed up
    empirically as elevated jerk and *higher* fall rate than a plain hard
    switch. ``ema_beta=0`` (default) reproduces the original memoryless
    behavior exactly.
    """
    if v_stop.ndim == 0:
        v_stop = v_stop.unsqueeze(0)
    if alpha_high <= alpha_low:
        raise ValueError("alpha_high must be > alpha_low")

    # w in [0, 1]: 1 = full L2, 0 = full L1
    w_raw = (v_stop - alpha_low) / (alpha_high - alpha_low)
    w_raw = torch.clamp(w_raw, 0.0, 1.0)

    if w_prev is not None and ema_beta > 0.0:
        w = ema_beta * w_prev + (1.0 - ema_beta) * w_raw
    else:
        w = w_raw

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
    w_prev: torch.Tensor | None = None,
    ema_beta: float = 0.0,
) -> ControllerOutput:
    """Dispatch by evaluation condition name.

    ``w_prev``/``ema_beta`` only affect the ``aap`` branch (temporal
    smoothing of the blend weight); ``always_l2``/``always_l1``/``hard_switch``
    remain exactly the original memoryless baselines for a fair comparison.
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
            w_prev=w_prev,
            ema_beta=ema_beta,
        )
    raise ValueError(f"Unknown controller mode: {mode}")
