"""Reset events for AAP pi_L1 (SafeStop) training.

Implements the "randomized-entry" idea from the AAP formalization: instead of
resetting pi_L1's training episodes only from a generic uniform box around the
origin, mix in real states recorded from running pi_L2 under randomized pushes
(see scripts/aap/collect_l2_states.py). This exposes pi_L1 to the actual kinds
of mid-gait / post-disturbance states it will be handed control from at
deployment time, rather than only clean, dynamically-arbitrary reset states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import reset_joints_by_scale, reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Cache loaded banks by path so we don't re-read the .pt file on every reset call.
_BANK_CACHE: dict[str, dict[str, torch.Tensor] | None] = {}

_BANK_KEYS = ("root_pos_rel", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel")


def _load_bank(path: str, device: torch.device) -> dict[str, torch.Tensor] | None:
    cache_key = f"{path}::{device}"
    if cache_key in _BANK_CACHE:
        return _BANK_CACHE[cache_key]
    try:
        data = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(
            f"[WARN] AAP L2 state bank not found at '{path}'. "
            "Falling back to uniform-box reset only. "
            "Run scripts/aap/collect_l2_states.py to enable randomized-entry training."
        )
        _BANK_CACHE[cache_key] = None
        return None

    bank = {k: data[k].to(device) for k in _BANK_KEYS if k in data}
    missing = [k for k in _BANK_KEYS if k not in data]
    if missing:
        print(f"[WARN] AAP L2 state bank '{path}' missing keys {missing}; disabling bank reset.")
        _BANK_CACHE[cache_key] = None
        return None

    print(f"[INFO] Loaded AAP L2 state bank '{path}' with {bank['root_pos_rel'].shape[0]} states.")
    _BANK_CACHE[cache_key] = bank
    return bank


def _apply_bank_states(env: ManagerBasedEnv, env_ids: torch.Tensor, bank: dict[str, torch.Tensor], asset: Articulation):
    """Teleport `env_ids` to freshly-sampled physical states from `bank`."""
    device = asset.device
    n_bank = bank["root_pos_rel"].shape[0]
    pick = torch.randint(0, n_bank, (len(env_ids),), device=device)

    origins = env.scene.env_origins[env_ids]
    positions = bank["root_pos_rel"][pick] + origins
    orientations = bank["root_quat"][pick]
    pose = torch.cat([positions, orientations], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)

    velocities = torch.cat([bank["root_lin_vel"][pick], bank["root_ang_vel"][pick]], dim=-1)
    if hasattr(asset, "write_root_velocity_to_sim"):
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    else:
        asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)

    joint_pos = bank["joint_pos"][pick].clone()
    joint_vel = bank["joint_vel"][pick].clone()
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_from_l2_bank(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    bank_path: str,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    joint_position_range: tuple[float, float] = (0.9, 1.1),
    joint_velocity_range: tuple[float, float] = (-1.0, 1.0),
    bank_prob: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset root state + joints from a mix of a pi_L2-induced state bank and a
    uniform-box fallback. (Runs at ``mode="reset"``, i.e. episode start.)

    For each env id, with probability ``bank_prob`` sample a full physical state
    (root pose+velocity, joint pos+velocity) recorded from real pi_L2 rollouts.
    Otherwise fall back to the standard uniform-box ``reset_root_state_uniform``
    + ``reset_joints_by_scale``. If the bank file is missing, every env id falls
    back to the uniform box (training still works, just without the
    randomized-entry benefit).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    bank = _load_bank(bank_path, device)

    if bank is None:
        reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg=asset_cfg)
        reset_joints_by_scale(env, env_ids, joint_position_range, joint_velocity_range, asset_cfg=asset_cfg)
        return

    use_bank = torch.rand(len(env_ids), device=device) < bank_prob
    bank_ids = env_ids[use_bank]
    uniform_ids = env_ids[~use_bank]

    if len(uniform_ids) > 0:
        reset_root_state_uniform(env, uniform_ids, pose_range, velocity_range, asset_cfg=asset_cfg)
        reset_joints_by_scale(env, uniform_ids, joint_position_range, joint_velocity_range, asset_cfg=asset_cfg)

    if len(bank_ids) > 0:
        _apply_bank_states(env, bank_ids, bank, asset)


def reteleport_from_l2_bank(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    bank_path: str,
    prob: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Mid-episode randomized re-entry. (Runs at ``mode="interval"``, i.e.
    periodically *during* an ongoing episode, without ending it.)

    Isaac Lab's EventManager already staggers *when* each env id gets called
    here per-env, via the term's ``interval_range_s``. On top of that, only a
    Bernoulli(``prob``) subset of the envs due this tick are actually
    re-teleported to a fresh state sampled from the pi_L2-induced bank; the
    rest simply continue their current pi_L1-controlled trajectory
    uninterrupted. This implements the AAP formalization's
    ``b_t ~ Bernoulli(p_abst)`` idea (randomize *when* the band is entered)
    while remaining fully on-policy for pi_L1's PPO training: only the state
    is perturbed, every action in the rollout (before and after the
    teleport) is still genuinely pi_L1's own choice.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    bank = _load_bank(bank_path, device)
    if bank is None or len(env_ids) == 0:
        return

    hit = torch.rand(len(env_ids), device=device) < prob
    hit_ids = env_ids[hit]
    if len(hit_ids) == 0:
        return
    _apply_bank_states(env, hit_ids, bank, asset)
