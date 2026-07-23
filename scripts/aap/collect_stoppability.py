"""Collect stoppability labels by intervening with π_L1 during π_L2 rollouts.

Example:
  python scripts/aap/collect_stoppability.py --headless \\
    --checkpoint_l2 logs/rsl_rl/unitree_g1_29dof_velocity/.../model_*.pt \\
    --checkpoint_l1 logs/rsl_rl/unitree_g1_29dof_safestop/.../model_*.pt \\
    --num_envs 64 --num_samples 2000 --output logs/aap/stoppability_dataset.pt
"""

"""Launch Isaac Sim Simulator first."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/ and scripts/rsl_rl on path for shared helpers
_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "rsl_rl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402  # scripts/rsl_rl/cli_args.py

parser = argparse.ArgumentParser(description="Collect AAP stoppability dataset.")
parser.add_argument("--task_l2", type=str, default="Unitree-G1-29dof-Velocity")
parser.add_argument("--task_l1", type=str, default="Unitree-G1-29dof-SafeStop")
parser.add_argument("--checkpoint_l2", type=str, required=True)
parser.add_argument("--checkpoint_l1", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_samples", type=int, default=2000)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--sample_interval", type=int, default=40)
parser.add_argument("--fallback_horizon", type=int, default=100)
parser.add_argument("--min_height", type=float, default=0.45)
parser.add_argument("--max_tilt", type=float, default=0.7)
parser.add_argument("--output", type=str, default="logs/aap/stoppability_dataset.pt")
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from importlib.metadata import version

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from policy_utils import load_inference_policy  # noqa: E402


def _get_obs(env):
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    return obs


def _root_height_tilt(env):
    robot = env.unwrapped.scene["robot"]
    height = robot.data.root_pos_w[:, 2]
    grav = robot.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-grav[:, 2], -1.0, 1.0))
    return height, tilt


def main():
    env_cfg = parse_env_cfg(
        args_cli.task_l2,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(args_cli.task_l2, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_l2 = cli_args.parse_rsl_rl_cfg(args_cli.task_l2, args_cli)
    agent_l1 = cli_args.parse_rsl_rl_cfg(args_cli.task_l1, args_cli)

    ckpt_l2 = retrieve_file_path(args_cli.checkpoint_l2)
    ckpt_l1 = retrieve_file_path(args_cli.checkpoint_l1)
    print(f"[INFO] Loading π_L2 from {ckpt_l2}")
    print(f"[INFO] Loading π_L1 from {ckpt_l1}")

    policy_l2, _ = load_inference_policy(env, agent_l2, ckpt_l2)
    policy_l1, _ = load_inference_policy(env, agent_l1, ckpt_l1)

    obs_buf: list[torch.Tensor] = []
    label_buf: list[torch.Tensor] = []

    obs = _get_obs(env)
    step = 0
    torch.manual_seed(args_cli.seed)

    with torch.inference_mode():
        while len(obs_buf) < args_cli.num_samples and simulation_app.is_running():
            # Warmup / cruise with L2.
            if step < args_cli.warmup_steps or (step - args_cli.warmup_steps) % args_cli.sample_interval != 0:
                actions = policy_l2(obs)
                obs, _, _, _ = env.step(actions)
                step += 1
                continue

            trigger_obs = obs.detach().clone()
            fell = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
            for _ in range(args_cli.fallback_horizon):
                actions = policy_l1(obs)
                obs, _, dones, _ = env.step(actions)
                height, tilt = _root_height_tilt(env)
                done_t = dones.bool() if torch.is_tensor(dones) else torch.tensor(dones, device=fell.device).bool()
                fell |= (height < args_cli.min_height) | (tilt > args_cli.max_tilt) | done_t
                step += 1

            height, tilt = _root_height_tilt(env)
            success = (~fell) & (height >= args_cli.min_height) & (tilt <= args_cli.max_tilt)

            for i in range(env.unwrapped.num_envs):
                if len(obs_buf) >= args_cli.num_samples:
                    break
                obs_buf.append(trigger_obs[i].cpu())
                label_buf.append(torch.tensor(float(success[i].item())))

            print(
                f"[INFO] collected {len(obs_buf)}/{args_cli.num_samples}  "
                f"batch_success={success.float().mean().item():.3f}"
            )

            obs = _get_obs(env)

    out = Path(args_cli.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"obs": torch.stack(obs_buf), "labels": torch.stack(label_buf)}, out)
    print(f"[INFO] Saved dataset with {len(obs_buf)} samples to {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
