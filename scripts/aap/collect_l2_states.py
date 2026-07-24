"""Collect a bank of realistic L2-induced states for robust pi_L1 (SafeStop) training.

This implements the "randomized-entry" idea from the AAP formalization (Section 6):
instead of training pi_L1 only from generic reset-randomized box states, expose it
to states that actually arise from running pi_L2 under randomized pushes -- real
mid-gait / post-disturbance states rather than a uniform box around the origin.

The resulting bank (.pt file) is consumed by a custom reset event
(unitree_rl_lab.tasks.aap.mdp.reset_from_l2_bank) that mixes bank samples with the
existing uniform-box reset during pi_L1 training. See safe_stop_env_cfg.py.

Example:
  python scripts/aap/collect_l2_states.py --headless \\
    --checkpoint_l2 logs/rsl_rl/unitree_g1_29dof_velocity/<run>/model_15000.pt \\
    --num_envs 64 --num_samples 20000 \\
    --push_prob 0.6 --push_vx 0.8 --push_vy 0.3 --sample_interval 20 \\
    --output logs/aap/l2_state_bank.pt
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "rsl_rl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Collect L2-induced state bank for robust L1 training.")
parser.add_argument("--task_l2", type=str, default="Unitree-G1-29dof-Velocity")
parser.add_argument("--checkpoint_l2", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_samples", type=int, default=20000)
parser.add_argument("--warmup_steps", type=int, default=30)
parser.add_argument("--sample_interval", type=int, default=20, help="Steps between samples per env.")
parser.add_argument("--push_prob", type=float, default=0.6, help="Probability of a push before each sample round.")
parser.add_argument("--push_vx", type=float, default=0.8)
parser.add_argument("--push_vy", type=float, default=0.3)
parser.add_argument(
    "--push_delay", type=int, default=5, help="L2 steps after push before sampling (lets dynamics respond)."
)
parser.add_argument("--min_height", type=float, default=0.3, help="Skip samples where the robot has already fallen.")
parser.add_argument("--output", type=str, default="logs/aap/l2_state_bank.pt")
parser.add_argument("--seed", type=int, default=7)
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


def _apply_push(env, vx: float, vy: float):
    robot = env.unwrapped.scene["robot"]
    try:
        lin = robot.data.root_lin_vel_w.clone()
        ang = robot.data.root_ang_vel_w.clone()
        n = lin.shape[0]
        sx = torch.sign(torch.randn(n, device=lin.device))
        sy = torch.sign(torch.randn(n, device=lin.device))
        lin[:, 0] += sx * vx
        lin[:, 1] += sy * abs(vy)
        root_vel = torch.cat([lin, ang], dim=-1)
        if hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(root_vel)
        elif hasattr(robot, "write_root_link_velocity_to_sim"):
            robot.write_root_link_velocity_to_sim(root_vel)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Push failed: {exc}")


def _snapshot_state(env):
    """Grab (root pos rel-to-env-origin, root quat, root lin/ang vel, joint pos/vel) for all envs."""
    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins
    root_pos_rel = (robot.data.root_pos_w - origins).detach().clone()
    root_quat = robot.data.root_quat_w.detach().clone()
    root_lin_vel = robot.data.root_lin_vel_w.detach().clone()
    root_ang_vel = robot.data.root_ang_vel_w.detach().clone()
    joint_pos = robot.data.joint_pos.detach().clone()
    joint_vel = robot.data.joint_vel.detach().clone()
    return root_pos_rel, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel


def _root_height(env):
    robot = env.unwrapped.scene["robot"]
    return robot.data.root_pos_w[:, 2]


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
    ckpt_l2 = retrieve_file_path(args_cli.checkpoint_l2)
    policy_l2, _ = load_inference_policy(env, agent_l2, ckpt_l2)

    bank: dict[str, list[torch.Tensor]] = {
        "root_pos_rel": [],
        "root_quat": [],
        "root_lin_vel": [],
        "root_ang_vel": [],
        "joint_pos": [],
        "joint_vel": [],
    }

    torch.manual_seed(args_cli.seed)
    obs = _get_obs(env)
    step = 0
    total = 0

    with torch.inference_mode():
        while total < args_cli.num_samples and simulation_app.is_running():
            actions = policy_l2(obs)
            obs, _, _, _ = env.step(actions)
            step += 1

            if step < args_cli.warmup_steps:
                continue
            if (step - args_cli.warmup_steps) % args_cli.sample_interval != 0:
                continue

            if args_cli.push_prob > 0 and torch.rand(1).item() < args_cli.push_prob:
                _apply_push(env, args_cli.push_vx, args_cli.push_vy)
                for _ in range(max(0, args_cli.push_delay)):
                    actions = policy_l2(obs)
                    obs, _, _, _ = env.step(actions)
                    step += 1

            height = _root_height(env)
            alive_mask = height >= args_cli.min_height
            if not bool(alive_mask.any().item()):
                continue

            root_pos_rel, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel = _snapshot_state(env)
            idxs = alive_mask.nonzero(as_tuple=False).squeeze(-1)
            for i in idxs.tolist():
                if total >= args_cli.num_samples:
                    break
                bank["root_pos_rel"].append(root_pos_rel[i].cpu())
                bank["root_quat"].append(root_quat[i].cpu())
                bank["root_lin_vel"].append(root_lin_vel[i].cpu())
                bank["root_ang_vel"].append(root_ang_vel[i].cpu())
                bank["joint_pos"].append(joint_pos[i].cpu())
                bank["joint_vel"].append(joint_vel[i].cpu())
                total += 1

            print(f"[INFO] step={step} collected={total}/{args_cli.num_samples}")

    out = {k: torch.stack(v) for k, v in bank.items()}
    out["metadata"] = {
        "checkpoint_l2": str(args_cli.checkpoint_l2),
        "num_envs": args_cli.num_envs,
        "sample_interval": args_cli.sample_interval,
        "push_prob": args_cli.push_prob,
        "push_vx": args_cli.push_vx,
        "push_vy": args_cli.push_vy,
        "push_delay": args_cli.push_delay,
        "min_height": args_cli.min_height,
    }
    out_path = Path(args_cli.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[INFO] Wrote {total} L2-induced states to {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
