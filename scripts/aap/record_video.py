"""Record AAP demo videos (~5 min sim time by default) and save MP4s.

Isaac Lab already supports video via gymnasium RecordVideo (same as play.py --video).
This script records Always-L2 / Hard-switch / AAP under the same disturbance.

Sim time math (G1 default):
  step_dt = decimation * sim.dt = 4 * 0.005 = 0.02 s
  5 minutes = 300 s -> 300 / 0.02 = 15000 env steps

Example:
  python scripts/aap/record_video.py \\
    --checkpoint_l2 logs/rsl_rl/unitree_g1_29dof_velocity/.../model_5000.pt \\
    --checkpoint_l1 logs/rsl_rl/unitree_g1_29dof_safestop/.../model_*.pt \\
    --vstop logs/aap/vstop.pt \\
    --conditions always_l2,hard_switch,aap \\
    --duration_sec 300 \\
    --output_dir logs/aap/videos
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

parser = argparse.ArgumentParser(description="Record AAP comparison videos and save MP4s.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity")
parser.add_argument("--task_l1", type=str, default="Unitree-G1-29dof-SafeStop")
parser.add_argument("--checkpoint_l2", type=str, required=True)
parser.add_argument("--checkpoint_l1", type=str, required=True)
parser.add_argument("--vstop", type=str, default=None, help="Required unless --conditions is only always_l2")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 for clean paper videos")
parser.add_argument(
    "--duration_sec",
    type=float,
    default=300.0,
    help="Simulated seconds to record (default 300 = 5 minutes).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Override length in env steps. If unset, computed from --duration_sec / step_dt.",
)
parser.add_argument("--fps", type=int, default=50, help="Video FPS hint for RecordVideo (if supported).")
parser.add_argument("--push_step", type=int, default=100, help="Apply push at this step (0 disables).")
parser.add_argument("--push_vx", type=float, default=0.8)
parser.add_argument("--push_vy", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=0.7)
parser.add_argument("--alpha_low", type=float, default=0.5)
parser.add_argument("--alpha_high", type=float, default=0.7)
parser.add_argument(
    "--ema_beta",
    type=float,
    default=0.85,
    help="Temporal smoothing on the AAP blend weight w (0 = no smoothing). Only affects 'aap'.",
)
parser.add_argument("--output_dir", type=str, default="logs/aap/videos")
parser.add_argument(
    "--conditions",
    type=str,
    default="always_l2,hard_switch,aap",
    help="Comma-separated: always_l2,always_l1,hard_switch,aap",
)
parser.add_argument("--name_prefix", type=str, default="aap_demo", help="Filename prefix for saved videos")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras required for rgb_array / RecordVideo
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from importlib.metadata import version

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from controller import select_controller  # noqa: E402
from policy_utils import load_inference_policy  # noqa: E402
from vstop_model import load_vstop, to_policy_tensor  # noqa: E402


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
        lin[:, 0] += vx
        lin[:, 1] += vy
        root_vel = torch.cat([lin, ang], dim=-1)
        if hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(root_vel)
        elif hasattr(robot, "write_root_link_velocity_to_sim"):
            robot.write_root_link_velocity_to_sim(root_vel)
        else:
            print("[WARN] Could not apply scripted push.")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Scripted push failed ({exc}).")


def _record_one(condition: str, video_length: int, out_dir: Path):
    """Build env, load policies, record one MP4 for this condition."""
    cond_dir = out_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    video_kwargs = {
        "video_folder": str(cond_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": video_length,
        "name_prefix": f"{args_cli.name_prefix}_{condition}",
        "disable_logger": True,
    }
    # fps is supported by some gymnasium versions
    try:
        env = gym.wrappers.RecordVideo(env, **video_kwargs, fps=args_cli.fps)
    except TypeError:
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print(f"[INFO] Recording condition={condition}")
    print_dict(video_kwargs, nesting=2)

    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_l2 = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_l1 = cli_args.parse_rsl_rl_cfg(args_cli.task_l1, args_cli)
    policy_l2, _ = load_inference_policy(env, agent_l2, retrieve_file_path(args_cli.checkpoint_l2))
    policy_l1, _ = load_inference_policy(env, agent_l1, retrieve_file_path(args_cli.checkpoint_l1))

    needs_vstop = condition.lower() not in {"always_l2", "l2", "always_l1", "l1"}
    vstop_net = None
    if needs_vstop:
        if not args_cli.vstop:
            raise ValueError(f"--vstop is required for condition '{condition}'")
        obs0 = to_policy_tensor(_get_obs(env))
        vstop_net = load_vstop(args_cli.vstop, obs_dim=obs0.shape[-1], device=env.unwrapped.device)

    obs = _get_obs(env)
    step_dt = float(env.unwrapped.step_dt)
    print(
        f"[INFO] step_dt={step_dt:.4f}s  video_length={video_length} steps  "
        f"~{video_length * step_dt:.1f}s sim time"
    )

    prev_w = torch.ones(env.unwrapped.num_envs, device=env.unwrapped.device)

    with torch.inference_mode():
        for t in range(video_length):
            if not simulation_app.is_running():
                break
            if args_cli.push_step > 0 and t == args_cli.push_step:
                _apply_push(env, args_cli.push_vx, args_cli.push_vy)
                print(f"[INFO] Applied push at step {t}")

            a2 = policy_l2(obs)
            a1 = policy_l1(obs)
            if vstop_net is None:
                # Dummy score: always high for L2-only / low for L1-only paths via select_controller
                v = torch.ones(obs.shape[0], device=obs.device)
            else:
                v = vstop_net(to_policy_tensor(obs))

            out = select_controller(
                condition,
                a2,
                a1,
                v,
                alpha=args_cli.alpha,
                alpha_low=args_cli.alpha_low,
                alpha_high=args_cli.alpha_high,
                w_prev=prev_w,
                ema_beta=args_cli.ema_beta,
            )
            prev_w = out.w_l2
            obs, _, _, _ = env.step(out.actions)

            if (t + 1) % 500 == 0:
                print(f"[INFO] {condition}: {t + 1}/{video_length} steps")

    env.close()
    print(f"[INFO] Saved videos under: {cond_dir}")
    for p in sorted(cond_dir.glob("*.mp4")):
        print(f"       {p}")


def main():
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args_cli.conditions.split(",") if c.strip()]
    if any(c.lower() not in {"always_l2", "l2", "always_l1", "l1"} for c in conditions) and not args_cli.vstop:
        raise ValueError("--vstop is required when recording hard_switch or aap")

    # Probe step_dt once with a tiny env to compute length from duration.
    if args_cli.video_length is None:
        probe_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=1,
            entry_point_key="play_env_cfg_entry_point",
        )
        probe = gym.make(args_cli.task, cfg=probe_cfg)
        step_dt = float(probe.unwrapped.step_dt)
        probe.close()
        video_length = max(1, int(round(args_cli.duration_sec / step_dt)))
        print(
            f"[INFO] duration_sec={args_cli.duration_sec} / step_dt={step_dt:.4f} "
            f"-> video_length={video_length} steps (~{video_length * step_dt / 60:.2f} min sim)"
        )
    else:
        video_length = args_cli.video_length
        print(f"[INFO] Using explicit --video_length={video_length}")

    for cond in conditions:
        _record_one(cond, video_length, out_dir)

    print(f"[INFO] Done. All videos in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
