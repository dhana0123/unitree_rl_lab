"""Collect stoppability labels, biased toward failure-near / boundary states.

Two bias modes (can combine):
  1) --push_before_sample: disturb with a velocity push, cruise a few L2 steps,
     then hand off to π_L1 (more unstoppable / boundary labels).
  2) --vstop PATH: after a candidate trigger, keep the sample if
        - L1 failed (y=0), or
        - V_stop is uncertain (in [v_low, v_high]), or
        - random keep with --easy_keep_prob (so easy states are not zero).

Example (recommended for paper):
  python scripts/aap/collect_stoppability.py --headless \\
    --checkpoint_l2 .../model_14900.pt --checkpoint_l1 .../model_5000.pt \\
    --num_envs 64 --num_samples 15000 \\
    --push_before_sample --push_vx 0.9 --push_vy 0.4 \\
    --sample_interval 25 --warmup_steps 30 \\
    --output logs/aap/stoppability_dataset.pt

Second pass (boundary refine with a trained monitor):
  python scripts/aap/collect_stoppability.py --headless \\
    --checkpoint_l2 ... --checkpoint_l1 ... --vstop logs/aap/vstop.pt \\
    --num_samples 8000 --push_before_sample \\
    --v_low 0.25 --v_high 0.75 --easy_keep_prob 0.15 \\
    --append --output logs/aap/stoppability_dataset.pt
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "rsl_rl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Collect AAP stoppability dataset (failure-near biased).")
parser.add_argument("--task_l2", type=str, default="Unitree-G1-29dof-Velocity")
parser.add_argument("--task_l1", type=str, default="Unitree-G1-29dof-SafeStop")
parser.add_argument("--checkpoint_l2", type=str, required=True)
parser.add_argument("--checkpoint_l1", type=str, required=True)
parser.add_argument("--vstop", type=str, default=None, help="Optional monitor for uncertain-state filtering")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_samples", type=int, default=15000)
parser.add_argument("--warmup_steps", type=int, default=30)
parser.add_argument("--sample_interval", type=int, default=25)
parser.add_argument("--fallback_horizon", type=int, default=100)
parser.add_argument("--min_height", type=float, default=0.45)
parser.add_argument("--max_tilt", type=float, default=0.7)
# Failure-near bias
parser.add_argument(
    "--push_before_sample",
    action="store_true",
    default=True,
    help="Apply a velocity push before each L1 handoff (default: on).",
)
parser.add_argument("--no_push_before_sample", action="store_true", help="Disable push-before-sample.")
parser.add_argument("--push_vx", type=float, default=0.9)
parser.add_argument("--push_vy", type=float, default=0.4)
parser.add_argument("--push_delay", type=int, default=5, help="L2 steps after push before handoff.")
parser.add_argument("--v_low", type=float, default=0.25, help="Uncertain band low (with --vstop)")
parser.add_argument("--v_high", type=float, default=0.75, help="Uncertain band high (with --vstop)")
parser.add_argument(
    "--easy_keep_prob",
    type=float,
    default=0.2,
    help="When using --vstop, probability to keep easy (confident+success) samples.",
)
parser.add_argument(
    "--success_keep_prob",
    type=float,
    default=None,
    help=(
        "No-monitor-needed success subsampling: keep ALL failures, keep successes with this "
        "probability. Lets a single collection pass land near a target pos_rate without a "
        "bootstrap V_stop. Ignored if --vstop is set (uses --easy_keep_prob logic instead)."
    ),
)
parser.add_argument("--append", action="store_true", help="Append to existing --output dataset if present.")
parser.add_argument("--output", type=str, default="logs/aap/stoppability_dataset.pt")
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.no_push_before_sample:
    args_cli.push_before_sample = False

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
from vstop_model import load_vstop, to_policy_tensor  # noqa: E402


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


def _apply_push(env, vx: float, vy: float):
    robot = env.unwrapped.scene["robot"]
    try:
        lin = robot.data.root_lin_vel_w.clone()
        ang = robot.data.root_ang_vel_w.clone()
        # Randomize push sign/magnitude a bit for diversity
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


def _keep_mask(success: torch.Tensor, v_pred: torch.Tensor | None, args) -> torch.Tensor:
    """Decide which envs to keep in the dataset (failure-near bias).

    Always keeps every failure. Successes are subsampled so failures aren't
    drowned out, using whichever bias signal is available:
      - if a V_stop monitor is provided: keep uncertain (boundary) successes,
        subsample confident/easy successes at --easy_keep_prob.
      - else if --success_keep_prob is set (<1.0): subsample ALL successes at
        that flat rate (no monitor needed).
      - else: keep everything (original, unbiased behavior).
    """
    device = success.device
    n = success.shape[0]
    failed = ~success
    rand = torch.rand(n, device=device)

    if v_pred is not None:
        uncertain = (v_pred >= args.v_low) & (v_pred <= args.v_high)
        easy = success & ~uncertain
        keep = failed | uncertain | (easy & (rand < args.easy_keep_prob))
        return keep

    if args.success_keep_prob is not None and args.success_keep_prob < 1.0:
        keep = failed | (success & (rand < args.success_keep_prob))
        return keep

    return torch.ones(n, dtype=torch.bool, device=device)


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
    print(f"[INFO] push_before_sample={args_cli.push_before_sample}  vstop={args_cli.vstop}")

    policy_l2, _ = load_inference_policy(env, agent_l2, ckpt_l2)
    policy_l1, _ = load_inference_policy(env, agent_l1, ckpt_l1)

    vstop_net = None
    if args_cli.vstop:
        obs0 = to_policy_tensor(_get_obs(env))
        vstop_net = load_vstop(args_cli.vstop, obs_dim=obs0.shape[-1], device=env.unwrapped.device)
        print(f"[INFO] Loaded V_stop filter band=[{args_cli.v_low}, {args_cli.v_high}]")

    obs_buf: list[torch.Tensor] = []
    label_buf: list[torch.Tensor] = []

    out = Path(args_cli.output)
    if args_cli.append and out.exists():
        old = torch.load(out, map_location="cpu", weights_only=False)
        for i in range(len(old["labels"])):
            obs_buf.append(old["obs"][i].cpu())
            label_buf.append(old["labels"][i].cpu().reshape(()).float())
        print(f"[INFO] Append mode: loaded {len(obs_buf)} existing samples from {out}")

    obs = _get_obs(env)
    step = 0
    skipped = 0
    torch.manual_seed(args_cli.seed)

    with torch.inference_mode():
        while len(obs_buf) < args_cli.num_samples and simulation_app.is_running():
            # Warmup / cruise with L2.
            if step < args_cli.warmup_steps or (step - args_cli.warmup_steps) % args_cli.sample_interval != 0:
                actions = policy_l2(obs)
                obs, _, _, _ = env.step(actions)
                step += 1
                continue

            # --- Failure-near: push, then a few L2 steps, then handoff ---
            if args_cli.push_before_sample:
                _apply_push(env, args_cli.push_vx, args_cli.push_vy)
                for _ in range(max(0, args_cli.push_delay)):
                    actions = policy_l2(obs)
                    obs, _, _, _ = env.step(actions)
                    step += 1

            trigger_obs = to_policy_tensor(obs).detach().clone()
            v_pred = vstop_net(trigger_obs) if vstop_net is not None else None

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
            keep = _keep_mask(success, v_pred, args_cli)

            kept_this = 0
            for i in range(env.unwrapped.num_envs):
                if len(obs_buf) >= args_cli.num_samples:
                    break
                if not bool(keep[i].item()):
                    skipped += 1
                    continue
                obs_buf.append(trigger_obs[i].cpu())
                label_buf.append(torch.tensor(float(success[i].item())))
                kept_this += 1

            labels_t = torch.stack(label_buf) if label_buf else torch.zeros(0)
            pos_rate = float(labels_t.mean().item()) if len(labels_t) else 0.0
            unsafe_n = int((labels_t < 0.5).sum().item()) if len(labels_t) else 0
            print(
                f"[INFO] collected {len(obs_buf)}/{args_cli.num_samples}  "
                f"batch_keep={kept_this}/{env.unwrapped.num_envs}  "
                f"batch_success={success.float().mean().item():.3f}  "
                f"pos_rate={pos_rate:.3f}  unsafe={unsafe_n}  skipped={skipped}"
            )

            obs = _get_obs(env)

    out.parent.mkdir(parents=True, exist_ok=True)
    labels = torch.stack(label_buf)
    metadata = {
        "task_l2": args_cli.task_l2,
        "task_l1": args_cli.task_l1,
        "checkpoint_l2": str(args_cli.checkpoint_l2),
        "checkpoint_l1": str(args_cli.checkpoint_l1),
        "num_envs": args_cli.num_envs,
        "warmup_steps": args_cli.warmup_steps,
        "sample_interval": args_cli.sample_interval,
        "fallback_horizon": args_cli.fallback_horizon,
        "push_before_sample": bool(args_cli.push_before_sample),
        "push_vx": args_cli.push_vx,
        "push_vy": args_cli.push_vy,
        "push_delay": args_cli.push_delay,
        "min_height": args_cli.min_height,
        "max_tilt": args_cli.max_tilt,
        "vstop_filter": args_cli.vstop,
        "v_low": args_cli.v_low,
        "v_high": args_cli.v_high,
        "easy_keep_prob": args_cli.easy_keep_prob,
        "success_keep_prob": args_cli.success_keep_prob,
        "skipped": skipped,
        "seed": args_cli.seed,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save({"obs": torch.stack(obs_buf), "labels": labels, "metadata": metadata}, out)
    print(
        f"[INFO] Saved {len(obs_buf)} samples to {out}  "
        f"pos_rate={labels.mean().item():.3f}  unsafe={(labels < 0.5).sum().item()}"
    )
    pos_rate_final = labels.mean().item()
    if pos_rate_final > 0.92:
        print(
            "[WARN] pos_rate > 0.92 (too few failures). "
            "Increase --push_vx/--push_vy, lower --push_delay, or run a --vstop refine pass."
        )
    elif pos_rate_final < 0.08:
        print(
            "[WARN] pos_rate < 0.08 (too few successes, dataset dominated by failures). "
            "Decrease --push_vx/--push_vy, raise --push_delay (let L2 recover more before handoff), "
            "or lower --num_envs push severity. A healthy target is roughly 0.3-0.7 pos_rate."
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
