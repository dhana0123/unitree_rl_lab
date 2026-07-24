"""Test pi_L1 in complete isolation: drop it into real bank D_L2 states and
run *only* pi_L1 (no pi_L2, no controller/handoff logic, no V_stop monitor)
to see whether it can recover on its own.

This directly answers "is the fall rate caused by pi_L1's training, or by the
handoff logic / monitor?" -- by removing the handoff logic and monitor from
the picture entirely. Every episode starts by teleporting the robot straight
into a state recorded from real pi_L2 rollouts (scripts/aap/collect_l2_states.py),
then pi_L1 runs solo for --max_steps and we measure fall rate.

By default this also disables the SafeStop env's own `push_robot` and
`reteleport_l2_states` events (scripted mid-episode disturbances), so a
failure can only be attributed to "pi_L1 couldn't recover from the initial
handoff state" -- not to a fresh unrelated push landing mid-episode. Pass
--keep_interference to leave those events on if you want the "as-trained"
dynamics instead.

Example:
  python scripts/aap/eval_l1_standalone.py --headless \\
    --checkpoint_l1 logs/rsl_rl/unitree_g1_29dof_safestop/<run>/model_5000.pt \\
    --bank_path logs/aap/l2_state_bank.pt \\
    --num_envs 64 --episodes 200 --max_steps 300 \\
    --output_dir logs/aap/results_l1_standalone

Optional baseline comparison (generic uniform-box reset instead of the bank,
to see whether pi_L1 is uniformly bad or specifically bad on realistic
post-L2/post-push states):
  python scripts/aap/eval_l1_standalone.py --headless \\
    --checkpoint_l1 ... --bank_prob 0.0 --output_dir logs/aap/results_l1_uniform
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "rsl_rl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate pi_L1 standalone from real bank D_L2 states (no L2, no AAP).")
parser.add_argument("--task_l1", type=str, default="Unitree-G1-29dof-SafeStop")
parser.add_argument("--checkpoint_l1", type=str, required=True)
parser.add_argument("--bank_path", type=str, default="logs/aap/l2_state_bank.pt", help="Path to D_L2 bank .pt file.")
parser.add_argument(
    "--bank_prob",
    type=float,
    default=1.0,
    help="Fraction of resets drawn from the D_L2 bank (1.0 = always; 0.0 = generic uniform-box baseline).",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=200, help="Total trials (approx via env resets).")
parser.add_argument("--max_steps", type=int, default=300, help="Control steps to let pi_L1 run per trial.")
parser.add_argument("--min_height", type=float, default=0.45)
parser.add_argument("--max_tilt", type=float, default=0.7)
parser.add_argument(
    "--keep_interference",
    action="store_true",
    help="Do NOT disable the env's scripted push_robot/reteleport_l2_states events "
    "(default: disabled, so failures are attributable only to the initial handoff state).",
)
parser.add_argument("--output_dir", type=str, default="logs/aap/results_l1_standalone")
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
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task_l1,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )

    # Force every (or --bank_prob fraction of) reset(s) to teleport straight
    # into a state recorded from real pi_L2 rollouts, instead of the default
    # 60/40 bank/uniform mix used during pi_L1 *training*.
    bank_path = str(Path(args_cli.bank_path).resolve())
    if not Path(bank_path).exists():
        raise FileNotFoundError(
            f"D_L2 bank not found at {bank_path}. Run scripts/aap/collect_l2_states.py first."
        )
    env_cfg.events.reset_base_and_joints.params["bank_path"] = bank_path
    env_cfg.events.reset_base_and_joints.params["bank_prob"] = args_cli.bank_prob
    print(f"[INFO] Using D_L2 bank '{bank_path}' with bank_prob={args_cli.bank_prob}")

    # Align the env's own episode length with --max_steps so the underlying
    # env actually times out (and auto-resets -> fresh bank teleport) exactly
    # when we want a new trial, instead of us just resetting our own
    # bookkeeping while the robot keeps walking on the *same* physical
    # rollout (which would silently turn one bank drop into several
    # artificially-inflated "episodes").
    control_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.episode_length_s = args_cli.max_steps * control_dt
    print(f"[INFO] Set episode_length_s={env_cfg.episode_length_s:.2f}s ({args_cli.max_steps} steps @ {1/control_dt:.1f} Hz)")

    if not args_cli.keep_interference:
        # Isolate "can pi_L1 recover from the handoff state" from "did a new
        # scripted disturbance land mid-episode". Isaac Lab skips manager
        # terms set to None.
        env_cfg.events.push_robot = None
        env_cfg.events.reteleport_l2_states = None
        print("[INFO] Disabled push_robot / reteleport_l2_states events for a clean isolation test.")
    else:
        print("[INFO] --keep_interference set: leaving scripted push/reteleport events active.")

    env = gym.make(args_cli.task_l1, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_l1 = cli_args.parse_rsl_rl_cfg(args_cli.task_l1, args_cli)
    ckpt_l1 = retrieve_file_path(args_cli.checkpoint_l1)
    policy_l1, _ = load_inference_policy(env, agent_l1, ckpt_l1)

    torch.manual_seed(args_cli.seed)
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    rows: list[dict] = []
    completed = 0
    ep_step = torch.zeros(num_envs, device=device, dtype=torch.long)
    ep_fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ep_fall_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    ep_min_height = torch.full((num_envs,), float("inf"), device=device)

    with torch.inference_mode():
        env.reset()
        obs = _get_obs(env)

        # Record initial (post-teleport) height as the handoff state quality.
        init_height, _ = _root_height_tilt(env)
        ep_min_height = torch.minimum(ep_min_height, init_height)

        while completed < args_cli.episodes and simulation_app.is_running():
            actions = policy_l1(obs)
            obs, _, dones, _ = env.step(actions)
            height, tilt = _root_height_tilt(env)
            ep_min_height = torch.minimum(ep_min_height, height)

            fell_now = (height < args_cli.min_height) | (tilt > args_cli.max_tilt)
            newly_fell = fell_now & ~ep_fell
            ep_fall_step[newly_fell] = ep_step[newly_fell]
            ep_fell |= fell_now
            ep_step += 1

            done_t = dones.bool() if torch.is_tensor(dones) else torch.tensor(dones, device=device).bool()
            ended = done_t | (ep_step >= args_cli.max_steps)
            if ended.any():
                idxs = ended.nonzero(as_tuple=False).squeeze(-1)
                for i in idxs.tolist():
                    if completed >= args_cli.episodes:
                        break
                    success = (not bool(ep_fell[i].item())) and float(height[i]) >= args_cli.min_height
                    rows.append(
                        {
                            "episode": completed,
                            "fell": int(ep_fell[i].item()),
                            "success": int(success),
                            "fall_step": int(ep_fall_step[i].item()),
                            "steps_survived": int(ep_step[i].item()),
                            "init_height": float(init_height[i].item()) if i < len(init_height) else float("nan"),
                            "min_height": float(ep_min_height[i].item()),
                            "final_height": float(height[i].item()),
                        }
                    )
                    completed += 1
                    ep_step[i] = 0
                    ep_fell[i] = False
                    ep_fall_step[i] = -1
                    ep_min_height[i] = float("inf")
                # New episodes for the ended envs auto-reset via the SafeStop
                # env's own reset() call inside env.step(); re-snapshot their
                # fresh (post-teleport) heights on the next loop iteration.
                new_height, _ = _root_height_tilt(env)
                init_height = torch.where(ended, new_height, init_height)
                ep_min_height = torch.where(ended, new_height, ep_min_height)

            if completed % 20 == 0 and completed > 0:
                n = len(rows)
                fr = sum(r["fell"] for r in rows) / n
                print(f"[INFO] {completed}/{args_cli.episodes} trials  running_fall_rate={fr:.3f}")

    # Write trial-level CSV.
    trials_csv = out_dir / "l1_standalone_trials.csv"
    with trials_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode", "fell", "success", "fall_step", "steps_survived",
                "init_height", "min_height", "final_height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    fall_rate = sum(r["fell"] for r in rows) / max(1, n)
    success_rate = sum(r["success"] for r in rows) / max(1, n)
    fall_steps = [r["fall_step"] for r in rows if r["fell"]]
    mean_fall_step = sum(fall_steps) / len(fall_steps) if fall_steps else float("nan")

    summary_csv = out_dir / "l1_standalone_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "bank_prob", "fall_rate", "success_rate", "mean_fall_step", "max_steps"])
        writer.writerow([n, args_cli.bank_prob, f"{fall_rate:.4f}", f"{success_rate:.4f}", f"{mean_fall_step:.1f}", args_cli.max_steps])

    print()
    print("==================================================================")
    print(f"[RESULT] pi_L1 standalone from D_L2 bank (bank_prob={args_cli.bank_prob}, n={n})")
    print(f"  fall_rate    = {fall_rate:.4f}")
    print(f"  success_rate = {success_rate:.4f}")
    print(f"  mean_fall_step (of those that fell) = {mean_fall_step:.1f} / {args_cli.max_steps}")
    print(f"  Wrote {trials_csv}")
    print(f"  Wrote {summary_csv}")
    print("==================================================================")
    if fall_rate > 0.3:
        print(
            "[CONCLUSION] pi_L1 fails a lot even with ZERO handoff logic, ZERO monitor, and "
            "no L2 involved at all -> the problem is pi_L1's training (undertrained / insufficient "
            "disturbance coverage), not the AAP controller or V_stop monitor."
        )
    else:
        print(
            "[CONCLUSION] pi_L1 recovers fine on its own from bank states -> the push-sweep regression "
            "is more likely coming from the handoff/blend logic or V_stop monitor miscalibration, not "
            "pi_L1's raw capability. Re-check alpha thresholds and V_stop trace behavior."
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
