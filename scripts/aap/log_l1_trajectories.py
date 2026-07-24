"""Log per-step pi_L1 trajectories from real D_L2 bank states to distinguish
"no corrective effort" (reward misweighting, e.g. action_rate/joint_deviation
penalties suppressing recovery motion) from "wrong corrective effort" (bad
training coverage: it tries, but in a direction that doesn't work).

For a small batch of trials, every control step we log:
  - base tilt angle and height
  - raw action norm and action-rate (||a_t - a_{t-1}||^2, the exact quantity
    the `action_rate` reward term (c2 = -0.2) penalizes)
  - joint deviation-from-default for the specific joint groups that have
    their own reward penalty (hip_roll/hip_yaw "legs", "waist", "arms") --
    exactly the DOFs a real recovery strategy (widen stance / twist / arm
    swing) would need to move, and exactly what `joint_deviation_l1` pushes
    back toward zero.
  - the *actual* per-step weighted reward contribution of every active
    reward term, pulled directly from the reward manager (no re-derivation,
    so it matches training exactly), if the installed Isaac Lab exposes
    `RewardManager._step_reward` (falls back gracefully if not).

Writes one CSV per trial plus a comparison PNG (tilt / action-rate / leg
deviation over time, colored by outcome) so you can eyeball whether the
policy is doing nothing, or doing something that doesn't work.

Example:
  python scripts/aap/log_l1_trajectories.py --headless \\
    --checkpoint_l1 logs/rsl_rl/unitree_g1_29dof_safestop/<run>/model_5000.pt \\
    --bank_path logs/aap/l2_state_bank.pt \\
    --num_trials 24 --max_log_steps 40 \\
    --output_dir logs/aap/l1_trajectories
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

parser = argparse.ArgumentParser(description="Log per-step pi_L1 trajectories from real D_L2 bank states.")
parser.add_argument("--task_l1", type=str, default="Unitree-G1-29dof-SafeStop")
parser.add_argument("--checkpoint_l1", type=str, required=True)
parser.add_argument("--bank_path", type=str, default="logs/aap/l2_state_bank.pt")
parser.add_argument("--bank_prob", type=float, default=1.0)
parser.add_argument("--num_trials", type=int, default=24, help="Number of envs / trials to log (one per env).")
parser.add_argument("--max_log_steps", type=int, default=40, help="Control steps to log per trial.")
parser.add_argument("--min_height", type=float, default=0.45)
parser.add_argument("--max_tilt", type=float, default=0.7)
parser.add_argument("--keep_interference", action="store_true", help="Leave push_robot/reteleport events active.")
parser.add_argument("--output_dir", type=str, default="logs/aap/l1_trajectories")
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

# Joint groups mirroring the reward penalties in safe_stop_env_cfg.py, so we
# can directly see whether the policy is (not) moving the exact DOFs those
# terms discourage.
JOINT_GROUPS = {
    "legs_hip": [".*_hip_roll_joint", ".*_hip_yaw_joint"],
    "waist": ["waist.*"],
    "arms": [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
}


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
        num_envs=args_cli.num_trials,
        entry_point_key="play_env_cfg_entry_point",
    )

    bank_path = str(Path(args_cli.bank_path).resolve())
    if not Path(bank_path).exists():
        raise FileNotFoundError(f"D_L2 bank not found at {bank_path}. Run scripts/aap/collect_l2_states.py first.")
    env_cfg.events.reset_base_and_joints.params["bank_path"] = bank_path
    env_cfg.events.reset_base_and_joints.params["bank_prob"] = args_cli.bank_prob

    # Generous margin so the env's own time_out never truncates the log window.
    control_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.episode_length_s = (args_cli.max_log_steps + 20) * control_dt

    if not args_cli.keep_interference:
        env_cfg.events.push_robot = None
        env_cfg.events.reteleport_l2_states = None

    env = gym.make(args_cli.task_l1, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_l1 = cli_args.parse_rsl_rl_cfg(args_cli.task_l1, args_cli)
    ckpt_l1 = retrieve_file_path(args_cli.checkpoint_l1)
    policy_l1, _ = load_inference_policy(env, agent_l1, ckpt_l1)

    torch.manual_seed(args_cli.seed)
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    group_ids = {}
    for gname, patterns in JOINT_GROUPS.items():
        ids: list[int] = []
        for p in patterns:
            found_ids, _ = robot.find_joints(p)
            ids.extend(found_ids)
        group_ids[gname] = sorted(set(ids))
        print(f"[INFO] Joint group '{gname}' -> {len(group_ids[gname])} joints: {group_ids[gname]}")

    reward_manager = getattr(env.unwrapped, "reward_manager", None)
    reward_term_names: list[str] = []
    if reward_manager is not None and hasattr(reward_manager, "active_terms"):
        reward_term_names = list(reward_manager.active_terms)
        if not hasattr(reward_manager, "_step_reward"):
            print("[WARN] reward_manager has no `_step_reward` buffer on this Isaac Lab version; "
                  "per-term reward columns will be omitted.")
            reward_term_names = []
    else:
        print("[WARN] Could not access reward_manager; per-term reward columns will be omitted.")

    default_joint_pos = robot.data.default_joint_pos.clone()

    with torch.inference_mode():
        env.reset()
        obs = _get_obs(env)

        prev_action = None
        fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
        fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

        # trial_logs[i] = list of per-step dict rows for env i
        trial_logs: list[list[dict]] = [[] for _ in range(num_envs)]

        for step in range(args_cli.max_log_steps):
            actions = policy_l1(obs)
            obs, _, dones, _ = env.step(actions)

            height, tilt = _root_height_tilt(env)
            joint_pos = robot.data.joint_pos

            action_norm = actions.norm(dim=-1)
            if prev_action is not None:
                action_rate_raw = (actions - prev_action).square().sum(dim=-1)
            else:
                action_rate_raw = torch.zeros(num_envs, device=device)

            group_dev = {}
            for gname, ids in group_ids.items():
                if len(ids) == 0:
                    group_dev[gname] = torch.zeros(num_envs, device=device)
                    continue
                group_dev[gname] = (joint_pos[:, ids] - default_joint_pos[:, ids]).abs().sum(dim=-1)

            step_terms = None
            if reward_term_names:
                step_terms = reward_manager._step_reward.clone()  # (num_envs, num_terms)

            newly_fell = ((height < args_cli.min_height) | (tilt > args_cli.max_tilt)) & ~fell
            fall_step[newly_fell] = step
            fell |= newly_fell

            for i in range(num_envs):
                row = {
                    "step": step,
                    "fell_so_far": int(fell[i].item()),
                    "height": float(height[i].item()),
                    "tilt_rad": float(tilt[i].item()),
                    "action_norm": float(action_norm[i].item()),
                    "action_rate_raw": float(action_rate_raw[i].item()),
                }
                for gname in JOINT_GROUPS:
                    row[f"dev_{gname}"] = float(group_dev[gname][i].item())
                if step_terms is not None:
                    for j, tname in enumerate(reward_term_names):
                        row[f"rew_{tname}"] = float(step_terms[i, j].item())
                trial_logs[i].append(row)

            prev_action = actions

    # --- Write per-trial CSVs ---
    fieldnames = list(trial_logs[0][0].keys()) if trial_logs and trial_logs[0] else []
    for i in range(num_envs):
        tag = "fell" if fell[i].item() else "ok"
        fstep = int(fall_step[i].item())
        path = out_dir / f"traj_env{i:02d}_{tag}_fallstep{fstep}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trial_logs[i])

    n_fell = int(fell.sum().item())
    print(f"[INFO] Logged {num_envs} trials to {out_dir} ({n_fell} fell, {num_envs - n_fell} survived to step {args_cli.max_log_steps}).")
    fall_steps = [int(fall_step[i].item()) for i in range(num_envs) if fell[i].item()]
    if fall_steps:
        print(f"[INFO] Fall steps among trials that fell: {sorted(fall_steps)}")

    # --- Plot gallery ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
        metrics = [
            ("tilt_rad", "Tilt angle (rad)"),
            ("action_rate_raw", r"$\|a_t - a_{t-1}\|^2$ (action_rate raw)"),
            ("dev_legs_hip", "Hip roll/yaw |dev from default| (sum)"),
            ("height", "Base height (m)"),
        ]
        for ax, (key, ylabel) in zip(axes, metrics):
            for i in range(num_envs):
                xs = [r["step"] for r in trial_logs[i]]
                ys = [r[key] for r in trial_logs[i]]
                color = "#d62728" if fell[i].item() else "#2ca02c"
                ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1.0)
                if fell[i].item():
                    fs = int(fall_step[i].item())
                    if 0 <= fs < len(ys):
                        ax.scatter([fs], [ys[fs]], color="black", s=10, zorder=5)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.4)
        axes[-1].set_xlabel("Step (red = fell, green = survived; dot = fall detected)")
        fig.suptitle("pi_L1 standalone trajectories from D_L2 bank states")
        fig.tight_layout()
        fig_path = out_dir / "fig_trajectories.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Wrote {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Plotting failed ({exc}); CSVs are still written.")

    if reward_term_names:
        print(f"[INFO] Logged per-step reward terms: {reward_term_names}")
        print(
            "[INFO] Compare mean rew_action_rate / rew_joint_deviation_legs / rew_joint_deviation_waists "
            "in the first ~5-10 steps of FELL trials vs the alive/flat_orientation terms: if the "
            "penalty terms dominate right when tilt starts increasing, that's evidence the reward "
            "is suppressing corrective effort. If action_rate/dev terms stay small while tilt still "
            "grows unchecked, the policy is making little effort at all -> also points at reward "
            "shaping (or an undertrained value function) rather than 'wrong direction' effort."
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
