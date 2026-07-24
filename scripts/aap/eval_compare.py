"""Evaluate Always-L2 / Hard-switch (PRISM) / AAP and write paper tables CSV + plots.

Example:
  python scripts/aap/eval_compare.py --headless \\
    --checkpoint_l2 .../model.pt --checkpoint_l1 .../model.pt \\
    --vstop logs/aap/vstop.pt --num_envs 32 --episodes 50 \\
    --output_dir logs/aap/results
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

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

parser = argparse.ArgumentParser(description="Compare AAP vs hard switch vs always L2.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity")
parser.add_argument("--task_l1", type=str, default="Unitree-G1-29dof-SafeStop")
parser.add_argument("--checkpoint_l2", type=str, required=True)
parser.add_argument("--checkpoint_l1", type=str, required=True)
parser.add_argument("--vstop", type=str, required=True, help="Path to trained V_stop .pt")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=50, help="Episodes per condition (approx via env resets)")
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--push_step", type=int, default=80, help="Apply velocity push at this step")
parser.add_argument("--push_vx", type=float, default=0.8)
parser.add_argument("--push_vy", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=0.7, help="Hard-switch threshold")
parser.add_argument("--alpha_low", type=float, default=0.5)
parser.add_argument("--alpha_high", type=float, default=0.7)
parser.add_argument("--min_height", type=float, default=0.45)
parser.add_argument("--max_tilt", type=float, default=0.7)
parser.add_argument("--output_dir", type=str, default="logs/aap/results")
parser.add_argument(
    "--conditions",
    type=str,
    default="always_l2,hard_switch,aap",
    help="Comma-separated: always_l2,always_l1,hard_switch,aap",
)
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

from controller import select_controller  # noqa: E402
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


def _apply_push(env, vx: float, vy: float, env_ids: torch.Tensor | None = None):
    """Add a base linear velocity impulse to `env_ids` (or all envs if None).

    (best-effort across Isaac Lab versions)
    """
    robot = env.unwrapped.scene["robot"]
    try:
        lin = robot.data.root_lin_vel_w.clone()
        ang = robot.data.root_ang_vel_w.clone()
        if env_ids is None:
            lin[:, 0] += vx
            lin[:, 1] += vy
        else:
            lin[env_ids, 0] += vx
            lin[env_ids, 1] += vy
        root_vel = torch.cat([lin, ang], dim=-1)
        if env_ids is not None:
            root_vel = root_vel[env_ids]
        write_kwargs = {} if env_ids is None else {"env_ids": env_ids}
        if hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(root_vel, **write_kwargs)
        elif hasattr(robot, "write_root_link_velocity_to_sim"):
            robot.write_root_link_velocity_to_sim(root_vel, **write_kwargs)
        else:
            print("[WARN] Could not apply scripted push; relying on env interval push.")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Scripted push failed ({exc}); relying on env interval push.")


def _run_condition(env, policy_l2, policy_l1, vstop_net, condition: str, args) -> list[dict]:
    rows: list[dict] = []
    completed = 0
    obs = _get_obs(env)
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    ep_step = torch.zeros(num_envs, device=device, dtype=torch.long)
    ep_fell = torch.zeros(num_envs, device=device, dtype=torch.bool)
    ep_peak_jerk = torch.zeros(num_envs, device=device)
    prev_action = None
    # Persistent per-env previous blend weight, used only to detect
    # authority switches for the jerk metric (Eq. 13). Assume full L2
    # authority (w=1) at the start of every episode. NOTE: this must only be
    # reset per-env on that env's own episode boundary, not for the whole
    # batch whenever *any* env finishes -- with num_envs>1 that happens on
    # almost every step.
    prev_w = torch.ones(num_envs, device=device)
    fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
    v_hist: list[float] = []
    w_hist: list[float] = []

    with torch.inference_mode():
        while completed < args.episodes and simulation_app.is_running():
            # Scripted push to create risk. Applied per-env, to exactly the
            # envs whose OWN episode has just reached push_step -- not only
            # when the whole batch happens to be synchronized. With
            # num_envs>1, envs desync after their first episode boundary
            # (different envs fall / time out at different real steps), so
            # requiring the whole batch to be at push_step simultaneously
            # (the old `.all()` check) meant only the very first batch of
            # episodes ever actually got pushed; every later episode ran
            # completely unperturbed and trivially "succeeded".
            if args.push_step > 0:
                push_mask = ep_step == args.push_step
                if push_mask.any():
                    push_ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                    _apply_push(env, args.push_vx, args.push_vy, env_ids=push_ids)

            a2 = policy_l2(obs)
            a1 = policy_l1(obs)
            v = vstop_net(to_policy_tensor(obs))
            out = select_controller(
                condition,
                a2,
                a1,
                v,
                alpha=args.alpha,
                alpha_low=args.alpha_low,
                alpha_high=args.alpha_high,
            )
            actions = out.actions

            if prev_action is not None:
                jerk = (actions - prev_action).norm(dim=-1)
                # Emphasize jerk when authority changes (handoff).
                switched = (out.w_l2 - prev_w).abs() > 1e-3
                jerk = torch.where(switched, jerk, jerk * 0.25)
                # Suppress spurious jerk for envs whose prev_action/prev_w
                # refer to a state before their own episode reset.
                jerk = torch.where(fresh, torch.zeros_like(jerk), jerk)
                ep_peak_jerk = torch.maximum(ep_peak_jerk, jerk)
            prev_action = actions
            prev_w = out.w_l2
            fresh = torch.zeros(num_envs, dtype=torch.bool, device=device)

            v_hist.append(float(v.mean().item()))
            w_hist.append(float(out.w_l2.mean().item()))

            obs, _, dones, _ = env.step(actions)
            height, tilt = _root_height_tilt(env)
            done_t = dones.bool() if torch.is_tensor(dones) else torch.tensor(dones, device=ep_fell.device).bool()
            ep_fell |= (height < args.min_height) | (tilt > args.max_tilt)
            ep_step += 1

            # Episode ends on done or max steps.
            ended = done_t | (ep_step >= args.max_steps)
            if ended.any():
                idxs = ended.nonzero(as_tuple=False).squeeze(-1)
                for i in idxs.tolist():
                    if completed >= args.episodes:
                        break
                    success = (not bool(ep_fell[i].item())) and float(height[i]) >= args.min_height
                    rows.append(
                        {
                            "condition": condition,
                            "episode": completed,
                            "fell": int(ep_fell[i].item()),
                            "success": int(success),
                            "peak_jerk": float(ep_peak_jerk[i].item()),
                            "final_height": float(height[i].item()),
                            "mean_v_stop": float(v.mean().item()),
                        }
                    )
                    completed += 1
                    ep_step[i] = 0
                    ep_fell[i] = False
                    ep_peak_jerk[i] = 0.0
                # Reset jerk-continuity state only for the envs that
                # actually ended (per-env, not the whole batch).
                prev_w[idxs] = 1.0
                fresh[idxs] = True

    # Save traces for first condition run plotting helper.
    trace_path = Path(args.output_dir) / f"trace_{condition}.pt"
    torch.save({"v_stop": torch.tensor(v_hist), "w_l2": torch.tensor(w_hist)}, trace_path)
    return rows


def _write_tables(rows: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "table_a_trials.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["condition", "episode", "fell", "success", "peak_jerk", "final_height", "mean_v_stop"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate Table A
    summary_csv = output_dir / "table_a_summary.csv"
    conditions = sorted({r["condition"] for r in rows})
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "n", "fall_rate", "success_rate", "mean_peak_jerk", "std_peak_jerk"])
        for c in conditions:
            subset = [r for r in rows if r["condition"] == c]
            n = len(subset)
            fall_rate = sum(r["fell"] for r in subset) / max(1, n)
            success_rate = sum(r["success"] for r in subset) / max(1, n)
            jerks = torch.tensor([r["peak_jerk"] for r in subset], dtype=torch.float32)
            writer.writerow(
                [c, n, f"{fall_rate:.4f}", f"{success_rate:.4f}", f"{jerks.mean().item():.4f}", f"{jerks.std().item():.4f}"]
            )
    print(f"[INFO] Wrote {raw_csv}")
    print(f"[INFO] Wrote {summary_csv}")


def main():
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_l2 = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_l1 = cli_args.parse_rsl_rl_cfg(args_cli.task_l1, args_cli)

    ckpt_l2 = retrieve_file_path(args_cli.checkpoint_l2)
    ckpt_l1 = retrieve_file_path(args_cli.checkpoint_l1)
    policy_l2, _ = load_inference_policy(env, agent_l2, ckpt_l2)
    policy_l1, _ = load_inference_policy(env, agent_l1, ckpt_l1)

    # NOTE: keep every env interaction (resets included) inside a single
    # inference_mode context. Auto-resets triggered from inside a step() call
    # that runs under inference_mode() can lazily allocate internal buffers
    # (e.g. root_link_pose_w) as "inference tensors"; a later env.reset()
    # called outside inference_mode would then fail with
    # "Inplace update to inference tensor outside InferenceMode is not
    # allowed" when writing to those same buffers.
    with torch.inference_mode():
        # Infer obs dim from one forward.
        obs0 = to_policy_tensor(_get_obs(env))
        vstop_net = load_vstop(args_cli.vstop, obs_dim=obs0.shape[-1], device=env.unwrapped.device)

        all_rows: list[dict] = []
        conditions = [c.strip() for c in args_cli.conditions.split(",") if c.strip()]
        for cond in conditions:
            print(f"[INFO] Evaluating condition: {cond}")
            env.reset()
            rows = _run_condition(env, policy_l2, policy_l1, vstop_net, cond, args_cli)
            all_rows.extend(rows)

    _write_tables(all_rows, out_dir)

    # Band ablation stub table (Table B): reuses same trial file naming.
    band_csv = out_dir / "table_b_band_template.csv"
    with band_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha_low", "alpha_high", "fall_rate", "success_rate", "mean_peak_jerk", "notes"])
        w.writerow([args_cli.alpha, args_cli.alpha, "", "", "", "hard switch (=PRISM)"])
        w.writerow([args_cli.alpha_low, args_cli.alpha_high, "", "", "", "fill by re-running --conditions aap"])
    print(f"[INFO] Wrote band template {band_csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
