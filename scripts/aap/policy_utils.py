"""Helpers shared by AAP Isaac scripts (policy loading)."""

from __future__ import annotations

import os

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.assets import retrieve_file_path


def resolve_checkpoint(experiment_name: str, load_run: str | None, load_checkpoint: str | None, checkpoint: str | None) -> str:
    if checkpoint:
        return retrieve_file_path(checkpoint)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    return get_checkpoint_path(log_root, load_run, load_checkpoint)


def load_inference_policy(env, agent_cfg: RslRlOnPolicyRunnerCfg, checkpoint_path: str):
    """Create an OnPolicyRunner, load weights, return inference callable."""
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    return runner.get_inference_policy(device=env.unwrapped.device), runner
