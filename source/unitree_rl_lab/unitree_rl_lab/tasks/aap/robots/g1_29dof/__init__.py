import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-SafeStop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.safe_stop_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.safe_stop_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.aap.agents.rsl_rl_ppo_cfg:SafeStopPPORunnerCfg",
    },
)
