"""Held-out evaluation environments for the Ant task (do not train on these)."""

import gymnasium as gym

# evaluate with the agent config of the training task, so that the network architecture matches
_AGENT_CFG = "isaaclab_tasks.manager_based.classic.ant_rough.agents.rsl_rl_ppo_cfg:AntWalkPPORunnerCfg"

for _name in ["Flat", "Waves", "Boxes", "Stairs", "Mixed"]:
    gym.register(
        id=f"Isaac-Ant-Unseen-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_unseen_env_cfg:AntUnseen{_name}EnvCfg",
            "rsl_rl_cfg_entry_point": _AGENT_CFG,
        },
    )
