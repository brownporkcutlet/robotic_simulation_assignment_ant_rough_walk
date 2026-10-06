"""처음 보는 지형에서 넘어지지 않는 Ant: 학습 태스크."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Ant-Rough-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntWalkPPORunnerCfg",
    },
)
