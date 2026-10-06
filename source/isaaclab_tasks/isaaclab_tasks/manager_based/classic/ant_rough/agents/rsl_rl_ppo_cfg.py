from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntWalkPPORunnerCfg(AntPPORunnerCfg):
    """원본 Ant PPO 설정에서 세 가지만 바꿨다. 로그는 logs/rsl_rl/ant_walk/ 에 쌓인다."""

    experiment_name = "ant_walk"
    max_iterations = 3000  # 지형 + 랜덤화로 어려워졌으니 1000 -> 3000
    save_interval = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,  # False -> True
        critic_obs_normalization=True,  # False -> True
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )
    # algorithm(PPO 하이퍼파라미터)은 원본 그대로 상속
