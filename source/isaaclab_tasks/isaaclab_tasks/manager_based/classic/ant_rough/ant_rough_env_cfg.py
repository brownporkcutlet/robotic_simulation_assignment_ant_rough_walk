"""처음 보는 지형에서 넘어지지 않는 Ant: 학습 환경 설정 (Isaac-Ant-Rough-Walk-v0).

원본 Ant(Isaac-Ant-v0)에서 아래만 바꾼다. 로봇 하드웨어와 물리 설정은 원본 그대로다.
- 학습 분포: 다양한 지형(요철, 경사, 단차) + 로봇 마찰 랜덤화
- 관측: 월드 좌표 몸통 높이(base_height) 제거. 평지에서만 통하는 지름길이다
- 보상: 진행 보상 속도 상한, 네 발이 모두 뜨면 벌점, 발 충격 벌점, 몸통 안정·동작 부드러움 벌점, 넘어짐 벌점
- 종료: 뒤집힘(몸통 기울기 90도 초과)도 넘어짐으로 종료
- 센서: 발 접촉 센서 (보상 계산에만 사용, 관측에는 넣지 않음)

평가용 시험 지형은 classic/ant_unseen에 있다.
"""

import math
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, EventCfg, RewardsCfg, TerminationsCfg
from isaaclab_tasks.manager_based.classic.humanoid.mdp import progress_reward

##
# 학습 분포와 관측
##


def apply_training_terrain(cfg: AntEnvCfg):
    """평지 대신 다양한 지형에서 학습한다. 시험 지형(ant_unseen)의 waves/boxes/stairs와는 겹치지 않는다."""
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        num_rows=20,  # +x 방향 160m
        num_cols=10,
        border_width=20.0,
        curriculum=False,
        # 모든 지형의 높이는 0 이상 (원본 넘어짐 규칙이 월드 z < 0.31이라 구덩이가 있으면 억울한 종료가 생긴다)
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
            # 10cm 칸마다 0~8cm 랜덤 요철
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.2, noise_range=(0.0, 0.08), noise_step=0.01),
            # 사각뿔 언덕, 경사 0~0.4
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.3, slope_range=(0.0, 0.4), platform_width=2.0),
            # 단차: 바닥 위에 0.4~1.2m 폭, 3~12cm 높이 블록을 흩뿌림
            "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.4,
                obstacle_height_mode="fixed",
                obstacle_width_range=(0.4, 1.2),
                obstacle_height_range=(0.03, 0.12),
                num_obstacles=80,
                platform_width=2.0,
            ),
        },
    )
    cfg.scene.terrain.max_init_terrain_level = 0  # 모두 첫 줄에서 출발
    cfg.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # 거친 지형용 접촉 버퍼


@configclass
class AntFrictionEventCfg(EventCfg):  # 원본 리셋 이벤트 2개는 상속으로 유지
    """로봇 재질의 마찰을 env마다 다르게 뽑는다 (시작할 때 한 번)."""

    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 지면(1.0)과 곱하기로 합쳐지므로 이 값이 그대로 실효 마찰이 된다
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,  # 동마찰 <= 정마찰
        },
    )


def remove_world_height_obs(cfg: AntEnvCfg):
    """관측에서 base_height(월드 좌표 몸통 높이)를 뺀다 (60 -> 59차원).

    평지 학습 정책은 이 값이 0.2m만 달라져도 멈췄다. 지면이 0보다 높은 곳에서는 자세가 같아도 값이 달라진다.
    """
    cfg.observations.policy.base_height = None


##
# 보상과 종료
##


class capped_progress_reward(progress_reward):
    """목표 쪽으로 다가가는 속도(m/s). max_speed를 넘는 부분은 보상하지 않는다.

    원래 진행 보상은 빠를수록 커서, 뛰어오르며 달리는 걸음(초속 5~7m, 몸통 위아래 +-12~15cm)이 최적이 된다.
    """

    def __call__(
        self, env, target_pos: tuple[float, float, float], max_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        return torch.clamp(super().__call__(env, target_pos, asset_cfg), max=max_speed)


def feet_off_ground(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """네 발이 모두 땅에서 떨어져 있으면 1, 하나라도 닿아 있으면 0. 접촉은 발에 걸리는 힘이 threshold(N)를 넘는지로 본다."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    in_contact = forces.norm(dim=-1).max(dim=1).values > threshold
    return (~in_contact.any(dim=1)).float()


@configclass
class AntWalkRewardsCfg(RewardsCfg):
    """원본 Ant 보상 7개 중 진행 보상만 속도 상한 버전으로 바꾸고, 벌점 6개를 더한다.

    넘어짐을 로봇별로 기록해 보니, 모든 넘어짐은 발이 땅에 파묻혔다가 물리 엔진에 튕겨 나가 공중에서 뒤집히는 것이었다.
    그래서 "발이 세게 박히지 않는 걸음"을 만드는 항들이다. 가중치는 초당 크기 기준이고 환경이 dt(1/60)를 곱한다.
    """

    # 진행 보상: 초속 2m까지만
    progress = RewTerm(
        func=capped_progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0), "max_speed": 2.0}
    )
    # 넘어지는 순간 -300 x dt = -5
    fall = RewTerm(func=mdp.is_terminated, weight=-300.0)
    # 몸통 안정·동작 부드러움
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.4)  # 위아래로 튀는 움직임
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)  # 몸통이 앞뒤·좌우로 흔들리는 움직임
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.25)  # 매 스텝 급변하는 토크 명령
    # 네 발이 모두 뜬 시간에 초당 -1
    flight = RewTerm(func=feet_off_ground, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("foot_contact")})
    # 발 충격: 발 하나에 로봇 무게(8.9N)의 2배를 넘는 힘이 걸리면 넘는 만큼(N) 벌점
    foot_impact = RewTerm(
        func=mdp.contact_forces, weight=-0.03, params={"threshold": 18.0, "sensor_cfg": SceneEntityCfg("foot_contact")}
    )


@configclass
class AntFlipTerminationsCfg(TerminationsCfg):
    """원본 종료 조건(시간 초과, 월드 z 몸통 높이 < 0.31) + 뒤집힘.

    월드 z 기준은 솟은 땅에서 발동하지 않아 뒤집힌 로봇을 놓친다. 뒤집힌 로봇은 다시 일어나지 못하므로,
    몸통이 90도 넘게 기울면 지형과 상관없이 넘어진 것으로 본다.
    """

    flipped = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 2})


##
# 학습 환경
##


@configclass
class AntWalkEnvCfg(AntEnvCfg):
    """지형·마찰 다양성 + base_height 제거 + 걸음 보상 + 뒤집힘 종료 + 발 접촉 센서."""

    events: AntFrictionEventCfg = AntFrictionEventCfg()
    rewards: AntWalkRewardsCfg = AntWalkRewardsCfg()
    terminations: AntFlipTerminationsCfg = AntFlipTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_training_terrain(self)
        # 지면 마찰 결합을 곱하기로 두어 로봇 마찰 범위(0.2~1.5)가 그대로 실효 마찰이 되게 한다
        self.scene.terrain.physics_material.friction_combine_mode = "multiply"
        remove_world_height_obs(self)
        # 발 접촉 센서. Fabric 복제를 쓰면 env_0 외의 발이 USD에 없어 센서가 못 찾으므로 USD로 복제한다
        self.scene.clone_in_fabric = False
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.foot_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3)
