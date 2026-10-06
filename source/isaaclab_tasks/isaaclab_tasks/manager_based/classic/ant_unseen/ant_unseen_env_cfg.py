"""Held-out "unseen" evaluation environments for the Ant task.

These environments are for EVALUATION ONLY. Never train on them and never add their terrain types
(waves, boxes, stairs) to the training terrains, otherwise the generalization numbers are meaningless.

Only the terrain shape differs from Isaac-Ant-v0. Actions, rewards, terminations and episode length are inherited
unchanged from the original ``AntEnvCfg``. The world-frame base height is removed from the observations, as in the
training task Isaac-Ant-Rough-Walk-v0 (59 dimensions).

Friction is changed from the command line, e.g.::

    env.scene.terrain.physics_material.friction_combine_mode=multiply \
    env.scene.terrain.physics_material.static_friction=0.2 env.scene.terrain.physics_material.dynamic_friction=0.2
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant_rough.ant_rough_env_cfg import remove_world_height_obs

# Default number of robots for evaluation (override with --num_envs).
EVAL_NUM_ENVS = 256


def _unseen_sub_terrains(names: list[str]) -> dict:
    """Creates fresh sub-terrain configs reserved for evaluation.

    Heights are kept small because the Ant is small (spawn height 0.5 m) and the original termination
    uses the world-frame torso height (< 0.31 m). Waves and boxes also go below z = 0.
    """
    all_sub_terrains = {
        # sinusoidal ground, height in [-amplitude, +amplitude], wavelength 2 m
        "waves": terrain_gen.HfWaveTerrainCfg(amplitude_range=(0.02, 0.08), num_waves=4),
        # grid of boxes, each cell shifted by U(-h, +h)
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=0.45, grid_height_range=(0.02, 0.06), platform_width=2.0
        ),
        # pyramid of low steps going up towards the tile center
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            step_height_range=(0.02, 0.06), step_width=0.5, platform_width=2.0, border_width=0.5, holes=False
        ),
    }
    return {name: all_sub_terrains[name] for name in names}


def _apply_unseen_terrain(cfg: AntEnvCfg, sub_terrain_names: list[str]):
    """Replaces the flat plane of Isaac-Ant-v0 with a fixed, reproducible terrain."""
    # keep the original prim path, collision group and physics material; only the shape changes
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        # fixed seed: every evaluation sees exactly the same terrain
        seed=0,
        size=(8.0, 8.0),
        # rows run along +x (the walking direction): 30 rows (236 m) + 20 m border of ground ahead
        num_rows=30,
        num_cols=8,
        # flat margin around the terrain; beyond it there is no ground
        border_width=20.0,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=False,
        use_cache=False,
        sub_terrains=_unseen_sub_terrains(sub_terrain_names),
    )
    # spawn every robot on the first row so that the whole terrain lies ahead in +x
    cfg.scene.terrain.max_init_terrain_level = 0


@configclass
class AntUnseenFlatEnvCfg(AntEnvCfg):
    """Flat plane as in Isaac-Ant-v0. Used to isolate the effect of friction."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = EVAL_NUM_ENVS
        remove_world_height_obs(self)


@configclass
class AntUnseenWavesEnvCfg(AntUnseenFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_unseen_terrain(self, ["waves"])


@configclass
class AntUnseenBoxesEnvCfg(AntUnseenFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_unseen_terrain(self, ["boxes"])


@configclass
class AntUnseenStairsEnvCfg(AntUnseenFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_unseen_terrain(self, ["stairs"])


@configclass
class AntUnseenMixedEnvCfg(AntUnseenFlatEnvCfg):
    """Every tile is randomly one of the unseen terrain types (fixed by the seed)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_unseen_terrain(self, ["waves", "boxes", "stairs"])
