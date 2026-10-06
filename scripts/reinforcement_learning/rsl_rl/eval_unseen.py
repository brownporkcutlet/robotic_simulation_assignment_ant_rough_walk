# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint for one episode per environment and append the statistics to a JSON-lines file.

Besides the episode reward and length (as in play_one_episode.py) this records, per environment:
- the distance covered towards the target,
- whether the episode ended by falling (termination) or by the time limit,
- the torso up-projection right before a fall (to separate real falls from height-only terminations),
- how far the policy observations are from the training statistics (z-score under the observation normalizer).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate one episode per environment and save the statistics.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--label", type=str, default="", help="Label stored with the results, e.g. 'step1|Boxes'.")
parser.add_argument(
    "--output", type=str, default="logs/eval/results.jsonl", help="JSON-lines file the summary is appended to."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch

from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# torso up-projection above which a termination is counted as "upright" (the robot did not tip over)
UPRIGHT_THRESHOLD = 0.8


def _stats(x: torch.Tensor) -> dict:
    x = x.to(dtype=torch.float64)
    return {"mean": x.mean().item(), "std": x.std(unbiased=False).item()}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate one episode per environment."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.checkpoint is None:
        raise ValueError("Please pass the checkpoint to evaluate with --checkpoint.")
    resume_path = retrieve_file_path(args_cli.checkpoint)

    # target of the progress reward (Ant: a far-away point in +x)
    target_xy = (1000.0, 0.0)
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is not None and hasattr(rewards_cfg, "progress"):
        target_xy = tuple(rewards_cfg.progress.params["target_pos"][:2])

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # observation normalizer of the acting policy (only present if the model was trained with normalization)
    normalizer = getattr(runner.alg.policy, "actor_obs_normalizer", None)
    if not isinstance(normalizer, EmpiricalNormalization):
        normalizer = None

    # index of the world-frame base height in the policy observation (first term of the Ant), if it is observed
    height_idx = None
    obs_manager = getattr(env.unwrapped, "observation_manager", None)
    if obs_manager is not None and obs_manager.active_terms["policy"][:1] == ["base_height"]:
        height_idx = 0

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = env.num_envs
    target = torch.tensor(target_xy, device=device)

    obs = env.get_observations()
    # the z-scores are computed on the "policy" group; skip them if the policy also reads other groups (teacher)
    if normalizer is not None and normalizer._mean.shape[-1] != obs["policy"].shape[-1]:
        normalizer = None
    start_xy = robot.data.root_pos_w[:, :2].clone()
    # state right before the last step; the state returned by a terminal step is already the reset state
    last_xy = start_xy.clone()
    last_up = -robot.data.projected_gravity_b[:, 2].clone()
    last_z = robot.data.root_pos_w[:, 2].clone()
    end_xy = start_xy.clone()
    end_up = last_up.clone()
    end_z = last_z.clone()

    episode_reward = torch.zeros(num_envs, dtype=torch.float64, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # observation shift: z-score of the policy observations under the training statistics
    z_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    z_max = torch.zeros(num_envs, device=device)
    height_z_max = torch.zeros(num_envs, device=device)
    # gait quality (per-step averages): action changes, vertical bouncing, torso rocking, tilt, mechanical power
    prev_actions = torch.zeros(num_envs, env.num_actions, device=device)
    action_rate_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    lin_vel_z_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    ang_vel_xy_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    tilt_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    power_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    state_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(env.max_episode_length + 1):
            active = ~finished
            last_xy[active] = robot.data.root_pos_w[active, :2]
            last_up[active] = -robot.data.projected_gravity_b[active, 2]
            last_z[active] = robot.data.root_pos_w[active, 2]
            if normalizer is not None:
                policy_obs = obs["policy"]
                z = ((policy_obs - normalizer._mean) / (normalizer._std + normalizer.eps)).abs()
                z_sum[active] += z[active].mean(dim=-1)
                z_max[active] = torch.maximum(z_max[active], z[active].max(dim=-1).values)
                if height_idx is not None:
                    height_z_max[active] = torch.maximum(height_z_max[active], z[active, height_idx])

            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            episode_reward[active] += rewards[active]
            episode_steps[active] += 1
            just_done = active & dones.bool()
            # gait quality; state-based terms skip environments whose state was just reset
            action_rate_sum[active] += torch.sum(torch.square(actions - prev_actions), dim=-1)[active]
            prev_actions = actions.clone()
            running = active & ~just_done
            lin_vel_z_sum[running] += torch.square(robot.data.root_lin_vel_b[running, 2])
            ang_vel_xy_sum[running] += torch.sum(torch.square(robot.data.root_ang_vel_b[running, :2]), dim=-1)
            tilt_sum[running] += torch.sum(torch.square(robot.data.projected_gravity_b[running, :2]), dim=-1)
            # commanded joint torque x joint velocity (applied_torque stays zero for implicit actuators)
            power_sum[running] += torch.sum(
                torch.abs(robot.data.joint_effort_target[running] * robot.data.joint_vel[running]), dim=-1
            )
            state_steps[running] += 1
            time_outs = extras["time_outs"].bool()
            fell[just_done] = ~time_outs[just_done]
            end_xy[just_done] = last_xy[just_done]
            end_up[just_done] = last_up[just_done]
            end_z[just_done] = last_z[just_done]
            finished |= just_done
            if finished.all():
                break

    # environments that did not finish (should not happen with a time limit) are measured where they are
    unfinished = ~finished
    end_xy[unfinished] = robot.data.root_pos_w[unfinished, :2]
    distance = torch.norm(target - start_xy, dim=-1) - torch.norm(target - end_xy, dim=-1)

    num_falls = int(fell.sum().item())
    upright_falls = int((fell & (end_up > UPRIGHT_THRESHOLD)).sum().item())
    result = {
        "label": args_cli.label,
        "task": args_cli.task,
        "checkpoint": resume_path,
        "num_envs": num_envs,
        "seed": agent_cfg.seed,
        "overrides": hydra_args,
        "completed": int(finished.sum().item()),
        "reward": _stats(episode_reward),
        "steps": _stats(episode_steps),
        "distance_m": _stats(distance),
        "fall_rate": num_falls / num_envs,
        "upright_fall_share": (upright_falls / num_falls) if num_falls > 0 else 0.0,
        "fall_end_height": _stats(end_z[fell]) if num_falls > 0 else None,
        "obs_z_mean": _stats(z_sum / episode_steps.clamp(min=1)) if normalizer is not None else None,
        "obs_z_max": _stats(z_max) if normalizer is not None else None,
        "height_z_max": _stats(height_z_max) if (normalizer is not None and height_idx is not None) else None,
        "gait": {
            "action_rate": _stats(action_rate_sum / episode_steps.clamp(min=1)),
            "lin_vel_z_sq": _stats(lin_vel_z_sum / state_steps.clamp(min=1)),
            "ang_vel_xy_sq": _stats(ang_vel_xy_sum / state_steps.clamp(min=1)),
            "tilt_sq": _stats(tilt_sum / state_steps.clamp(min=1)),
            "power_w": _stats(power_sum / state_steps.clamp(min=1)),
        },
    }
    print(f"[RESULT] {json.dumps(result)}")
    out_dir = os.path.dirname(args_cli.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args_cli.output, "a") as f:
        f.write(json.dumps(result) + "\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
