#!/usr/bin/env python3
# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
from typing import Iterable

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip


def _dump_names(label: str, names: Iterable[str]) -> None:
    names_list = list(names)
    print(f"\n[INFO] {label} (count={len(names_list)}):")
    for name in names_list:
        print(name)


def _get_name_list(robot, attr: str):
    if hasattr(robot, "data") and hasattr(robot.data, attr):
        return getattr(robot.data, attr)
    if hasattr(robot, attr):
        return getattr(robot, attr)
    return None


parser = argparse.ArgumentParser(description="Smoke test for WalkerE assets (USD load + joint/body names).")
parser.add_argument("--task", type=str, default="walkerE_walk", help="Task name to load.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Start camera rendering when sensor tasks are used.
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa: F401,F403,E402


def main() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    robot = env.robot
    joint_names = _get_name_list(robot, "joint_names")
    body_names = _get_name_list(robot, "body_names")

    if joint_names is None or body_names is None:
        raise RuntimeError("Unable to access joint/body names from the robot instance.")

    _dump_names("Joint names", joint_names)
    _dump_names("Body names", body_names)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
