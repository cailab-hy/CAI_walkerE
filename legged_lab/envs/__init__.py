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

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg

# --- Tienkung Pro ----------------------------------------------------------------------------------
from legged_lab.envs.tienkung_pro.tienkung_env import TienKungEnv as TienKungProEnv

from legged_lab.envs.tienkung_pro.pick_cfg import TienKungPickAgentCfg as TienKungProPickAgentCfg
from legged_lab.envs.tienkung_pro.pick_cfg import TienKungPickFlatEnvCfg as TienKungProPickFlatEnvCfg
from legged_lab.envs.tienkung_pro.walk_cfg import TienKungWalkAgentCfg as TienKungProWalkAgentCfg
from legged_lab.envs.tienkung_pro.walk_cfg import TienKungWalkFlatEnvCfg as TienKungProWalkFlatEnvCfg
from legged_lab.envs.tienkung_pro.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg as TienKungProWalkWithSensorAgentCfg,
)
from legged_lab.envs.tienkung_pro.walk_with_sensor_cfg import (
    TienKungWalkWithSensorFlatEnvCfg as TienKungProWalkWithSensorFlatEnvCfg,
)
# ---------------------------------------------------------------------------------------------------
from legged_lab.utils.task_registry import task_registry

# --- Tienkung Pro ----------------------------------------------------------------------------------
task_registry.register("pro_pick", TienKungProEnv, TienKungProPickFlatEnvCfg(), TienKungProPickAgentCfg())
task_registry.register("pro_walk", TienKungProEnv, TienKungProWalkFlatEnvCfg(), TienKungProWalkAgentCfg())
task_registry.register(
    "pro_walk_with_sensor",
    TienKungProEnv,
    TienKungProWalkWithSensorFlatEnvCfg(),
    TienKungProWalkWithSensorAgentCfg(),
)
# ---------------------------------------------------------------------------------------------------
