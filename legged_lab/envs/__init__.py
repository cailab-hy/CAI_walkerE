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
from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung_walkerE.run_cfg import TienKungRunAgentCfg as TienKungWalkerERunAgentCfg
from legged_lab.envs.tienkung_walkerE.run_cfg import TienKungRunFlatEnvCfg as TienKungWalkerERunFlatEnvCfg
from legged_lab.envs.tienkung_walkerE.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg as TienKungWalkerERunWithSensorAgentCfg,
)
from legged_lab.envs.tienkung_walkerE.run_with_sensor_cfg import (
    TienKungRunWithSensorFlatEnvCfg as TienKungWalkerERunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung_walkerE.tienkung_env import TienKungEnv as TienKungWalkerEEnv
from legged_lab.envs.tienkung_walkerE.walk_cfg import TienKungWalkAgentCfg as TienKungWalkerEWalkAgentCfg
from legged_lab.envs.tienkung_walkerE.walk_cfg import TienKungWalkFlatEnvCfg as TienKungWalkerEWalkFlatEnvCfg
from legged_lab.envs.tienkung_walkerE.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg as TienKungWalkerEWalkWithSensorAgentCfg,
)
from legged_lab.envs.tienkung_walkerE.walk_with_sensor_cfg import (
    TienKungWalkWithSensorFlatEnvCfg as TienKungWalkerEWalkWithSensorFlatEnvCfg,
)
# --- Tienkung Pro ----------------------------------------------------------------------------------
from legged_lab.envs.tienkung_pro.run_cfg import TienKungRunAgentCfg as TienKungProRunAgentCfg
from legged_lab.envs.tienkung_pro.run_cfg import TienKungRunFlatEnvCfg as TienKungProRunFlatEnvCfg
from legged_lab.envs.tienkung_pro.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg as TienKungProRunWithSensorAgentCfg,
)
from legged_lab.envs.tienkung_pro.run_with_sensor_cfg import (
    TienKungRunWithSensorFlatEnvCfg as TienKungProRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung_pro.tienkung_env import TienKungEnv as TienKungProEnv

from legged_lab.envs.tienkung_pro.easy_cfg import TienKungEasyAgentCfg as TienKungProEasyAgentCfg
from legged_lab.envs.tienkung_pro.easy_cfg import TienKungEasyFlatEnvCfg as TienKungProEasyFlatEnvCfg
from legged_lab.envs.tienkung_pro.jab_cfg import TienKungJabAgentCfg as TienKungProJabAgentCfg
from legged_lab.envs.tienkung_pro.jab_cfg import TienKungJabFlatEnvCfg as TienKungProJabFlatEnvCfg
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

task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)
task_registry.register(
    "walkerE_walk", TienKungWalkerEEnv, TienKungWalkerEWalkFlatEnvCfg(), TienKungWalkerEWalkAgentCfg()
)
task_registry.register(
    "walkerE_run", TienKungWalkerEEnv, TienKungWalkerERunFlatEnvCfg(), TienKungWalkerERunAgentCfg()
)
task_registry.register(
    "walkerE_walk_with_sensor",
    TienKungWalkerEEnv,
    TienKungWalkerEWalkWithSensorFlatEnvCfg(),
    TienKungWalkerEWalkWithSensorAgentCfg(),
)
task_registry.register(
    "walkerE_run_with_sensor",
    TienKungWalkerEEnv,
    TienKungWalkerERunWithSensorFlatEnvCfg(),
    TienKungWalkerERunWithSensorAgentCfg(),
)

# --- Tienkung Pro ----------------------------------------------------------------------------------
task_registry.register("pro_walk", TienKungProEnv, TienKungProWalkFlatEnvCfg(), TienKungProWalkAgentCfg())
task_registry.register("pro_run", TienKungProEnv, TienKungProRunFlatEnvCfg(), TienKungProRunAgentCfg())
task_registry.register("pro_jab", TienKungProEnv, TienKungProJabFlatEnvCfg(), TienKungProJabAgentCfg())
task_registry.register("pro_easy", TienKungProEnv, TienKungProEasyFlatEnvCfg(), TienKungProEasyAgentCfg())
task_registry.register(
    "pro_walk_with_sensor",
    TienKungProEnv,
    TienKungProWalkWithSensorFlatEnvCfg(),
    TienKungProWalkWithSensorAgentCfg(),
)
task_registry.register(
    "pro_run_with_sensor",
    TienKungProEnv,
    TienKungProRunWithSensorFlatEnvCfg(),
    TienKungProRunWithSensorAgentCfg(),
)
# ---------------------------------------------------------------------------------------------------
