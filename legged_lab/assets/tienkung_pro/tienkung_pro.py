"""Configuration for the TienKung Pro robot.

This mirrors the existing WalkerE config and should be adjusted to match
the Pro model's joint names, limits, and gains.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


TIENKUNG_PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/tienkung_pro/usd/tiangong2.0_pro.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "hip_roll_l_joint": 0.0,
            "hip_pitch_l_joint": -0.5,
            "hip_yaw_l_joint": 0.0,
            "knee_pitch_l_joint": 1.0,
            "ankle_pitch_l_joint": -0.5,
            "ankle_roll_l_joint": -0.0,
            "hip_roll_r_joint": -0.0,
            "hip_pitch_r_joint": -0.5,
            "hip_yaw_r_joint": 0.0,
            "knee_pitch_r_joint": 1.0,
            "ankle_pitch_r_joint": -0.5,
            "ankle_roll_r_joint": 0.0,
            "shoulder_pitch_l_joint": 0.0,
            "shoulder_roll_l_joint": 0.1,
            "shoulder_yaw_l_joint": -0.0,
            "elbow_pitch_l_joint": -0.3,
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": -0.1,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_roll_.*_joint",
                "hip_pitch_.*_joint",
                "hip_yaw_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_roll_.*_joint": 180,
                "hip_pitch_.*_joint": 300,
                "hip_yaw_.*_joint": 180,
                "knee_pitch_.*_joint": 300,
            },
            velocity_limit_sim={
                "hip_roll_.*_joint": 15.6,
                "hip_pitch_.*_joint": 15.6,
                "hip_yaw_.*_joint": 15.6,
                "knee_pitch_.*_joint": 15.6,
            },
            stiffness={
                "hip_roll_.*_joint": 700,
                "hip_pitch_.*_joint": 700,
                "hip_yaw_.*_joint": 500,
                "knee_pitch_.*_joint": 700,
            },
            damping={
                "hip_roll_.*_joint": 10,
                "hip_pitch_.*_joint": 10,
                "hip_yaw_.*_joint": 5,
                "knee_pitch_.*_joint": 10,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "ankle_pitch_.*_joint",
                "ankle_roll_.*_joint",
            ],
            effort_limit_sim={
                "ankle_pitch_.*_joint": 60,
                "ankle_roll_.*_joint": 30,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 12.8,
                "ankle_roll_.*_joint": 7.8,
            },
            stiffness={
                "ankle_pitch_.*_joint": 30,
                "ankle_roll_.*_joint": 16.8,
            },
            damping={
                "ankle_pitch_.*_joint": 2.5,
                "ankle_roll_.*_joint": 1.4,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_pitch_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 52.5,
                "shoulder_roll_.*_joint": 52.5,
                "shoulder_yaw_.*_joint": 52.5,
                "elbow_pitch_.*_joint": 52.5,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 14.1,
                "shoulder_roll_.*_joint": 14.1,
                "shoulder_yaw_.*_joint": 14.1,
                "elbow_pitch_.*_joint": 14.1,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 60,
                "shoulder_roll_.*_joint": 20,
                "shoulder_yaw_.*_joint": 10,
                "elbow_pitch_.*_joint": 10,
            },
            damping={
                "shoulder_pitch_.*_joint": 3,
                "shoulder_roll_.*_joint": 1.5,
                "shoulder_yaw_.*_joint": 1,
                "elbow_pitch_.*_joint": 1,
            },
        ),
        "arms_extra": ImplicitActuatorCfg(
            joint_names_expr=[
                "elbow_yaw_.*_joint",
                "wrist_pitch_.*_joint",
                "wrist_roll_.*_joint",
            ],
            effort_limit_sim={
                "elbow_yaw_.*_joint": 40,
                "wrist_pitch_.*_joint": 20,
                "wrist_roll_.*_joint": 20,
            },
            velocity_limit_sim={
                "elbow_yaw_.*_joint": 10.0,
                "wrist_pitch_.*_joint": 8.0,
                "wrist_roll_.*_joint": 8.0,
            },
            stiffness={
                "elbow_yaw_.*_joint": 15,
                "wrist_pitch_.*_joint": 8,
                "wrist_roll_.*_joint": 8,
            },
            damping={
                "elbow_yaw_.*_joint": 1,
                "wrist_pitch_.*_joint": 0.5,
                "wrist_roll_.*_joint": 0.5,
            },
        ),
        "torso_head": ImplicitActuatorCfg(
            joint_names_expr=[
                "body_yaw_joint",
                "head_yaw_joint",
                "head_pitch_joint",
                "head_roll_joint",
            ],
            effort_limit_sim={
                "body_yaw_joint": 50,
                "head_yaw_joint": 20,
                "head_pitch_joint": 20,
                "head_roll_joint": 20,
            },
            velocity_limit_sim={
                "body_yaw_joint": 6.0,
                "head_yaw_joint": 6.0,
                "head_pitch_joint": 6.0,
                "head_roll_joint": 6.0,
            },
            stiffness={
                "body_yaw_joint": 20,
                "head_yaw_joint": 10,
                "head_pitch_joint": 10,
                "head_roll_joint": 10,
            },
            damping={
                "body_yaw_joint": 1,
                "head_yaw_joint": 0.5,
                "head_pitch_joint": 0.5,
                "head_roll_joint": 0.5,
            },
        ),
    },
)
