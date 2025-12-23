"""Configuration for the TienKung WalkerE robot.

This is an initial draft mirroring the structure of ``tienkung2_lite``.
Update joint names, limits, gains, and USD paths to match the WalkerE model.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


TIENKUNG_WALKER_E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/tienkung_walkerE/usd/tiangong2.0_pro_with_hands.usd",
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
        # Replace with WalkerE joint names and a stable nominal pose.
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
    # Replace joint regexes and gains with WalkerE specs.
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
    },
)
