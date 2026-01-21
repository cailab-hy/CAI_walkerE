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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(env: TienKungEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)


def hip_yaw_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: TienKungEnv) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


def track_upper_body_pose_from_amp(
    env: BaseEnv | TienKungEnv, std: float, include_torso: bool = True, include_arms: bool = True
) -> torch.Tensor:
    """Track upper-body joint positions against the AMP reference motion."""
    if not hasattr(env, "amp_loader_display"):
        return torch.zeros(env.num_envs, device=env.device)

    traj_len = float(env.amp_loader_display.trajectory_lens[0])
    times = (env.episode_length_buf * env.step_dt).detach().cpu().numpy()
    times = np.mod(times, traj_len)
    traj_idxs = np.zeros(env.num_envs, dtype=np.int64)
    ref_frame = env.amp_loader_display.get_full_frame_at_time_batch(traj_idxs, times)
    dof_pos_frame = ref_frame[:, 6 : 6 + env.motion_dof]

    target_chunks = []
    current_chunks = []

    if include_torso and env.motion_dof in (24, 30):
        target_chunks.append(dof_pos_frame[:, 12:16])
        current_chunks.append(env.robot.data.joint_pos[:, env.torso_head_ids])

    if include_arms:
        if env.motion_dof == 20:
            target_chunks.extend([dof_pos_frame[:, 12:16], dof_pos_frame[:, 16:20]])
            current_chunks.extend(
                [env.robot.data.joint_pos[:, env.left_arm_ids], env.robot.data.joint_pos[:, env.right_arm_ids]]
            )
        elif env.motion_dof == 24:
            target_chunks.extend([dof_pos_frame[:, 16:20], dof_pos_frame[:, 20:24]])
            current_chunks.extend(
                [env.robot.data.joint_pos[:, env.left_arm_ids], env.robot.data.joint_pos[:, env.right_arm_ids]]
            )
        elif env.motion_dof == 30:
            target_chunks.extend([dof_pos_frame[:, 16:23], dof_pos_frame[:, 23:30]])
            current_chunks.extend(
                [
                    env.robot.data.joint_pos[:, env.left_arm_full_ids],
                    env.robot.data.joint_pos[:, env.right_arm_full_ids],
                ]
            )

    if not target_chunks:
        return torch.zeros(env.num_envs, device=env.device)

    target = torch.cat(target_chunks, dim=1)
    current = torch.cat(current_chunks, dim=1)
    err = torch.sum(torch.square(current - target), dim=1)
    return torch.exp(-err / std**2)


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


def gait_feet_spd_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score


def gait_feet_frc_support_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


# ==============================================================================================
# Pick-and-Place Task Reward Functions (Object Tracking & Interaction)
# ==============================================================================================

def reach_object(
    env: BaseEnv | TienKungEnv,
    std: float = 0.5,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wrist_roll_.*_link"),
    obj_cfg: SceneEntityCfg = SceneEntityCfg("bread_box"),
) -> torch.Tensor:
    """
    Reward for the humanoid reaching towards the object.
    Encourages the end-effector to get close to the object's center.
    
    Args:
        env: The environment instance.
        std: Standard deviation for the exponential decay (controls sensitivity).
        ee_cfg: End-effector configuration (default: both wrist_roll_l_link and wrist_roll_r_link).
        obj_cfg: Object configuration (default: bread_box).
    
    Returns:
        Reward tensor with shape [num_envs], higher when end-effector is closer to object.
    """
    robot: Articulation = env.scene["robot"]
    obj: Articulation = env.scene[obj_cfg.name]
    
    # Get end-effector positions (both left and right wrists)
    ee_pos = robot.data.body_pos_w[:, ee_cfg.body_ids, :]  # [num_envs, num_ee, 3]
    obj_pos = obj.data.root_pos_w  # [num_envs, 3]
    
    # Calculate distances from each end-effector to the object
    ee_to_obj_dist = torch.norm(ee_pos - obj_pos.unsqueeze(1), dim=2)  # [num_envs, num_ee]
    
    # Use minimum distance (closer end-effector gets reward)
    min_dist = torch.min(ee_to_obj_dist, dim=1)[0]  # [num_envs]
    
    # Exponential decay reward
    reach_reward = torch.exp(-min_dist / std)
    
    return reach_reward


def contact_object(
    env: BaseEnv | TienKungEnv,
    contact_threshold: float = 5.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names="wrist_roll_.*_link"),
    min_contact_force: float = 1.0,
) -> torch.Tensor:
    """
    Reward for making contact with the object.
    Encourages the humanoid to apply sufficient contact force to the object.
    
    Args:
        env: The environment instance.
        contact_threshold: Force threshold above which contact is considered established.
        sensor_cfg: Contact sensor configuration for end-effectors.
        min_contact_force: Minimum force required to generate reward.
    
    Returns:
        Contact reward tensor with shape [num_envs], 1.0 if contact established, decaying otherwise.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact forces on end-effectors
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [num_envs, num_ee, 3]
    contact_force_magnitude = torch.norm(contact_forces, dim=2)  # [num_envs, num_ee]
    
    # Check if either hand has contact above threshold
    max_contact_force = torch.max(contact_force_magnitude, dim=1)[0]  # [num_envs]
    
    # Binary contact detection with smooth transition
    contact_reward = torch.sigmoid((max_contact_force - contact_threshold) / 5.0)
    
    return contact_reward


def grasp_stability(
    env: BaseEnv | TienKungEnv,
    alpha: float = 10.0,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wrist_roll_.*_link"),
    obj_cfg: SceneEntityCfg = SceneEntityCfg("bread_box"),
) -> torch.Tensor:
    """
    Reward for stable grasping (grasp_lock).
    Encourages maintaining a fixed relative transformation between hand and object over time.
    Penalizes slippage and transient contacts.
    
    This reward should be gated to activate only when contact is established.
    
    Args:
        env: The environment instance.
        alpha: Sensitivity coefficient for slippage penalty (higher = stricter grasp).
        ee_cfg: End-effector configuration.
        obj_cfg: Object configuration.
    
    Returns:
        Grasp stability reward tensor with shape [num_envs].
    """
    if not hasattr(env, "prev_ee_to_obj_transform"):
        env.prev_ee_to_obj_transform = None
    
    robot: Articulation = env.scene["robot"]
    obj: Articulation = env.scene[obj_cfg.name]
    
    # Get end-effector and object poses
    ee_pos = robot.data.body_pos_w[:, ee_cfg.body_ids[0], :]  # Use left wrist [num_envs, 3]
    ee_quat = robot.data.body_quat_w[:, ee_cfg.body_ids[0], :]  # [num_envs, 4]
    
    obj_pos = obj.data.root_pos_w  # [num_envs, 3]
    obj_quat = obj.data.root_quat_w  # [num_envs, 4]
    
    # Calculate relative position and rotation
    rel_pos = obj_pos - ee_pos  # [num_envs, 3]
    
    # For rotation, compute the relative quaternion: q_rel = q_ee^-1 * q_obj
    ee_quat_inv = math_utils.quat_conjugate(ee_quat)
    rel_quat = math_utils.quat_mul(ee_quat_inv, obj_quat)  # [num_envs, 4]
    
    # Extract rotation angle from quaternion: angle = 2 * acos(w)
    # For small rotations, norm of xyz component approximates the rotation vector
    rot_vec = rel_quat[:, 1:]  # [num_envs, 3] - xyz components
    rot_magnitude = torch.norm(rot_vec, dim=1)  # [num_envs]
    
    # Current relative transform (position + rotation magnitude)
    current_transform = torch.cat([rel_pos, rot_magnitude.unsqueeze(1)], dim=1)  # [num_envs, 4]
    
    if env.prev_ee_to_obj_transform is None:
        env.prev_ee_to_obj_transform = current_transform.clone().detach()
        grasp_lock_reward = torch.ones(env.num_envs, device=env.device)
    else:
        # Calculate change in relative transform (slippage)
        transform_change = torch.norm(current_transform - env.prev_ee_to_obj_transform, dim=1)  # [num_envs]
        
        # Exponential reward for small changes (stable grasp)
        grasp_lock_reward = torch.exp(-alpha * transform_change)
        
        # Update previous transform
        env.prev_ee_to_obj_transform = current_transform.clone().detach()
    
    return grasp_lock_reward


def carry_object(
    env: BaseEnv | TienKungEnv,
    std: float = 0.3,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("bread_box"),
    target_pos: torch.Tensor = None,
) -> torch.Tensor:
    """
    Reward for carrying the object towards the target destination.
    Encourages movement of the object from support0 to support1.
    
    Args:
        env: The environment instance.
        std: Standard deviation for the exponential decay.
        obj_cfg: Object configuration.
        target_pos: Target position (support1). If None, uses a predefined target.
    
    Returns:
        Carry reward tensor with shape [num_envs].
    """
    obj: Articulation = env.scene[obj_cfg.name]
    obj_pos = obj.data.root_pos_w  # [num_envs, 3]
    
    # Default target position (support1 location) - adjust based on your scene setup
    if target_pos is None:
        if hasattr(env.scene, "support1"):
            support1: Articulation = env.scene["support1"]
            target_pos = support1.data.root_pos_w  # [num_envs, 3]
        else:
            # Fallback: assume target is offset from initial position
            target_pos = torch.zeros_like(obj_pos)
            target_pos[:, 0] = obj_pos[:, 0] + 0.5  # 0.5m forward
    
    # Calculate distance from object to target
    dist_to_target = torch.norm(obj_pos - target_pos, dim=1)  # [num_envs]
    
    # Exponential decay reward (higher reward when object is closer to target)
    carry_reward = torch.exp(-dist_to_target / std)
    
    return carry_reward


def place_object(
    env: BaseEnv | TienKungEnv,
    position_tolerance: float = 0.05,
    height_tolerance: float = 0.05,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("bread_box"),
    target_pos: torch.Tensor = None,
) -> torch.Tensor:
    """
    Reward for placing the object at the target destination (support1).
    Encourages precise placement with the object at rest.
    
    Args:
        env: The environment instance.
        position_tolerance: XY-plane position tolerance for successful placement.
        height_tolerance: Height tolerance for successful placement (Z-axis).
        obj_cfg: Object configuration.
        target_pos: Target position (support1). If None, uses a predefined target.
    
    Returns:
        Place reward tensor with shape [num_envs].
    """
    obj: Articulation = env.scene[obj_cfg.name]
    obj_pos = obj.data.root_pos_w  # [num_envs, 3]
    obj_vel = obj.data.root_lin_vel_w  # [num_envs, 3]
    
    # Default target position
    if target_pos is None:
        if hasattr(env.scene, "support1"):
            support1: Articulation = env.scene["support1"]
            target_pos = support1.data.root_pos_w  # [num_envs, 3]
        else:
            target_pos = torch.zeros_like(obj_pos)
            target_pos[:, 0] = obj_pos[:, 0] + 0.5
    
    # Check position error (XY and Z separately for finer control)
    xy_error = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=1)  # [num_envs]
    z_error = torch.abs(obj_pos[:, 2] - target_pos[:, 2])  # [num_envs]
    
    # Check velocity (object should be at rest)
    obj_speed = torch.norm(obj_vel, dim=1)  # [num_envs]
    
    # Reward for being within tolerance and at rest
    xy_in_tolerance = (xy_error <= position_tolerance).float()
    z_in_tolerance = (z_error <= height_tolerance).float()
    at_rest = (obj_speed <= 0.1).float()  # Velocity threshold for "at rest"
    
    place_reward = xy_in_tolerance * z_in_tolerance * at_rest
    
    # Bonus reward for being closer to the target within tolerance
    # Smooth transition from 0 to 1 as we approach the target
    xy_bonus = torch.exp(-xy_error / (position_tolerance + 1e-6))
    z_bonus = torch.exp(-z_error / (height_tolerance + 1e-6))
    velocity_bonus = torch.exp(-obj_speed)
    
    place_reward = place_reward + (1 - place_reward) * xy_bonus * z_bonus * velocity_bonus * 0.5
    
    return place_reward
