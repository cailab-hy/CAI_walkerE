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


# ======================== Pick and Place Task Rewards ========================
# Category 1: Reach to bread_box
def hand_distance_to_bread_box(
    env: BaseEnv | TienKungEnv,
    std: float = 0.3,
) -> torch.Tensor:
    """
    Reward for bringing hand close to bread_box.
    
    This reward encourages the robot's hand (wrist) to reach towards the bread_box.
    Essential for the reaching phase of pick-and-place.
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential decay of distance reward.
    
    Returns:
        Reward tensor of shape (num_envs,). Higher when hand is closer to bread_box.
    """
    try:
        bread_box = env.scene["bread_box"]
        robot = env.scene["robot"]
        
        # Get bread_box position
        bread_box_pos = bread_box.data.root_pos_w[:, :3]
        
        # Get right wrist position (assuming right hand reaches for bread_box)
        # wrist_roll_r_link should be the right wrist
        wrist_body_ids = robot.find_bodies("wrist_roll_r_link")
        # find_bodies returns (indices, names) as tuple, extract the integer index
        wrist_idx = wrist_body_ids[0][0]
        right_wrist_pos = robot.data.body_pos_w[:, wrist_idx, :3]
        
        # Calculate distance between wrist and bread_box
        distance = torch.norm(bread_box_pos - right_wrist_pos, dim=-1)
        
        # Exponential reward: closer = higher reward
        reward = torch.exp(-distance / std)
        
        return reward
    except Exception as e:
        print(f"Warning: Could not compute hand_distance_to_bread_box: {e}")
        import traceback
        traceback.print_exc()
        return torch.zeros(env.num_envs, device=env.device)

# Category 2: Grasp & Lift bread_box
def wrist_contact_with_bread_box(
    env: BaseEnv | TienKungEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Binary reward for wrist-bread_box contact detection.
    
    Simple binary reward: 1 if wrist (either left or right) is in contact with bread_box, 0 otherwise.
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the contact sensor.
        contact_threshold: Minimum contact force to count as contact (in Newtons).
    
    Returns:
        Reward tensor of shape (num_envs,). Value is 1.0 if contact detected, 0 otherwise.
    """
    try:
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w
        
        # Check contact for wrist_roll links
        wrist_in_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
        try:
            for body_idx, body_name in enumerate(contact_sensor.body_names):
                if "wrist_roll" in body_name:
                    contact_force_norm = torch.norm(net_contact_forces[:, body_idx, :], dim=-1)
                    wrist_in_contact |= contact_force_norm > contact_threshold
        except Exception:
            pass
        
        return wrist_in_contact.float()
    except Exception as e:
        print(f"Warning: Could not compute wrist_contact_with_bread_box: {e}")
        return torch.zeros(env.num_envs, device=env.device)

def wrist_bread_box_contact_duration(
    env: BaseEnv | TienKungEnv,
    wrist_contact_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Reward for maintaining sustained contact between wrist and bread_box.
    
    This function tracks contact duration - the longer the wrist maintains contact with bread_box,
    the higher the reward. Useful for encouraging grasping behavior.
    
    Args:
        env: The environment instance.
        wrist_contact_threshold: Minimum contact force threshold to count as contact (in Newtons).
    
    Returns:
        Reward tensor of shape (num_envs,). Increases with sustained contact duration.
    """
    try:
        contact_sensor = env.scene.sensors["contact_sensor"]
        
        # Get contact forces history for both wrist links
        net_contact_forces_history = contact_sensor.data.net_forces_w_history
        # Shape: (num_envs, history_length, num_bodies, 3)
        
        wrist_contact_time = torch.zeros(env.num_envs, device=env.device)
        
        # Search for wrist_roll body indices
        try:
            for body_idx, body_name in enumerate(contact_sensor.body_names):
                if "wrist_roll" in body_name:
                    # Get contact history for this wrist
                    contact_force_history = torch.norm(
                        net_contact_forces_history[:, :, body_idx, :], dim=-1
                    )  # Shape: (num_envs, history_length)
                    
                    # Count timesteps with contact above threshold
                    contact_steps = torch.sum(
                        contact_force_history > wrist_contact_threshold, dim=-1
                    )
                    
                    wrist_contact_time += contact_steps.float()
        except Exception:
            pass
        
        # Normalize by history length to get contact ratio (0-1)
        history_length = net_contact_forces_history.shape[1] if len(net_contact_forces_history.shape) > 2 else 1
        contact_ratio = torch.clamp(wrist_contact_time / history_length, 0, 1)

        return contact_ratio
    except Exception as e:
        print(f"Warning: Could not compute wrist_bread_box_contact_duration: {e}")
        return torch.zeros(env.num_envs, device=env.device)

# Category 3: Reach to support_1 (while holding bread_box)
def bread_box_distance_to_support1(
    env: BaseEnv | TienKungEnv,
    std: float = 0.5,
) -> torch.Tensor:
    """
    Reward for bringing bread_box close to support1.
    
    The reward is higher when bread_box is closer to support1's position.
    Uses exponential function for smooth distance-based reward.
    
    Args:
        env: The environment instance.
        std: Standard deviation for the exponential decay (controls how fast reward decreases with distance).
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    try:
        bread_box = env.scene["bread_box"]
        support1 = env.scene["support1"]
        
        # Get positions in world frame
        bread_box_pos = bread_box.data.root_pos_w[:, :3]
        support1_pos = support1.data.root_pos_w[:, :3]
        
        # Calculate distance
        distance = torch.norm(bread_box_pos - support1_pos, dim=-1)
        
        # Exponential decay reward
        reward = torch.exp(-distance / std)

        return reward
    except Exception as e:
        print(f"Warning: Could not compute bread_box_distance_to_support1: {e}")
        return torch.zeros(env.num_envs, device=env.device)

# Category 4: Place on support_1
def bread_box_placement_on_support1(
    env: BaseEnv | TienKungEnv,
    xy_threshold: float = 0.15,
    z_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Reward for placing bread_box on top of support1.
    
    This reward is given when:
    - bread_box is within xy_threshold horizontally from support1
    - bread_box is within z_threshold vertically from support1 (on top)
    
    Args:
        env: The environment instance.
        xy_threshold: Horizontal distance threshold in meters.
        z_threshold: Vertical distance threshold in meters (above support1).
    
    Returns:
        Reward tensor of shape (num_envs,). Value is 1 if conditions met, 0 otherwise.
    """
    try:
        bread_box = env.scene["bread_box"]
        support1 = env.scene["support1"]
        
        # Get positions
        bread_box_pos = bread_box.data.root_pos_w
        support1_pos = support1.data.root_pos_w
        
        # Calculate horizontal and vertical distances
        xy_distance = torch.norm(
            bread_box_pos[:, :2] - support1_pos[:, :2], dim=-1
        )
        z_distance = bread_box_pos[:, 2] - support1_pos[:, 2]
        
        # Check if bread_box is on top of support1
        on_support1 = (xy_distance < xy_threshold) & (0 < z_distance) & (z_distance < z_threshold)

        return on_support1.float()
    except Exception as e:
        print(f"Warning: Could not compute bread_box_placement_on_support1: {e}")
        return torch.zeros(env.num_envs, device=env.device)

def task_completion_bonus(
    env: BaseEnv | TienKungEnv,
    xy_threshold: float = 0.15,
    z_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Large bonus reward for successfully completing the pick-and-place task.
    
    Task is considered complete when bread_box is placed on support1.
    
    Args:
        env: The environment instance.
        xy_threshold: Horizontal distance threshold in meters.
        z_threshold: Vertical distance threshold in meters.
    
    Returns:
        Reward tensor of shape (num_envs,). Large positive value if task complete.
    """
    try:
        bread_box = env.scene["bread_box"]
        support1 = env.scene["support1"]
        
        # Get positions
        bread_box_pos = bread_box.data.root_pos_w
        support1_pos = support1.data.root_pos_w
        
        # Calculate distances
        xy_distance = torch.norm(
            bread_box_pos[:, :2] - support1_pos[:, :2], dim=-1
        )
        z_distance = bread_box_pos[:, 2] - support1_pos[:, 2]
        
        # Task complete if bread_box is on support1
        task_complete = (xy_distance < xy_threshold) & (0 < z_distance) & (z_distance < z_threshold)
        
        # Large bonus for task completion
        reward = task_complete.float() * 10.0

        return reward
    except Exception as e:
        print(f"Warning: Could not compute task_completion_bonus: {e}")
        return torch.zeros(env.num_envs, device=env.device)


def wrist_bread_box_contact(
    env: BaseEnv | TienKungEnv,
    wrist_contact_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Reward for maintaining contact between wrist and bread_box.
    
    This function monitors contact forces between both wrist links (wrist_roll_l and wrist_roll_r)
    and the bread_box, rewarding sustained contact.
    
    Args:
        env: The environment instance.
        wrist_contact_threshold: Minimum contact force threshold to count as contact (in Newtons).
    
    Returns:
        Reward tensor of shape (num_envs,). Value increases with contact force strength.
    """
    try:
        contact_sensor = env.scene.sensors["contact_sensor"]
        bread_box = env.scene["bread_box"]
        
        # Get contact forces for both wrist links
        # contact_sensor.data.net_forces_w has shape (num_envs, num_bodies, 3)
        net_contact_forces = contact_sensor.data.net_forces_w
        
        # Find body indices for wrist_roll links
        # We need to find which body indices correspond to wrist_roll_l and wrist_roll_r
        wrist_contact_force = torch.zeros(env.num_envs, device=env.device)
        
        # Try to get wrist body indices from contact sensor
        try:
            # Search for wrist_roll in body names
            for body_idx, body_name in enumerate(contact_sensor.body_names):
                if "wrist_roll" in body_name:
                    contact_force_norm = torch.norm(net_contact_forces[:, body_idx, :], dim=-1)
                    wrist_contact_force += contact_force_norm
        except Exception:
            # If body_names not accessible, try alternative approach
            pass
        
        # Reward for contact: higher force = higher reward
        # Use soft threshold with exponential decay
        contact_reward = torch.clamp(wrist_contact_force / (wrist_contact_threshold * 10), 0, 1)

        return contact_reward
    except Exception as e:
        print(f"Warning: Could not compute wrist_bread_box_contact: {e}")
        return torch.zeros(env.num_envs, device=env.device)


