# TienKung Pro CAN ID to URDF/MJCF joint mapping

This table maps the provided CAN motor IDs to the URDF/MJCF joint names used in the
simulation assets.

Note: The SDK label "Wrist Yaw" corresponds to `elbow_yaw_*_joint` in URDF/MJCF
(forearm rotation between elbow_pitch and wrist_pitch).

## Head and waist

| CAN ID | SDK joint label | URDF/MJCF joint_name |
| ---: | --- | --- |
| 1 | Head Roll | head_roll_joint |
| 2 | Head Pitch | head_pitch_joint |
| 3 | Head Yaw | head_yaw_joint |
| 31 | Waist Yaw | body_yaw_joint |

## Left arm

| CAN ID | SDK joint label | URDF/MJCF joint_name |
| ---: | --- | --- |
| 11 | Left Shoulder Pitch | shoulder_pitch_l_joint |
| 12 | Left Shoulder Roll | shoulder_roll_l_joint |
| 13 | Left Shoulder Yaw | shoulder_yaw_l_joint |
| 14 | Left Elbow Pitch | elbow_pitch_l_joint |
| 15 | Left Wrist Yaw | elbow_yaw_l_joint |
| 16 | Left Wrist Pitch | wrist_pitch_l_joint |
| 17 | Left Wrist Roll | wrist_roll_l_joint |

## Right arm

| CAN ID | SDK joint label | URDF/MJCF joint_name |
| ---: | --- | --- |
| 21 | Right Shoulder Pitch | shoulder_pitch_r_joint |
| 22 | Right Shoulder Roll | shoulder_roll_r_joint |
| 23 | Right Shoulder Yaw | shoulder_yaw_r_joint |
| 24 | Right Elbow Pitch | elbow_pitch_r_joint |
| 25 | Right Wrist Yaw | elbow_yaw_r_joint |
| 26 | Right Wrist Pitch | wrist_pitch_r_joint |
| 27 | Right Wrist Roll | wrist_roll_r_joint |

## Legs

| SDK joint label | Left CAN ID | Right CAN ID | URDF/MJCF joint_name (L/R) |
| --- | ---: | ---: | --- |
| Hip Roll | 51 | 61 | hip_roll_l_joint / hip_roll_r_joint |
| Hip Pitch | 52 | 62 | hip_pitch_l_joint / hip_pitch_r_joint |
| Hip Yaw | 53 | 63 | hip_yaw_l_joint / hip_yaw_r_joint |
| Knee Pitch | 54 | 64 | knee_pitch_l_joint / knee_pitch_r_joint |
| Ankle Pitch | 55 | 65 | ankle_pitch_l_joint / ankle_pitch_r_joint |
| Ankle Roll | 56 | 66 | ankle_roll_l_joint / ankle_roll_r_joint |
