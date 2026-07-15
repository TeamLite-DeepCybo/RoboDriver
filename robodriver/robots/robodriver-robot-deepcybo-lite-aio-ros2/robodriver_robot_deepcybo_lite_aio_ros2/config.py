"""DeepCybo Lite 双臂机器人 — RoboDriver 配置与 ROS2 话题约定。

向量顺序：
  左臂 7 关节 -> 右臂 7 关节 -> 左夹爪 -> 右夹爪（共 16 维）
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode
from lerobot.robots.config import RobotConfig


# ---------------------------------------------------------------------------
# ROS2 话题（可按现场在实例化时覆盖）
# ---------------------------------------------------------------------------
@dataclass
class DeepcyboLiteRos2Topics:
    """obs = 从臂反馈；action = 发往从臂的 MIT 控制指令。"""

    # --- observation：从臂 / 本体反馈 (sensor_msgs/JointState) ---
    joint_states: str = "/slave/lite/joint_states"

    # --- action：遥操 / 目标指令 (bar_msgs/MITCommand) ---
    command: str = "/slave/remote_policy_controller/command"

    # --- 相机 (CompressedImage, 30Hz) ---
    camera_head: str = "/deepcybo/lite/camera/head/image_raw/compressed"
    camera_wrist_left: str = "/deepcybo/lite/camera/wrist_left/image_raw/compressed"
    camera_wrist_right: str = "/deepcybo/lite/camera/wrist_right/image_raw/compressed"


ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_wrist_yaw",
    "left_wrist_roll",
    "left_wrist_pitch",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_wrist_yaw",
    "right_wrist_roll",
    "right_wrist_pitch",
)

GRIPPER_JOINT_NAMES: tuple[str, ...] = (
    "left_gripper",
    "right_gripper",
)

LITE_JOINT_NAMES: tuple[str, ...] = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES


DEFAULT_DATA_ROOT = Path(
    os.getenv("DEEPCYBO_LITE_DATA_ROOT", "/media/stvli/0EE4-E658")
).expanduser()


def _build_lite_motor_block(
    norm_mode_body: MotorNormMode,
    *,
    start_id: int,
) -> Dict[str, Motor]:
    """返回有序 dict（插入顺序 = 向量下标 = Lite 16 维顺序）。"""
    motors: Dict[str, Motor] = {}
    for offset, name in enumerate(LITE_JOINT_NAMES):
        motors[name] = Motor(start_id + offset, "deepcybo_joint", norm_mode_body)
    return motors


@RobotConfig.register_subclass("deepcybo-lite-aio-ros2")
@dataclass
class DeepcyboLiteAioRos2RobotConfig(RobotConfig):
    """DeepCybo Lite 双臂 aio-ros2 采集配置。"""

    use_degrees = False
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    # 与 galaxealite 一致：外层 key 须与 node.recv_leader / recv_follower 的组件名一致
    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "leader_arms": _build_lite_motor_block(norm_mode_body, start_id=1),
        }
    )

    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "follower_arms": _build_lite_motor_block(norm_mode_body, start_id=1),
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_head": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
            "image_wrist_left": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=640,
                height=480,
            ),
            "image_wrist_right": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # 关节与相机均为 30Hz，与 Record 主循环及中台出图一致
    control_fps: int = 30
    camera_fps: int = 30

    # RoboDriver 回放时发布 MITCommand 所需的默认 PD 参数。
    command_stiffness: float = 20.0
    command_damping: float = 2.0

    use_videos: bool = False

    # 推理部署模式下无需主臂 leader 数据（默认 True 保持向后兼容）
    require_leader: bool = True

    microphones: Dict[str, int] = field(default_factory=dict)

    ros2_topics: DeepcyboLiteRos2Topics = field(
        default_factory=DeepcyboLiteRos2Topics
    )
