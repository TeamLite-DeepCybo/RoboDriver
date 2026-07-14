# robodriver_robot_deepcybo_lite_umi_ros2/config.py
"""DeepCybo Lite UMI handheld rig — RoboDriver config and ROS2 topic contract.

State vector (16-dim, spec §5):
  left eef pose7 (x,y,z,qx,qy,qz,qw) | left gripper |
  right eef pose7                    | right gripper
Quality vector (7-dim): flags for filtering only — never feed to the policy.
"""
from dataclasses import dataclass, field
from typing import Dict

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig


@dataclass
class DeepcyboLiteUmiRos2Topics:
    """All live inputs of the UMI rig (spec §4). Overridable at instantiation."""

    track_left: str = "/umi/left/track"        # lite_aruco_umi_msgs/GripperTrack
    track_right: str = "/umi/right/track"      # lite_aruco_umi_msgs/GripperTrack
    world_head: str = "/umi/world_head/pose"   # geometry_msgs/PoseStamped
    joint_states: str = "/lite/joint_states"   # sensor_msgs/JointState
    camera_head: str = "/deepcybo/lite/camera/head/image_raw/compressed"
    camera_wrist_left: str = "/deepcybo/lite/camera/wrist_left/image_raw/compressed"
    camera_wrist_right: str = "/deepcybo/lite/camera/wrist_right/image_raw/compressed"


EEF_FEATURE_NAMES: tuple[str, ...] = (
    "left_eef_x", "left_eef_y", "left_eef_z",
    "left_eef_qx", "left_eef_qy", "left_eef_qz", "left_eef_qw",
    "left_gripper",
    "right_eef_x", "right_eef_y", "right_eef_z",
    "right_eef_qx", "right_eef_qy", "right_eef_qz", "right_eef_qw",
    "right_gripper",
)

QUALITY_FEATURE_NAMES: tuple[str, ...] = (
    "left_tracked", "left_present", "left_reproj",
    "right_tracked", "right_present", "right_reproj",
    "world_fresh",
)

GRIPPER_JOINTS: tuple[str, ...] = ("left_gripper", "right_gripper")

STATE_DIM = len(EEF_FEATURE_NAMES)      # 16
QUALITY_DIM = len(QUALITY_FEATURE_NAMES)  # 7


@RobotConfig.register_subclass("deepcybo-lite-umi-ros2")
@dataclass
class DeepcyboLiteUmiRos2RobotConfig(RobotConfig):
    """DeepCybo Lite UMI handheld rig eef-pose collection config."""

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_head": OpenCVCameraConfig(
                index_or_path=0, fps=30, width=640, height=480
            ),
            "image_wrist_left": OpenCVCameraConfig(
                index_or_path=1, fps=30, width=640, height=480
            ),
            "image_wrist_right": OpenCVCameraConfig(
                index_or_path=2, fps=30, width=640, height=480
            ),
        }
    )

    control_fps: int = 30
    camera_fps: int = 30

    # RViz verification overlay (spec §8): off for real sessions.
    publish_debug: bool = False

    use_videos: bool = False
    microphones: Dict[str, int] = field(default_factory=dict)

    ros2_topics: DeepcyboLiteUmiRos2Topics = field(
        default_factory=DeepcyboLiteUmiRos2Topics
    )
