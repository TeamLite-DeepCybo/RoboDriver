import time
from functools import cached_property
from typing import Any

import logging_mp
import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import DeepcyboLiteAioRos2RobotConfig
from .node import STATE_DIM, DeepcyboLiteAioRos2RobotNode
from .status import DeepcyboLiteAioRos2RobotStatus

logger = logging_mp.getLogger(__name__)

LEADER_COMP = "leader_arms"
FOLLOWER_COMP = "follower_arms"


class DeepcyboLiteAioRos2Robot(Robot):
    config_class = DeepcyboLiteAioRos2RobotConfig
    name = "deepcybo-lite-aio-ros2"

    def __init__(self, config: DeepcyboLiteAioRos2RobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.leader_motors = config.leader_motors
        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras: list[str] = []

        self.status = DeepcyboLiteAioRos2RobotStatus()
        self.robot_ros2_node = DeepcyboLiteAioRos2RobotNode(
            topics=config.ros2_topics,
            control_fps=config.control_fps,
            camera_fps=config.camera_fps,
            command_stiffness=config.command_stiffness,
            command_damping=config.command_damping,
        )

        self.connected = False
        self.logs: dict[str, Any] = {}

    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {
            f"follower_{joint_name}.pos": float
            for _comp_name, joints in self.follower_motors.items()
            for joint_name in joints.keys()
        }

    @property
    def _leader_motors_ft(self) -> dict[str, type]:
        return {
            f"leader_{joint_name}.pos": float
            for _comp_name, joints in self.leader_motors.items()
            for joint_name in joints.keys()
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._leader_motors_ft

    @property
    def is_connected(self) -> bool:
        return self.connected

    # ------------------------------------------------------------------
    # 与 node 缓存一致性检查
    # ------------------------------------------------------------------
    def _has_valid_leader_vector(self) -> bool:
        node = self.robot_ros2_node
        return (
            node._leader_arm_ok
            and LEADER_COMP in node.recv_leader
            and node.recv_leader[LEADER_COMP].shape == (STATE_DIM,)
        )

    def _has_valid_follower_vector(self) -> bool:
        node = self.robot_ros2_node
        return (
            node._follower_arm_ok
            and FOLLOWER_COMP in node.recv_follower
            and node.recv_follower[FOLLOWER_COMP].shape == (STATE_DIM,)
        )

    def _missing_cameras(self) -> list[str]:
        return [
            name
            for name in self.cameras
            if name not in self.connect_excluded_cameras
            and name not in self.robot_ros2_node.recv_images
        ]

    def _received_cameras(self) -> list[str]:
        return [
            name
            for name in self.cameras
            if name not in self.connect_excluded_cameras
            and name in self.robot_ros2_node.recv_images
        ]

    def connect(self) -> None:
        timeout = 20
        start_time = time.perf_counter()

        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        while True:
            cameras_ok = len(self._missing_cameras()) == 0
            leader_ok = self._has_valid_leader_vector() if self.config.require_leader else True
            follower_ok = self._has_valid_follower_vector()

            if cameras_ok and leader_ok and follower_ok:
                break

            if time.perf_counter() - start_time > timeout:
                parts: list[str] = []
                if not cameras_ok:
                    parts.append(
                        f"等待摄像头超时: 未收到 [{', '.join(self._missing_cameras())}]; "
                        f"已收到 [{', '.join(self._received_cameras())}]"
                    )
                if not leader_ok and self.config.require_leader:
                    parts.append(
                        f"等待 action(leader) 超时: 需要缓存键 '{LEADER_COMP}' "
                        f"且 {STATE_DIM} 维 canonical Lite 向量; "
                        f"当前 leader_ok={self.robot_ros2_node._leader_arm_ok}, "
                        f"keys={list(self.robot_ros2_node.recv_leader.keys())}"
                    )
                if not follower_ok:
                    parts.append(
                        f"等待 observation(follower) 超时: 需要缓存键 '{FOLLOWER_COMP}' "
                        f"且 {STATE_DIM} 维 canonical Lite 向量; "
                        f"当前 follower_ok={self.robot_ros2_node._follower_arm_ok}, "
                        f"keys={list(self.robot_ros2_node.recv_follower.keys())}"
                    )
                raise TimeoutError(
                    "连接超时，未满足的条件: " + "; ".join(parts)
                )

            time.sleep(0.01)

        success_messages = [
            f"摄像头: {', '.join(self._received_cameras())}",
            f"action ({LEADER_COMP}): {STATE_DIM} 维",
            f"observation ({FOLLOWER_COMP}): {STATE_DIM} 维",
        ]
        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)

        for i in range(self.status.specifications.camera.number):
            self.status.specifications.camera.information[i].is_connect = True
        for i in range(self.status.specifications.arm.number):
            self.status.specifications.arm.information[i].is_connect = True

        self.connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self._has_valid_follower_vector():
            raise DeviceNotConnectedError(
                f"{self}: follower 关节/夹爪数据无效或 canonical {STATE_DIM} 维未恢复，本帧不采集 observation"
            )

        obs_dict: dict[str, Any] = {}
        for comp_name, joints in self.follower_motors.items():
            if comp_name not in self.robot_ros2_node.recv_follower:
                continue
            vec = self.robot_ros2_node.recv_follower[comp_name]
            for i, joint_name in enumerate(joints.keys()):
                obs_dict[f"follower_{joint_name}.pos"] = float(vec[i])

        for cam_key in self.cameras:
            if cam_key in self.robot_ros2_node.recv_images:
                obs_dict[cam_key] = self.robot_ros2_node.recv_images[cam_key]

        return obs_dict

    def get_action(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.config.require_leader and not self._has_valid_leader_vector():
            raise DeviceNotConnectedError(
                f"{self}: leader 关节/夹爪数据无效或 canonical {STATE_DIM} 维未恢复，本帧不采集 action"
            )

        act_dict: dict[str, Any] = {}
        for comp_name, joints in self.leader_motors.items():
            if comp_name not in self.robot_ros2_node.recv_leader:
                continue
            # 变量名说明：数据来自 recv_leader（遥操/目标），不是 follower
            leader_vec = self.robot_ros2_node.recv_leader[comp_name]
            for i, joint_name in enumerate(joints.keys()):
                act_dict[f"leader_{joint_name}.pos"] = float(leader_vec[i])

        return act_dict

    def send_action(self, action: dict[str, Any]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. Run `robot.connect()` first."
            )

        ordered: list[float] = []
        for joint_name in self.leader_motors[LEADER_COMP].keys():
            key = f"leader_{joint_name}.pos"
            if key not in action:
                raise ValueError(f"send_action 缺少键: {key}")
            ordered.append(float(action[key]))

        goal_joint_numpy = np.array(ordered, dtype=np.float32)
        try:
            self.robot_ros2_node.ros_replay(goal_joint_numpy)
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            raise

    def update_status(self) -> str:
        node = self.robot_ros2_node
        for cam in self.status.specifications.camera.information:
            cam.is_connect = node.recv_images_status.get(cam.name, 0) > 0

        for arm in self.status.specifications.arm.information:
            if arm.name == LEADER_COMP:
                arm.is_connect = (
                    node._leader_arm_ok
                    and node.recv_leader_status.get(LEADER_COMP, 0) > 0
                )
            elif arm.name == FOLLOWER_COMP:
                arm.is_connect = (
                    node._follower_arm_ok
                    and node.recv_follower_status.get(FOLLOWER_COMP, 0) > 0
                )

        return self.status.to_json()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. Run `robot.connect()` before disconnecting."
            )
        if hasattr(self, "robot_ros2_node"):
            self.robot_ros2_node.destroy()
        self.connected = False

    def __del__(self) -> None:
        try:
            if getattr(self, "connected", False):
                self.disconnect()
        except Exception:
            pass

    def get_node(self) -> DeepcyboLiteAioRos2RobotNode:
        return self.robot_ros2_node
