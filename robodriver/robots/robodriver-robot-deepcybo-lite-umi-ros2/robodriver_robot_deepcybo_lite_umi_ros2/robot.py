# robodriver_robot_deepcybo_lite_umi_ros2/robot.py
import time
from functools import cached_property
from typing import Any

import logging_mp
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import (
    EEF_FEATURE_NAMES,
    QUALITY_FEATURE_NAMES,
    DeepcyboLiteUmiRos2RobotConfig,
)
from .node import DeepcyboLiteUmiRos2RobotNode
from .status import LEFT_EEF, RIGHT_EEF, DeepcyboLiteUmiRos2RobotStatus

logger = logging_mp.get_logger(__name__)


class DeepcyboLiteUmiRos2Robot(Robot):
    config_class = DeepcyboLiteUmiRos2RobotConfig
    name = "deepcybo-lite-umi-ros2"

    def __init__(self, config: DeepcyboLiteUmiRos2RobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.connect_excluded_cameras: list[str] = []

        self.status = DeepcyboLiteUmiRos2RobotStatus()
        self.robot_ros2_node = DeepcyboLiteUmiRos2RobotNode(
            topics=config.ros2_topics,
            control_fps=config.control_fps,
            camera_fps=config.camera_fps,
            publish_debug=config.publish_debug,
        )

        self.connected = False
        self.logs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Feature contract (spec §5)
    # ------------------------------------------------------------------
    @property
    def _state_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in EEF_FEATURE_NAMES}

    @property
    def _quality_ft(self) -> dict[str, type]:
        # Filtering-only flags — training configs must EXCLUDE these from
        # the policy input (spec §10).
        return {f"{name}.flag": float for name in QUALITY_FEATURE_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._quality_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Mirror of the 16 core state floats (spec §5) — no quality flags.
        return {f"{name}.pos": float for name in EEF_FEATURE_NAMES}

    @property
    def is_connected(self) -> bool:
        return self.connected

    # ------------------------------------------------------------------
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

        node = self.robot_ros2_node
        while True:
            cameras_ok = len(self._missing_cameras()) == 0
            left_ok = node.left_valid()
            right_ok = node.right_valid()
            grippers_ok = node.grippers_valid()
            if cameras_ok and left_ok and right_ok and grippers_ok:
                break
            if time.perf_counter() - start_time > timeout:
                parts: list[str] = []
                if not cameras_ok:
                    parts.append(
                        f"cameras missing [{', '.join(self._missing_cameras())}]; "
                        f"received [{', '.join(self._received_cameras())}]"
                    )
                if not left_ok:
                    parts.append(
                        "left eef never composed (needs GripperTrack with "
                        "present+has_tcp AND a stamp-matched world_head pose)"
                    )
                if not right_ok:
                    parts.append("right eef never composed (same requirements)")
                if not grippers_ok:
                    parts.append(
                        "gripper joints not seen on joint_states "
                        "(need left_gripper + right_gripper)"
                    )
                raise TimeoutError("connect timeout, unmet: " + "; ".join(parts))
            time.sleep(0.01)

        logger.info(
            "[connected] cameras=%s | left/right eef composed | grippers ok | %.2fs",
            ", ".join(self._received_cameras()),
            time.perf_counter() - start_time,
        )
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

    # ------------------------------------------------------------------
    def _state_and_quality(self) -> dict[str, float]:
        node = self.robot_ros2_node
        state = node.state_vector()
        quality = node.quality_vector()
        if state is None or quality is None:
            # Post-connect this cannot regress to None (hold-last keeps the
            # vectors); guard anyway rather than KeyError downstream.
            raise DeviceNotConnectedError(
                f"{self}: eef state unavailable — was connect() successful?"
            )
        out: dict[str, float] = {}
        for i, name in enumerate(EEF_FEATURE_NAMES):
            out[f"{name}.pos"] = float(state[i])
        for i, name in enumerate(QUALITY_FEATURE_NAMES):
            out[f"{name}.flag"] = float(quality[i])
        return out

    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        obs_dict: dict[str, Any] = self._state_and_quality()
        for cam_key in self.cameras:
            if cam_key in self.robot_ros2_node.recv_images:
                obs_dict[cam_key] = self.robot_ros2_node.recv_images[cam_key]
        return obs_dict

    def get_action(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        state = self._state_and_quality()
        # Action mirrors the 16 core floats exactly (spec §5) — no flags.
        return {
            f"{name}.pos": state[f"{name}.pos"] for name in EEF_FEATURE_NAMES
        }

    def send_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "UMI rig is passive; deploy via joint-space replay (Route B) "
            "or a future IK bridge"
        )

    def update_status(self) -> str:
        node = self.robot_ros2_node
        for cam in self.status.specifications.camera.information:
            cam.is_connect = node.recv_images_status.get(cam.name, 0) > 0
        for arm in self.status.specifications.arm.information:
            if arm.name == LEFT_EEF:
                arm.is_connect = node.left_valid()
            elif arm.name == RIGHT_EEF:
                arm.is_connect = node.right_valid()
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

    def get_node(self) -> DeepcyboLiteUmiRos2RobotNode:
        return self.robot_ros2_node
