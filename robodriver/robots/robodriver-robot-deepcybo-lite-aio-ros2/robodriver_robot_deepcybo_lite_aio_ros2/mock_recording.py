"""ROS2 mock inputs for DeepCybo Lite RoboDriver recording smoke tests.

This node is intentionally small and boring: it publishes deterministic arm and
gripper motion on the BAR Lite topics and mirrors one USB camera compressed
stream to the three camera topics expected by the RoboDriver package.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState

from bar_msgs.msg import MITCommand

from .config import ARM_JOINT_NAMES, LITE_JOINT_NAMES, DeepcyboLiteRos2Topics


def _reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        durability=DurabilityPolicy.VOLATILE,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _best_effort_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        durability=DurabilityPolicy.VOLATILE,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class DeepcyboLiteMockRecordingNode(Node):
    """Publish arm/gripper mock data and bridge a single camera to three topics."""

    def __init__(
        self,
        *,
        topics: DeepcyboLiteRos2Topics | None = None,
        camera_source_topic: str = "/camera1/image_raw/compressed",
        arm_rate_hz: float = 50.0,
        command_stiffness: float = 20.0,
        command_damping: float = 2.0,
        motion_scale: float = 0.35,
        synthetic_camera_rate_hz: float | None = None,
    ) -> None:
        super().__init__("deepcybo_lite_mock_recording")
        self.topics = topics or DeepcyboLiteRos2Topics()
        self.camera_source_topic = camera_source_topic
        self.arm_rate_hz = float(arm_rate_hz)
        self.command_stiffness = float(command_stiffness)
        self.command_damping = float(command_damping)
        self.motion_scale = float(motion_scale)
        self.synthetic_camera_rate_hz = synthetic_camera_rate_hz
        self._tick = 0
        self._image_tick = 0

        self._qos_reliable = _reliable_qos()
        self._qos_best_effort = _best_effort_qos()

        self._joint_pub = self.create_publisher(
            JointState,
            self.topics.joint_states,
            self._qos_best_effort,
        )
        self._command_pub = self.create_publisher(
            MITCommand,
            self.topics.command,
            self._qos_reliable,
        )
        self._camera_publishers = [
            self.create_publisher(CompressedImage, topic, self._qos_reliable)
            for topic in (
                self.topics.camera_head,
                self.topics.camera_wrist_left,
                self.topics.camera_wrist_right,
            )
        ]
        self.create_subscription(
            CompressedImage,
            self.camera_source_topic,
            self._on_camera,
            self._qos_reliable,
        )

        period = 1.0 / max(self.arm_rate_hz, 1.0)
        self.create_timer(period, self._publish_arm_messages)
        if self.synthetic_camera_rate_hz is not None:
            image_period = 1.0 / max(float(self.synthetic_camera_rate_hz), 1.0)
            self.create_timer(image_period, self._publish_synthetic_camera)

        self.get_logger().info(
            f"mock recording node ready | arm {self.arm_rate_hz:.1f} Hz | "
            f"{self.camera_source_topic} -> "
            f"[{self.topics.camera_head}, {self.topics.camera_wrist_left}, "
            f"{self.topics.camera_wrist_right}]"
            + (
                f" | synthetic_camera={self.synthetic_camera_rate_hz:.1f} Hz"
                if self.synthetic_camera_rate_hz is not None
                else ""
            )
        )

    def _positions(self, phase_shift: float = 0.0) -> list[float]:
        t = self._tick / max(self.arm_rate_hz, 1.0)
        values: list[float] = []
        for i, _name in enumerate(ARM_JOINT_NAMES):
            phase = phase_shift + 0.37 * i
            slow = math.sin(2.0 * math.pi * 0.13 * t + phase)
            fast = 0.25 * math.sin(2.0 * math.pi * 0.43 * t + phase * 0.5)
            values.append(self.motion_scale * (slow + fast))

        left_gripper = 0.5 + 0.5 * math.sin(2.0 * math.pi * 0.11 * t + phase_shift)
        right_gripper = 0.5 + 0.5 * math.sin(
            2.0 * math.pi * 0.11 * t + phase_shift + 0.7
        )
        values.extend([left_gripper, right_gripper])
        return values

    def _publish_arm_messages(self) -> None:
        stamp = self.get_clock().now().to_msg()
        follower_position = self._positions(phase_shift=0.0)
        leader_position = self._positions(phase_shift=0.12)

        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = list(LITE_JOINT_NAMES)
        joint_state.position = follower_position
        joint_state.velocity = [0.0] * len(LITE_JOINT_NAMES)
        joint_state.effort = [0.0] * len(LITE_JOINT_NAMES)
        self._joint_pub.publish(joint_state)

        command = MITCommand()
        command.header.stamp = stamp
        command.joint_names = list(LITE_JOINT_NAMES)
        command.position = leader_position
        command.velocity = [0.0] * len(LITE_JOINT_NAMES)
        command.effort = [0.0] * len(LITE_JOINT_NAMES)
        command.stiffness = [self.command_stiffness] * len(LITE_JOINT_NAMES)
        command.damping = [self.command_damping] * len(LITE_JOINT_NAMES)
        self._command_pub.publish(command)

        self._tick += 1

    def _on_camera(self, msg: CompressedImage) -> None:
        # Preserve stamp and payload so the three republished streams remain
        # time-aligned for ApproximateTimeSynchronizer.
        for publisher in self._camera_publishers:
            publisher.publish(msg)

    def _publish_synthetic_camera(self) -> None:
        height, width = 480, 640
        x = np.linspace(0, 255, width, dtype=np.uint16)
        y = np.linspace(0, 255, height, dtype=np.uint16)[:, None]
        phase = (self._image_tick * 3) % 255
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = ((x[None, :] + phase) % 255).astype(np.uint8)
        frame[:, :, 1] = ((y + phase * 2) % 255).astype(np.uint8)
        frame[:, :, 2] = 120
        cv2.putText(
            frame,
            f"DeepCybo Lite smoke {self._image_tick:04d}",
            (24, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            self.get_logger().warning("failed to encode synthetic camera frame")
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "synthetic_camera"
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self._on_camera(msg)
        self._image_tick += 1


def _make_topics_from_args(args: argparse.Namespace) -> DeepcyboLiteRos2Topics:
    return DeepcyboLiteRos2Topics(
        joint_states=args.joint_states_topic,
        command=args.command_topic,
        camera_head=args.camera_head_topic,
        camera_wrist_left=args.camera_wrist_left_topic,
        camera_wrist_right=args.camera_wrist_right_topic,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish DeepCybo Lite mock arm/gripper messages and bridge one compressed "
            "USB camera stream to the three RoboDriver camera topics."
        )
    )
    defaults = DeepcyboLiteRos2Topics()
    parser.add_argument("--camera-source-topic", default="/camera1/image_raw/compressed")
    parser.add_argument("--joint-states-topic", default=defaults.joint_states)
    parser.add_argument("--command-topic", default=defaults.command)
    parser.add_argument("--camera-head-topic", default=defaults.camera_head)
    parser.add_argument("--camera-wrist-left-topic", default=defaults.camera_wrist_left)
    parser.add_argument("--camera-wrist-right-topic", default=defaults.camera_wrist_right)
    parser.add_argument("--arm-rate-hz", type=float, default=50.0)
    parser.add_argument("--command-stiffness", type=float, default=20.0)
    parser.add_argument("--command-damping", type=float, default=2.0)
    parser.add_argument("--motion-scale", type=float, default=0.35)
    parser.add_argument(
        "--synthetic-camera-rate-hz",
        type=float,
        default=None,
        help=(
            "If set, publish generated JPEG frames directly to the three "
            "RoboDriver camera topics at this rate."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rclpy.init()
    node = DeepcyboLiteMockRecordingNode(
        topics=_make_topics_from_args(args),
        camera_source_topic=args.camera_source_topic,
        arm_rate_hz=args.arm_rate_hz,
        command_stiffness=args.command_stiffness,
        command_damping=args.command_damping,
        motion_scale=args.motion_scale,
        synthetic_camera_rate_hz=args.synthetic_camera_rate_hz,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
