"""Synthetic UMI rig publisher for off-rig testing (spec §8).

Per camera-frame tick (fps): ONE stamp shared by both GripperTrack msgs, the
world_head pose, and all three camera images — exactly like the real rig,
where they all derive from the same head image.

--drop-every N: every N frames, publish drop_len frames of "tracking lost"
GripperTrack (present=False) for the LEFT arm to exercise hold-last + flags.
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node as ROS2Node
from sensor_msgs.msg import CompressedImage, JointState

from .config import GRIPPER_JOINTS, DeepcyboLiteUmiRos2Topics

try:
    from lite_aruco_umi_msgs.msg import GripperTrack
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lite_aruco_umi_msgs not on the ROS overlay; source the collection "
        "workspace first."
    ) from exc

# Fixed T_world_head used by every frame: head 1 m above the tag, looking
# straight down (180 deg about x  => quat (1,0,0,0)).
WORLD_HEAD_POS = (0.2, 0.3, 1.0)
WORLD_HEAD_QUAT = (1.0, 0.0, 0.0, 0.0)


def _set_pose(msg: Pose, pos, quat) -> None:
    msg.position.x, msg.position.y, msg.position.z = map(float, pos)
    (msg.orientation.x, msg.orientation.y,
     msg.orientation.z, msg.orientation.w) = map(float, quat)


def _test_image(frame_idx: int, label: str) -> bytes:
    img = np.full((480, 640, 3), 32, dtype=np.uint8)
    cv2.putText(img, f"{label} {frame_idx}", (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class DeepcyboLiteUmiMockNode(ROS2Node):
    def __init__(
        self,
        topics: DeepcyboLiteUmiRos2Topics | None = None,
        fps: float = 30.0,
        joint_rate_hz: float = 50.0,
        drop_every: int = 0,
        drop_len: int = 15,
    ):
        super().__init__("deepcybo_lite_umi_mock")
        t = topics or DeepcyboLiteUmiRos2Topics()
        self.topics = t
        self.drop_every = int(drop_every)
        self.drop_len = int(drop_len)

        self.pub_track_left = self.create_publisher(GripperTrack, t.track_left, 10)
        self.pub_track_right = self.create_publisher(GripperTrack, t.track_right, 10)
        self.pub_world = self.create_publisher(PoseStamped, t.world_head, 10)
        self.pub_joints = self.create_publisher(JointState, t.joint_states, 10)
        self.pub_cam = {
            "head": self.create_publisher(CompressedImage, t.camera_head, 10),
            "wl": self.create_publisher(CompressedImage, t.camera_wrist_left, 10),
            "wr": self.create_publisher(CompressedImage, t.camera_wrist_right, 10),
        }

        self._frame = 0
        self.create_timer(1.0 / fps, self._tick_frame)
        self.create_timer(1.0 / joint_rate_hz, self._tick_joints)
        self.get_logger().info(
            f"UMI mock up @ {fps} Hz (drop_every={drop_every}, drop_len={drop_len})"
        )

    def _dropping_left(self) -> bool:
        if self.drop_every <= 0:
            return False
        return (self._frame % self.drop_every) < self.drop_len

    def _track_msg(self, stamp, phase: float, dropped: bool) -> GripperTrack:
        msg = GripperTrack()
        msg.header.stamp = stamp
        msg.header.frame_id = "head"
        msg.timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if dropped:
            msg.tracked = False
            msg.present = False
            msg.has_tcp = False
            msg.n_markers = 0
            msg.reproj = float("inf")
            return msg
        # circle r=0.15 m at z=0.5 m in the HEAD frame
        pos = (0.15 * math.cos(phase), 0.15 * math.sin(phase), 0.5)
        _set_pose(msg.tcp_pose, pos, (0.0, 0.0, 0.0, 1.0))
        _set_pose(msg.cube_pose, pos, (0.0, 0.0, 0.0, 1.0))
        msg.tracked = True
        msg.present = True
        msg.has_tcp = True
        msg.n_markers = 3
        msg.reproj = 0.5
        return msg

    def _tick_frame(self) -> None:
        stamp = self.get_clock().now().to_msg()
        phase = 2.0 * math.pi * (self._frame / 90.0)  # 3 s per revolution

        self.pub_track_left.publish(
            self._track_msg(stamp, phase, dropped=self._dropping_left())
        )
        self.pub_track_right.publish(
            self._track_msg(stamp, phase + math.pi, dropped=False)
        )

        world = PoseStamped()
        world.header.stamp = stamp
        world.header.frame_id = "world"
        _set_pose(world.pose, WORLD_HEAD_POS, WORLD_HEAD_QUAT)
        self.pub_world.publish(world)

        for key, label in (("head", "HEAD"), ("wl", "WRIST-L"), ("wr", "WRIST-R")):
            img = CompressedImage()
            img.header.stamp = stamp
            img.format = "jpeg"
            img.data = _test_image(self._frame, label)
            self.pub_cam[key].publish(img)

        self._frame += 1

    def _tick_joints(self) -> None:
        now = self.get_clock().now()
        t_sec = now.nanoseconds * 1e-9
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = list(GRIPPER_JOINTS)
        msg.position = [
            0.5 + 0.5 * math.sin(t_sec),          # left in [0,1]
            0.5 + 0.5 * math.sin(t_sec + 1.0),    # right in [0,1]
        ]
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]
        self.pub_joints.publish(msg)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Publish synthetic UMI rig topics.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--joint-rate-hz", type=float, default=50.0)
    parser.add_argument("--drop-every", type=int, default=0,
                        help="every N frames, drop LEFT tracking for --drop-len frames")
    parser.add_argument("--drop-len", type=int, default=15)
    args = parser.parse_args(argv)

    rclpy.init()
    node = DeepcyboLiteUmiMockNode(
        fps=args.fps,
        joint_rate_hz=args.joint_rate_hz,
        drop_every=args.drop_every,
        drop_len=args.drop_len,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
