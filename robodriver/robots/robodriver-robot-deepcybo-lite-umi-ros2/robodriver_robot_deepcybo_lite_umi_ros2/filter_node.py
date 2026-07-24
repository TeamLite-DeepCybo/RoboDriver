# robodriver_robot_deepcybo_lite_umi_ros2/filter_node.py
"""Republish the live ArUco gripper poses, filtered for teleop (spec 2026-07-22).

Subscribes to the tracker's raw GripperTrack topics, runs each arm through a
causal pose filter, and republishes the smoothed pose plus a staleness flag.

The TRACKER keeps publishing raw measurements -- this node is downstream, so
recorded datasets stay pristine and the offline smoother's anchor set stays
honest. Nothing consumes the filtered topics until the IK bridge exists.

Run:
    umi-filter-node --filter one-euro
"""
from __future__ import annotations

import argparse

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node as ROS2Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool

from .config import DeepcyboLiteUmiRos2RobotConfig
from .pose_filter import EkfPoseFilter, OneEuroPoseFilter

try:
    from lite_aruco_umi_msgs.msg import GripperTrack
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lite_aruco_umi_msgs not on the ROS overlay; source the collection "
        "workspace first."
    ) from exc

FILTERS = {"one-euro": OneEuroPoseFilter, "ekf": EkfPoseFilter}


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class FilteredPoseNode(ROS2Node):
    def __init__(self, filter_name: str = "one-euro",
                 max_predict_frames: int = 3):
        super().__init__("umi_filtered_pose")
        cfg = DeepcyboLiteUmiRos2RobotConfig()
        t = cfg.ros2_topics
        factory = FILTERS[filter_name]
        self._filters = {
            arm: factory(max_predict_frames=max_predict_frames)
            for arm in ("left", "right")
        }
        qos = QoSProfile(durability=DurabilityPolicy.VOLATILE,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self._pose_pub = {}
        self._stale_pub = {}
        for arm, topic in (("left", t.track_left), ("right", t.track_right)):
            self.create_subscription(
                GripperTrack, topic,
                lambda msg, a=arm: self._on_track(a, msg), qos)
            self._pose_pub[arm] = self.create_publisher(
                PoseStamped, f"/umi/filtered/eef_{arm}", 10)
            self._stale_pub[arm] = self.create_publisher(
                Bool, f"/umi/filtered/stale_{arm}", 10)
        self.get_logger().info(
            f"filtered-pose node up | filter={filter_name} "
            f"max_predict_frames={max_predict_frames} | publishing "
            f"/umi/filtered/eef_left|right + /umi/filtered/stale_left|right"
        )

    def _on_track(self, arm: str, msg) -> None:
        t = _stamp_to_sec(msg.header.stamp)
        usable = bool(msg.tracked) and bool(msg.present) and bool(msg.has_tcp)
        p = msg.tcp_pose.position
        o = msg.tcp_pose.orientation
        out = self._filters[arm].update(
            t, np.array([p.x, p.y, p.z]),
            np.array([o.x, o.y, o.z, o.w]), usable)

        self._stale_pub[arm].publish(Bool(data=bool(out.stale)))
        if out.pos is None:
            return                      # uninitialised: publish no pose at all
        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = msg.header.frame_id
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
            float(out.pos[0]), float(out.pos[1]), float(out.pos[2]))
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = (
            float(out.quat[0]), float(out.quat[1]),
            float(out.quat[2]), float(out.quat[3]))
        self._pose_pub[arm].publish(ps)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Republish live ArUco gripper poses, filtered for teleop.")
    parser.add_argument("--filter", choices=tuple(FILTERS), default="one-euro")
    parser.add_argument("--max-predict-frames", type=int, default=3)
    args = parser.parse_args(argv)

    rclpy.init()
    node = FilteredPoseNode(args.filter, args.max_predict_frames)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
