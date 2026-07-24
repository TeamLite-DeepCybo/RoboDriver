# robodriver_robot_deepcybo_lite_umi_ros2/filter_node.py
"""Republish the live ArUco gripper poses, filtered for teleop (spec 2026-07-22).

Subscribes to the tracker's raw GripperTrack topics, runs each arm through a
causal pose filter, and republishes the smoothed pose plus a staleness flag.

The TRACKER keeps publishing raw measurements -- this node is downstream, so
recorded datasets stay pristine and the offline smoother's anchor set stays
honest. Nothing consumes the filtered topics until the IK bridge exists.

Fail-safe contract: while an arm is `stale` (frozen, or silent past the
watchdog timeout), NO pose is published for it at all -- only the `Bool`. A
consumer that never receives a pose cannot act on a stale one, which a stale
pose republished under a fresh timestamp could otherwise trick it into doing.

Run:
    umi-filter-node --filter one-euro
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node as ROS2Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool

from .config import DeepcyboLiteUmiRos2RobotConfig
from .pose_filter import (DEFAULT_MAX_PREDICT_DISPLACEMENT_M,
                          DEFAULT_MAX_PREDICT_FRAMES, EkfPoseFilter,
                          OneEuroPoseFilter)

try:
    from lite_aruco_umi_msgs.msg import GripperTrack
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lite_aruco_umi_msgs not on the ROS overlay; source the collection "
        "workspace first."
    ) from exc

FILTERS = {"one-euro": OneEuroPoseFilter, "ekf": EkfPoseFilter}

#: Wall-clock silence watchdog: how often to check for a dead arm, and how
#: long an arm may go without a track message before it is declared stale.
DEFAULT_WATCHDOG_HZ = 10.0
DEFAULT_SILENCE_TIMEOUT_S = 0.2

ARMS = ("left", "right")


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class FilteredPoseNode(ROS2Node):
    def __init__(self, filter_name: str = "one-euro",
                 max_predict_frames: int = DEFAULT_MAX_PREDICT_FRAMES,
                 max_predict_displacement_m: float =
                 DEFAULT_MAX_PREDICT_DISPLACEMENT_M,
                 min_cutoff: float | None = None,
                 beta: float | None = None,
                 silence_timeout_s: float = DEFAULT_SILENCE_TIMEOUT_S):
        super().__init__("umi_filtered_pose")
        cfg = DeepcyboLiteUmiRos2RobotConfig()
        t = cfg.ros2_topics
        factory = FILTERS[filter_name]

        filter_kwargs = dict(
            max_predict_frames=max_predict_frames,
            max_predict_displacement_m=max_predict_displacement_m,
        )
        # min_cutoff/beta are One-Euro-only; passing them to EkfPoseFilter
        # would raise a TypeError, so only forward what the chosen filter
        # actually accepts.
        if filter_name == "one-euro":
            if min_cutoff is not None:
                filter_kwargs["min_cutoff"] = min_cutoff
            if beta is not None:
                filter_kwargs["beta"] = beta

        self._filters = {arm: factory(**filter_kwargs) for arm in ARMS}
        self._silence_timeout_s = float(silence_timeout_s)
        self._last_recv_time = {arm: None for arm in ARMS}
        self._silence_warned = {arm: False for arm in ARMS}

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

        self.create_timer(1.0 / DEFAULT_WATCHDOG_HZ, self._check_silence)

        self.get_logger().info(
            f"filtered-pose node up | filter={filter_name} "
            f"max_predict_frames={max_predict_frames} "
            f"max_predict_displacement_m={max_predict_displacement_m} "
            f"silence_timeout_s={self._silence_timeout_s} | publishing "
            f"/umi/filtered/eef_left|right + /umi/filtered/stale_left|right"
        )

    def _on_track(self, arm: str, msg) -> None:
        try:
            self._last_recv_time[arm] = time.monotonic()
            self._silence_warned[arm] = False

            t = _stamp_to_sec(msg.header.stamp)
            usable = bool(msg.tracked) and bool(msg.present) and bool(msg.has_tcp)
            p = msg.tcp_pose.position
            o = msg.tcp_pose.orientation
            out = self._filters[arm].update(
                t, np.array([p.x, p.y, p.z]),
                np.array([o.x, o.y, o.z, o.w]), usable)

            self._stale_pub[arm].publish(Bool(data=bool(out.stale)))
            if out.stale or out.pos is None:
                return  # fail safe: never republish a pose while stale
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
        except Exception as e:
            # A single malformed message (e.g. an all-zero quaternion raising
            # in `Rotation.from_quat`) must not unwind `executor.spin()` and
            # kill the node -- log and drop this frame instead.
            self.get_logger().error(
                f"track[{arm}] callback error: {e}", throttle_duration_sec=5.0)

    def _check_silence(self) -> None:
        """Wall-clock watchdog: `_on_track` only runs when a message arrives,
        so a dead tracker/camera/topic (no callback firing at all) would
        otherwise leave the last `stale=False` standing forever. Runs at
        `DEFAULT_WATCHDOG_HZ` independent of message arrival.
        """
        now = time.monotonic()
        for arm in ARMS:
            last = self._last_recv_time[arm]
            if last is None:
                continue  # never received a message for this arm yet
            if now - last <= self._silence_timeout_s:
                continue
            if not self._silence_warned[arm]:
                self.get_logger().warn(
                    f"{arm}: no track message received for over "
                    f"{self._silence_timeout_s:.2f}s -- publishing stale "
                    f"and halting pose output for this arm"
                )
                self._silence_warned[arm] = True
            self._stale_pub[arm].publish(Bool(data=True))
            # fail safe: no pose publish while silent, same as a frozen filter


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Republish live ArUco gripper poses, filtered for teleop.")
    parser.add_argument("--filter", choices=tuple(FILTERS), default="one-euro")
    parser.add_argument("--max-predict-frames", type=int,
                        default=DEFAULT_MAX_PREDICT_FRAMES)
    parser.add_argument("--max-predict-displacement-m", type=float,
                        default=DEFAULT_MAX_PREDICT_DISPLACEMENT_M)
    parser.add_argument("--min-cutoff", type=float, default=None,
                        help="One-Euro min_cutoff (ignored for --filter ekf)")
    parser.add_argument("--beta", type=float, default=None,
                        help="One-Euro beta (ignored for --filter ekf)")
    parser.add_argument("--silence-timeout-s", type=float,
                        default=DEFAULT_SILENCE_TIMEOUT_S,
                        help="Publish stale=True and halt pose output for an "
                             "arm after this many seconds with no track "
                             "message received (wall-clock watchdog)")
    args = parser.parse_args(argv)

    rclpy.init()
    node = FilteredPoseNode(args.filter, args.max_predict_frames,
                            args.max_predict_displacement_m,
                            args.min_cutoff, args.beta,
                            args.silence_timeout_s)
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
