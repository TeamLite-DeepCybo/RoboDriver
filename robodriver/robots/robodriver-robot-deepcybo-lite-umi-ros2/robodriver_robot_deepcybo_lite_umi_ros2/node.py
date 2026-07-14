# robodriver_robot_deepcybo_lite_umi_ros2/node.py
"""DeepCybo Lite UMI rig — ROS2 subscribe / stamp-pair / compose / cache.

Data flow per head-camera frame (all stamps identical — they derive from the
same image):  GripperTrack (head frame)  +  world_head PoseStamped
              -> T_world_tcp = T_world_head @ T_head_tcp  (compose.py)
Gripper opening comes from /lite/joint_states; cameras via approx-time sync.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node as ROS2Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import logging_mp

from . import se3
from .compose import (
    EefComposer,
    WorldBuffer,
    build_quality_vector,
    build_state_vector,
    stamp_to_ns,
)
from .config import GRIPPER_JOINTS, DeepcyboLiteUmiRos2Topics

try:
    from lite_aruco_umi_msgs.msg import GripperTrack
except ImportError as exc:  # pragma: no cover - needs collection ws overlay
    GripperTrack = None
    _MSGS_IMPORT_ERROR = exc
else:
    _MSGS_IMPORT_ERROR = None

CONNECT_TIMEOUT_FRAME = 10
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

logger = logging_mp.get_logger(__name__)


def _pose_to_T(pose) -> np.ndarray:
    """geometry_msgs/Pose -> 4x4 (pure floats in, se3 does the math)."""
    p, q = pose.position, pose.orientation
    return se3.pos_quat_to_T([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])


class DeepcyboLiteUmiRos2RobotNode(ROS2Node):
    def __init__(
        self,
        topics: Optional[DeepcyboLiteUmiRos2Topics] = None,
        control_fps: int = 30,
        camera_fps: int = 30,
        publish_debug: bool = False,
    ):
        if GripperTrack is None:
            raise ImportError(
                "Cannot import lite_aruco_umi_msgs.msg.GripperTrack. Source the "
                "collection workspace overlay before starting RoboDriver, e.g. "
                "`source ~/ros2_ws/install/setup.bash`."
            ) from _MSGS_IMPORT_ERROR

        super().__init__("deepcybo_lite_umi_ros2_driver")
        self.topics = topics or DeepcyboLiteUmiRos2Topics()
        self.control_fps = control_fps
        self.camera_fps = camera_fps
        self.publish_debug = bool(publish_debug)

        self.qos = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.qos_best_effort = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        t = self.topics
        self.create_subscription(
            GripperTrack, t.track_left,
            lambda msg: self._track_callback("left", msg), self.qos,
        )
        self.create_subscription(
            GripperTrack, t.track_right,
            lambda msg: self._track_callback("right", msg), self.qos,
        )
        self.create_subscription(
            PoseStamped, t.world_head, self._world_callback, self.qos,
        )
        self.create_subscription(
            JointState, t.joint_states, self._joint_state_callback,
            self.qos_best_effort,
        )

        self.last_image_recv_time_ns = 0
        self.min_camera_interval_ns = int(1e9 / max(camera_fps, 1))

        self._world = WorldBuffer()
        self._composer = {"left": EefComposer(), "right": EefComposer()}
        self._eef_state = {"left": None, "right": None}   # latest EefState
        self._grippers: Dict[str, float] = {}             # joint -> opening
        # A track and its same-stamp world pose arrive from independent nodes in
        # any order. Compose is triggered by whichever arrives SECOND (see
        # _track_callback / _world_callback), so ingest order does not matter.
        # _pending_track holds a track still waiting for its world; _last_composed_ns
        # guards against composing the same frame twice.
        self._pending_track: Dict[str, Optional[tuple]] = {"left": None, "right": None}
        self._last_composed_ns: Dict[str, Optional[int]] = {"left": None, "right": None}

        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}

        self.lock = threading.Lock()

        self._init_image_message_filters()
        self._init_debug_publishers()   # no-op unless publish_debug (Task 9)

        logger.info(
            "[DeepCybo Lite UMI] node ready | tracks=(%s, %s) world=%s "
            "joints=%s control_fps=%s camera_fps=%s debug=%s",
            t.track_left, t.track_right, t.world_head, t.joint_states,
            control_fps, camera_fps, self.publish_debug,
        )

    # ------------------------------------------------------------------
    # Stream A + C -> composed world-frame eef state
    # ------------------------------------------------------------------
    def _finalize_frame(self, arm: str, track: tuple):
        """Compose one frame for `arm` exactly once. Call with self.lock held.

        Returns (state, stamp_msg) for a later out-of-lock debug publish, or None
        if this frame's stamp was already composed. `EefComposer.update` looks up
        the world itself, so a still-missing world yields world_fresh=0 + hold-last.
        """
        ns, T_head_tcp, tracked, present, reproj, stamp_msg = track
        if self._last_composed_ns[arm] == ns:
            return None
        state = self._composer[arm].update(
            ns, T_head_tcp,
            tracked=tracked, present=present, reproj=reproj, world=self._world,
        )
        self._eef_state[arm] = state
        self._last_composed_ns[arm] = ns
        return state, stamp_msg

    def _world_callback(self, msg: PoseStamped) -> None:
        try:
            ns = stamp_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            to_publish = []
            with self.lock:
                self._world.add(ns, _pose_to_T(msg.pose))
                # A pending track whose world just arrived can now compose fresh.
                for arm in ("left", "right"):
                    pend = self._pending_track[arm]
                    if pend is not None and self._world.lookup(pend[0]) is not None:
                        done = self._finalize_frame(arm, pend)
                        self._pending_track[arm] = None
                        if done is not None:
                            to_publish.append((arm, done))
            for arm, (state, stamp_msg) in to_publish:
                self._publish_debug_pose(arm, state, stamp_msg)  # Task 9
        except Exception as e:
            self.get_logger().error(f"world_head callback error: {e}")

    def _track_callback(self, arm: str, msg) -> None:
        try:
            ns = stamp_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            usable = bool(msg.present) and bool(msg.has_tcp)
            T_head_tcp = _pose_to_T(msg.tcp_pose) if usable else None
            track = (ns, T_head_tcp, bool(msg.tracked), bool(msg.present),
                     float(msg.reproj), msg.header.stamp)
            to_publish = []
            with self.lock:
                # A pending track from an earlier frame means its world never
                # arrived; finalize it now (world lookup misses -> world_fresh=0,
                # hold-last) so the dropout is still recorded, before this frame.
                pend = self._pending_track[arm]
                if pend is not None and pend[0] != ns:
                    done = self._finalize_frame(arm, pend)
                    if done is not None:
                        to_publish.append(done)
                    self._pending_track[arm] = None
                # This frame: compose now if its world is already buffered, else
                # stash and let _world_callback finalize it when the world arrives.
                if self._world.lookup(ns) is not None:
                    done = self._finalize_frame(arm, track)
                    self._pending_track[arm] = None
                    if done is not None:
                        to_publish.append(done)
                else:
                    self._pending_track[arm] = track
            for state, stamp_msg in to_publish:
                self._publish_debug_pose(arm, state, stamp_msg)  # Task 9
        except Exception as e:
            self.get_logger().error(f"track[{arm}] callback error: {e}")

    # ------------------------------------------------------------------
    # Stream B — gripper opening
    # ------------------------------------------------------------------
    def _joint_state_callback(self, msg: JointState) -> None:
        try:
            index = {name: i for i, name in enumerate(msg.name)}
            if not all(j in index for j in GRIPPER_JOINTS):
                return
            with self.lock:
                for joint in GRIPPER_JOINTS:
                    i = index[joint]
                    if i < len(msg.position):
                        self._grippers[joint] = float(msg.position[i])
        except Exception as e:
            self.get_logger().error(f"JointState callback error: {e}")

    # ------------------------------------------------------------------
    # Cameras — 3x CompressedImage @ camera_fps (aio pattern)
    # ------------------------------------------------------------------
    def _init_image_message_filters(self) -> None:
        t = self.topics
        # BEST_EFFORT so we receive from a best-effort usb_cam publisher (a
        # RELIABLE subscriber gets nothing from a best-effort publisher, while a
        # best-effort subscriber accepts both -- strictly the safer choice).
        sub_head = Subscriber(self, CompressedImage, t.camera_head,
                              qos_profile=self.qos_best_effort)
        sub_wrist_l = Subscriber(self, CompressedImage, t.camera_wrist_left,
                                 qos_profile=self.qos_best_effort)
        sub_wrist_r = Subscriber(self, CompressedImage, t.camera_wrist_right,
                                 qos_profile=self.qos_best_effort)
        self.image_sync = ApproximateTimeSynchronizer(
            [sub_head, sub_wrist_l, sub_wrist_r], queue_size=5, slop=0.05
        )
        self.image_sync.registerCallback(self._image_synchronized_callback)

    def _image_synchronized_callback(self, head, wrist_left, wrist_right) -> None:
        try:
            now = time.time_ns()
            if now - self.last_image_recv_time_ns < self.min_camera_interval_ns:
                return
            self.last_image_recv_time_ns = now
            self._images_recv(head, "image_head")
            self._images_recv(wrist_left, "image_wrist_left")
            self._images_recv(wrist_right, "image_wrist_right")
        except Exception as e:
            self.get_logger().error(f"Image synchronized callback error: {e}")

    def _images_recv(self, msg: CompressedImage, event_id: str) -> None:
        try:
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] != CAMERA_HEIGHT or frame.shape[1] != CAMERA_WIDTH:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            with self.lock:
                self.recv_images[event_id] = frame
                self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"recv image error ({event_id}): {e}")

    # ------------------------------------------------------------------
    # Accessors for robot.py (connect gating + per-frame vectors)
    # ------------------------------------------------------------------
    def left_valid(self) -> bool:
        with self.lock:
            s = self._eef_state["left"]
            return s is not None and s.valid

    def right_valid(self) -> bool:
        with self.lock:
            s = self._eef_state["right"]
            return s is not None and s.valid

    def grippers_valid(self) -> bool:
        with self.lock:
            return all(j in self._grippers for j in GRIPPER_JOINTS)

    def state_vector(self) -> Optional[np.ndarray]:
        with self.lock:
            left, right = self._eef_state["left"], self._eef_state["right"]
            if (
                left is None or right is None
                or not left.valid or not right.valid
                or not all(j in self._grippers for j in GRIPPER_JOINTS)
            ):
                return None
            return build_state_vector(
                left, right,
                self._grippers[GRIPPER_JOINTS[0]],
                self._grippers[GRIPPER_JOINTS[1]],
            )

    def quality_vector(self) -> Optional[np.ndarray]:
        with self.lock:
            left, right = self._eef_state["left"], self._eef_state["right"]
            if left is None or right is None:
                return None
            return build_quality_vector(left, right)

    # ------------------------------------------------------------------
    # Debug overlay — implemented in Task 9; keep no-op stubs until then
    # ------------------------------------------------------------------
    def _init_debug_publishers(self) -> None:
        if not self.publish_debug:
            self._debug_pubs = None
            self._debug_marker_pub = None
            return
        self._debug_pubs = {
            "left": self.create_publisher(PoseStamped, "/umi/debug/eef_left", 10),
            "right": self.create_publisher(PoseStamped, "/umi/debug/eef_right", 10),
        }
        self._debug_marker_pub = self.create_publisher(
            MarkerArray, "/umi/debug/markers", 10
        )
        self.get_logger().info("debug overlay ON: /umi/debug/eef_* + /umi/debug/markers")

    @staticmethod
    def _quality_color(state) -> ColorRGBA:
        # green = fresh compose; yellow = held (tracking or world gap);
        # red = never composed / long stale
        if state.world_fresh >= 1.0 and state.present >= 1.0:
            return ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.9)
        if state.valid:
            return ColorRGBA(r=0.9, g=0.8, b=0.1, a=0.9)
        return ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.9)

    def _publish_debug_pose(self, arm, state, stamp) -> None:
        if self._debug_pubs is None or not state.valid:
            return
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = "world"
        p7 = state.pose7
        ps.pose.position.x = float(p7[0])
        ps.pose.position.y = float(p7[1])
        ps.pose.position.z = float(p7[2])
        ps.pose.orientation.x = float(p7[3])
        ps.pose.orientation.y = float(p7[4])
        ps.pose.orientation.z = float(p7[5])
        ps.pose.orientation.w = float(p7[6])
        self._debug_pubs[arm].publish(ps)

        marker = Marker()
        marker.header = ps.header
        marker.ns = "umi_eef"
        marker.id = 0 if arm == "left" else 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = ps.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.color = self._quality_color(state)
        arr = MarkerArray()
        arr.markers.append(marker)
        self._debug_marker_pub.publish(arr)

    def destroy(self) -> None:
        super().destroy_node()
