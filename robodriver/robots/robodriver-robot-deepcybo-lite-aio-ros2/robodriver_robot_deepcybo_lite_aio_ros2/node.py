"""DeepCybo Lite — ROS2 订阅 / 同步 / 拼向量 / 回放发布。

机械臂向量布局（14 维，与 bar_bringup_lite/config/lite_hardware.yaml 一致）::
    left arm 7 | right arm 7

机械臂话题::
    observation: sensor_msgs/JointState @ /slave/lite/joint_states
    action:      bar_msgs/MITCommand   @ /slave/remote_policy_controller/command

限频设计::
    control_fps / camera_fps 默认均为 30Hz，与 Record 主循环及相机帧率一致
"""

import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node as ROS2Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState

# 保留：后续可能用末端位姿 (PoseStamped) 扩展遥操
from geometry_msgs.msg import PoseStamped  # noqa: F401

import logging_mp

from .config import ARM_JOINT_NAMES, DeepcyboLiteRos2Topics

try:
    from bar_msgs.msg import MITCommand
except ImportError as exc:  # pragma: no cover - depends on sourced bar_ws overlay
    MITCommand = None
    _BAR_MSGS_IMPORT_ERROR = exc
else:
    _BAR_MSGS_IMPORT_ERROR = None

CONNECT_TIMEOUT_FRAME = 10
STATE_DIM = len(ARM_JOINT_NAMES)  # 14

# 解码尺寸（与 Lite包使用说明 / 中台出图一致）
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

logger = logging_mp.get_logger(__name__)


class JointVectorError(Exception):
    """机械臂消息无法转换为 canonical 14 维向量。"""


# 兼容旧文档 / 排障脚本里引用的异常名
JointStateLengthError = JointVectorError


class DeepcyboLiteAioRos2RobotNode(ROS2Node):
    def __init__(
        self,
        topics: Optional[DeepcyboLiteRos2Topics] = None,
        control_fps: int = 30,
        camera_fps: int = 30,
        command_stiffness: float = 20.0,
        command_damping: float = 2.0,
    ):
        if MITCommand is None:
            raise ImportError(
                "Cannot import bar_msgs.msg.MITCommand. Source the bar_ws ROS2 "
                "overlay before starting RoboDriver, e.g. "
                "`source /home/stvli/Desktop/bar_ws/install/setup.bash`."
            ) from _BAR_MSGS_IMPORT_ERROR

        super().__init__("deepcybo_lite_ros2_driver")
        self.topics = topics or DeepcyboLiteRos2Topics()
        self.control_fps = control_fps
        self.camera_fps = camera_fps
        self.command_stiffness = float(command_stiffness)
        self.command_damping = float(command_damping)

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
        self.publisher_command = self.create_publisher(MITCommand, t.command, self.qos)
        self.create_subscription(
            JointState,
            t.joint_states,
            self.joint_state_callback,
            self.qos_best_effort,
        )
        self.create_subscription(
            MITCommand,
            t.command,
            self.command_callback,
            self.qos,
        )

        self.last_main_send_time_ns = 0
        self.last_follow_send_time_ns = 0
        self.last_image_send_time_ns = 0
        self.min_control_interval_ns = int(1e9 / max(control_fps, 1))
        self.min_camera_interval_ns = int(1e9 / max(camera_fps, 1))

        # 关节向量是否处于「可落盘」健康态（长度恢复前保持 False）
        self._follower_arm_ok = False
        self._leader_arm_ok = False

        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_leader: Dict[str, np.ndarray] = {}
        self.recv_follower: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}
        self.recv_leader_status: Dict[str, int] = {}
        self.recv_follower_status: Dict[str, int] = {}

        self.lock = threading.Lock()

        self._init_image_message_filters()

        logger.info(
            "[DeepCybo Lite] node ready | joint_states=%s command=%s "
            "control_fps=%s camera_fps=%s",
            t.joint_states,
            t.command,
            control_fps,
            camera_fps,
        )

    # ------------------------------------------------------------------
    # 机械臂消息 -> canonical 14 维向量
    # ------------------------------------------------------------------
    @staticmethod
    def _vector_from_joint_state(msg: JointState, label: str) -> np.ndarray:
        if len(msg.name) != len(msg.position):
            raise JointVectorError(
                f"{label}: JointState.name 长度={len(msg.name)}, "
                f"position 长度={len(msg.position)}；本帧丢弃"
            )

        position_by_name = {
            name: float(pos)
            for name, pos in zip(msg.name, msg.position, strict=False)
        }
        missing = [name for name in ARM_JOINT_NAMES if name not in position_by_name]
        if missing:
            raise JointVectorError(
                f"{label}: JointState 缺少 canonical arm joints: {missing}；本帧丢弃"
            )

        return np.asarray(
            [position_by_name[name] for name in ARM_JOINT_NAMES],
            dtype=np.float32,
        )

    @staticmethod
    def _vector_from_mit_command(msg, label: str) -> np.ndarray:
        if len(msg.joint_names) != len(msg.position):
            raise JointVectorError(
                f"{label}: MITCommand.joint_names 长度={len(msg.joint_names)}, "
                f"position 长度={len(msg.position)}；本帧丢弃"
            )

        position_by_name = {
            name: float(pos)
            for name, pos in zip(msg.joint_names, msg.position, strict=False)
        }
        missing = [name for name in ARM_JOINT_NAMES if name not in position_by_name]
        if missing:
            raise JointVectorError(
                f"{label}: MITCommand 缺少 canonical arm joints: {missing}；本帧丢弃"
            )

        return np.asarray(
            [position_by_name[name] for name in ARM_JOINT_NAMES],
            dtype=np.float32,
        )

    def _invalidate_arm_state(self, stream: str) -> None:
        """长度异常：清空缓存，上层 connect/Record 拿不到有效向量。"""
        comp = "follower_arms" if stream == "follower" else "leader_arms"
        with self.lock:
            if stream == "follower":
                self.recv_follower.pop(comp, None)
                self.recv_follower_status[comp] = 0
                self._follower_arm_ok = False
            else:
                self.recv_leader.pop(comp, None)
                self.recv_leader_status[comp] = 0
                self._leader_arm_ok = False

    def _commit_arm_state(self, stream: str, merged: np.ndarray) -> None:
        comp = "follower_arms" if stream == "follower" else "leader_arms"
        was_ok = self._follower_arm_ok if stream == "follower" else self._leader_arm_ok
        with self.lock:
            if stream == "follower":
                self.recv_follower[comp] = merged
                self.recv_follower_status[comp] = CONNECT_TIMEOUT_FRAME
                self._follower_arm_ok = True
            else:
                self.recv_leader[comp] = merged
                self.recv_leader_status[comp] = CONNECT_TIMEOUT_FRAME
                self._leader_arm_ok = True
        if not was_ok:
            self.get_logger().info(f"[{stream}] 机械臂向量已恢复，继续写入缓存")

    # ------------------------------------------------------------------
    # Observation — /slave/lite/joint_states (JointState)
    # ------------------------------------------------------------------
    def joint_state_callback(self, msg: JointState) -> None:
        try:
            now = time.time_ns()
            if now - self.last_follow_send_time_ns < self.min_control_interval_ns:
                return
            self.last_follow_send_time_ns = now

            merged = self._vector_from_joint_state(msg, "follower")
            self._commit_arm_state("follower", merged)
        except JointVectorError as e:
            self.get_logger().warning(str(e))
            self._invalidate_arm_state("follower")
        except Exception as e:
            self.get_logger().error(f"JointState callback error: {e}")

    # ------------------------------------------------------------------
    # Action — /slave/remote_policy_controller/command (MITCommand)
    # ------------------------------------------------------------------
    def command_callback(self, msg) -> None:
        try:
            now = time.time_ns()
            if now - self.last_main_send_time_ns < self.min_control_interval_ns:
                return
            self.last_main_send_time_ns = now

            merged = self._vector_from_mit_command(msg, "leader")
            self._commit_arm_state("leader", merged)
        except JointVectorError as e:
            self.get_logger().warning(str(e))
            self._invalidate_arm_state("leader")
        except Exception as e:
            self.get_logger().error(f"MITCommand callback error: {e}")

    # ------------------------------------------------------------------
    # 相机 — 3 路 CompressedImage @ camera_fps
    # ------------------------------------------------------------------
    def _init_image_message_filters(self) -> None:
        t = self.topics
        sub_head = Subscriber(self, CompressedImage, t.camera_head)
        sub_wrist_l = Subscriber(self, CompressedImage, t.camera_wrist_left)
        sub_wrist_r = Subscriber(self, CompressedImage, t.camera_wrist_right)

        self.image_sync = ApproximateTimeSynchronizer(
            [sub_head, sub_wrist_l, sub_wrist_r],
            queue_size=5,
            slop=0.05,
        )
        self.image_sync.registerCallback(self.image_synchronized_callback)

    def image_synchronized_callback(
        self,
        head: CompressedImage,
        wrist_left: CompressedImage,
        wrist_right: CompressedImage,
    ) -> None:
        try:
            now = time.time_ns()
            if now - self.last_image_send_time_ns < self.min_camera_interval_ns:
                return
            self.last_image_send_time_ns = now

            self.images_recv(head, "image_head")
            self.images_recv(wrist_left, "image_wrist_left")
            self.images_recv(wrist_right, "image_wrist_right")
        except Exception as e:
            self.get_logger().error(f"Image synchronized callback error: {e}")

    def images_recv(self, msg: CompressedImage, event_id: str, encoding: str = "jpeg") -> None:
        try:
            if "image" not in event_id:
                return
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            frame = None
            if encoding in ("jpeg", "jpg", "jpe", "bmp", "webp", "png"):
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif encoding == "bgr8":
                frame = img_array.reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3)).copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif encoding == "rgb8":
                frame = img_array.reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3))

            if frame is not None:
                if frame.shape[0] != CAMERA_HEIGHT or frame.shape[1] != CAMERA_WIDTH:
                    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                with self.lock:
                    self.recv_images[event_id] = frame
                    self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"recv image error ({event_id}): {e}")

    # ------------------------------------------------------------------
    # 回放
    # ------------------------------------------------------------------
    def ros_replay(self, array: np.ndarray) -> None:
        try:
            vec = np.asarray(array, dtype=np.float32).flatten()
            if vec.shape[0] != STATE_DIM:
                raise ValueError(f"replay action dim {vec.shape[0]}, expected {STATE_DIM}")

            def norm(v: float) -> float:
                v = float(v)
                if np.isnan(v) or np.isinf(v):
                    return 0.0
                return round(v, 3)

            n = STATE_DIM
            msg = MITCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = list(ARM_JOINT_NAMES)
            msg.position = [norm(v) for v in vec]
            msg.velocity = [0.0] * n
            msg.effort = [0.0] * n
            msg.stiffness = [self.command_stiffness] * n
            msg.damping = [self.command_damping] * n
            self.publisher_command.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error during replay: {e}")
            raise

    def destroy(self) -> None:
        super().destroy_node()


# robot.py 尚未改名前，保留旧类名别名
GALAXEALITEAIORos2RobotNode = DeepcyboLiteAioRos2RobotNode

# ---------------------------------------------------------------------------
# Galaxea 原版 leader 同步（供对照，DeepCybo Lite 当前已改为 BAR Lite 原生链路）
# ---------------------------------------------------------------------------
# 旧版 4 路 JointState:
#   /motion_target/target_joint_state_arm_left
#   /motion_target/target_joint_state_arm_right
#   /motion_target/target_position_gripper_left
#   /motion_target/target_position_gripper_right
# 当前 BAR Lite:
#   /slave/lite/joint_states
#   /slave/remote_policy_controller/command
