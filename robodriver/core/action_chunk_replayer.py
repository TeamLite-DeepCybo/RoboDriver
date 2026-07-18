"""Action chunk 逐帧节拍回放器。

接收 (N, 128) 的 action chunk，在独立线程中按 FPS 节拍逐帧下发：
  1. 每帧取前 16 维（或补零/截断），映射为 leader_*.pos 命名 dict
  2. 调用 robot.send_action() 发布 MITCommand
  3. busy_wait 保证节拍

与 replayer.py 的核心区别：
  - replayer.py 从本地 parquet 逐行读取 action
  - 此类从内存中的 action_chunk (N×128 ndarray) 逐行下发
  - 支持中途替换 chunk（load_chunk）

LITE_JOINT_NAMES 顺序（前16维）:
  左臂7 + 右臂7 + 左夹爪 + 右夹爪
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

import logging_mp
from lerobot.robots import Robot

from robodriver.robots.utils import busy_wait
from .inference_client import ACTION_DIM, ROBOT_ACTION_DIM, server_action_to_robot

logger = logging_mp.getLogger(__name__)


class ActionChunkPublisher:

    # 前 16 维 → leader 关节名（顺序与 LITE_JOINT_NAMES 严格一致）
    _LITE_JOINT_NAMES: tuple = (
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
        "left_gripper",
        "right_gripper",
    )
    """按节拍逐帧下发 128 维 action chunk。

    使用示例::

        replayer = ActionChunkPublisher(robot, fps=30)
        replayer.start()
        replayer.load_chunk(action_chunk_128)
        replayer.wait_idle()       # 阻塞直到 chunk 回放完毕
        replayer.stop()
    """

    def __init__(self, robot: Robot, fps: int = 30):
        self.robot = robot
        self.fps = max(1, int(fps))
        self._frame_interval_s = 1.0 / self.fps

        # ---- 线程状态 ----
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # ---- chunk 状态 ----
        self._chunk: Optional[np.ndarray] = None      # (N, 128) float32
        self._chunk_index: int = 0
        self._chunk_request_id: str = ""
        self._last_action_dict: Optional[Dict[str, float]] = None
        self._lock = threading.Lock()

        # ---- 统计 ----
        self.frames_sent: int = 0

    # ------------------------------------------------------------------
    # Chunk 管理
    # ------------------------------------------------------------------
    def load_chunk(self, action_chunk: np.ndarray, request_id: str = "") -> None:
        """加载新的 action chunk，替换当前正在回放的 chunk。

        线程安全：可在回放线程运行期间调用。
        若当前 chunk 未耗尽，新 chunk 会覆盖旧 chunk（旧 chunk 剩余帧丢弃）。
        """
        arr = np.asarray(action_chunk, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"action_chunk must be 2D (N, D), got shape {arr.shape}"
            )
        if arr.shape[1] < ROBOT_ACTION_DIM:
            raise ValueError(
                f"action_chunk dim={arr.shape[1]} < {ROBOT_ACTION_DIM}, "
                f"cannot control robot"
            )

        with self._lock:
            self._chunk = arr[:, :ACTION_DIM].copy()  # 截断到 128 维
            self._chunk_index = 0
            self._chunk_request_id = request_id

        logger.info(
            f"[ActionChunkPublisher] chunk loaded: {arr.shape[0]} frames, "
            f"dim={arr.shape[1]} → {ACTION_DIM}, request_id={request_id}"
        )

    @property
    def remaining(self) -> int:
        """当前 chunk 剩余帧数。"""
        with self._lock:
            if self._chunk is None:
                return 0
            return max(0, self._chunk.shape[0] - self._chunk_index)

    @property
    def is_idle(self) -> bool:
        """当前无待回放帧。"""
        return self.remaining <= 0

    @property
    def total_frames_in_chunk(self) -> int:
        with self._lock:
            return 0 if self._chunk is None else self._chunk.shape[0]

    # ------------------------------------------------------------------
    # 回放线程
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动回放线程。幂等（重复调用无副作用）。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._replay_loop,
            name="ActionChunkPublisher",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[ActionChunkPublisher] started @ {self.fps} Hz")

    def stop(self) -> None:
        """停止回放线程。幂等。"""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info(
            f"[ActionChunkPublisher] stopped, total frames sent={self.frames_sent}"
        )

    def wait_idle(self, poll_interval_s: float = 0.01) -> None:
        """阻塞直到当前 chunk 耗尽。"""
        while self._running and not self.is_idle:
            time.sleep(poll_interval_s)

    # ------------------------------------------------------------------
    # 主回放循环
    # ------------------------------------------------------------------
    # 主回放循环
    # ------------------------------------------------------------------
    def _replay_loop(self) -> None:
        """主回放循环：逐帧下发 chunk，无 chunk 时持续发送保持指令。

        保持策略：
          - 从未收到过 chunk → 从 robot.get_observation() 取当前关节角
          - chunk 耗尽 → 持续发送最后一帧 action
        """
        while self._running:
            action_vector = self._next_action()
            if action_vector is None:
                # 无 chunk：发送保持指令（当前位置 或 最后一帧）
                hold = self._get_hold_action()
                if hold is not None:
                    try:
                        self.robot.send_action(hold)
                    except Exception:
                        pass
                busy_wait(self._frame_interval_s)
                continue

            start_t = time.perf_counter()

            try:
                action_dict = self._action_vector_to_dict(action_vector)
                self.robot.send_action(action_dict)
                self.frames_sent += 1
                self._last_action_dict = action_dict
            except Exception:
                logger.exception(
                    f"[ActionChunkPublisher] send_action failed at "
                    f"chunk_index={self._chunk_index - 1}"
                )
            dt_s = time.perf_counter() - start_t
            busy_wait(self._frame_interval_s - dt_s)

    def _next_action(self) -> Optional[np.ndarray]:
        """取出当前 chunk 的下一帧 128 维 action。无剩余时返回 None。"""
        with self._lock:
            if self._chunk is None:
                return None
            if self._chunk_index >= self._chunk.shape[0]:
                return None
            vec = self._chunk[self._chunk_index].copy()
            self._chunk_index += 1
            return vec

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 保持指令
    # ------------------------------------------------------------------
    def _get_hold_action(self):
        """获取保持指令：last action 或当前位置。

        优先使用最近一次有效的 chunk action（chunk 耗尽时保持最后一帧），
        从未收到过 chunk 时从 robot.get_observation() 读取当前位置。
        """
        if self._last_action_dict is not None:
            return self._last_action_dict
        try:
            obs = self.robot.get_observation()
            return {
                f"leader_{name}.pos": float(obs.get(f"follower_{name}.pos", 0.0))
                for name in self._LITE_JOINT_NAMES
            }
        except Exception:
            logger.warning("[ActionChunkPublisher] failed to read current position for hold")
            return None

    # 向量格式转换：128 维 → leader 命名 dict
    # ------------------------------------------------------------------
    @classmethod
    def _action_vector_to_dict(cls, vec_128: np.ndarray) -> Dict[str, float]:
        """128 维 action 向量 → robot.send_action() 所需的命名 dict。

        流程：
          1. 取前 16 维（server_action_to_robot）
          2. 映射为 {"leader_<joint_name>.pos": float, ...}
        """
        vec_16 = server_action_to_robot(vec_128)
        return {
            f"leader_{name}.pos": float(vec_16[i])
            for i, name in enumerate(cls._LITE_JOINT_NAMES)
        }
