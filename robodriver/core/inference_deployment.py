"""PI05 推理部署同步主循环。

职责：
  1. 协调 InferenceClient（网络）与 ActionChunkReplayer（本地执行）
  2. 同步主循环: 采集 obs → 请求推理 → 等待响应 → 回放 chunk → 重新采集
  3. 异常处理: 超时重试、错误降级、优雅退出

用法::

    loop = InferenceDeploymentLoop(robot, server_url="http://...", prompt="...")
    asyncio.run(loop.run())
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import numpy as np

import logging_mp
from lerobot.robots import Robot

from .action_chunk_replayer import ActionChunkReplayer
from .inference_client import (
    ACTION_DIM,
    STATE_DIM,
    InferenceClient,
    InferenceRequest,
    InferenceResponse,
)

logger = logging_mp.get_logger(__name__)

# 最大连续推理失败次数（超过后退出主循环）
MAX_CONSECUTIVE_INFER_FAILURES = 5


class InferenceDeploymentLoop:
    """PI05 推理部署主循环 — 同步请求-回放模式。

    主循环逻辑::

        while running:
            1. 采集 observation
            2. POST /api/v1/infer → 阻塞等待
            3. 收到 chunk → load 到 replayer
            4. 等待 replayer 回放完毕
            5. 回到 1
    """

    def __init__(
        self,
        robot: Robot,
        server_url: str = "http://192.168.1.100:9090",
        fps: int = 30,
        chunk_size: int = 50,
        prompt: str = "",
    ):
        self.robot = robot
        self.fps = max(1, int(fps))
        self.chunk_size = max(1, int(chunk_size))
        self.prompt = prompt

        self.client = InferenceClient(server_url)
        self.replayer = ActionChunkReplayer(robot, fps=fps)

        self._running = False
        self._request_counter: int = 0
        self._consecutive_failures: int = 0
        self._total_frames_sent: int = 0
        self._total_requests: int = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """启动部署主循环（阻塞直到主动停止或异常退出）。"""
        # 1. 连接服务端 + health check
        await self.client.start()
        health = await self.client.health_check()

        if health.get("status") != "ok":
            raise RuntimeError(
                f"Inference server unhealthy at {self.client.server_url}: {health}"
            )

        logger.info(
            f"[InferenceDeployment] server healthy: model={health.get('model_loaded', '?')} "
            f"gpu={health.get('gpu_available', False)} "
            f"mock={health.get('mock_mode', False)}"
        )

        # 2. 启动回放线程
        self.replayer.start()
        self._running = True

        logger.info(
            f"[InferenceDeployment] loop started: "
            f"server={self.client.server_url} fps={self.fps} "
            f"chunk_size={self.chunk_size} prompt='{self.prompt[:60]}'"
        )

        # 3. 主循环
        try:
            while self._running:
                await self._tick()
        except KeyboardInterrupt:
            logger.info("[InferenceDeployment] interrupted by user")
        finally:
            await self._shutdown()

        logger.info(
            f"[InferenceDeployment] loop ended: "
            f"requests={self._total_requests} frames={self.replayer.frames_sent}"
        )

    async def stop(self) -> None:
        """外部停止信号（可在其他协程中调用）。"""
        self._running = False

    async def _shutdown(self) -> None:
        """清理资源。"""
        self._running = False
        self.replayer.stop()
        await self.client.stop()

    # ------------------------------------------------------------------
    # 主循环 tick
    # ------------------------------------------------------------------
    async def _tick(self) -> None:
        """一次完整的采集→推理→回放周期。

        时序::

            obs = capture()
            resp = await infer(obs)
            if resp.ok:
                replayer.load_chunk(resp.action_chunk)
                replayer.wait_idle()
            else:
                handle_error(resp)
        """
        # ---- 1. 采集 observation ----
        obs = self._capture_observation()

        # ---- 2. 推理请求 ----
        resp = await self._infer_with_retry(obs, max_retries=3)

        if resp.is_error:
            self._consecutive_failures += 1
            logger.error(
                f"[InferenceDeployment] inference failed ({self._consecutive_failures}/"
                f"{MAX_CONSECUTIVE_INFER_FAILURES}): {resp.error_code} — {resp.error_message}"
            )

            if self._consecutive_failures >= MAX_CONSECUTIVE_INFER_FAILURES:
                logger.critical(
                    f"[InferenceDeployment] too many consecutive failures, stopping"
                )
                self._running = False

            # 失败后等待 1s 再重试（让服务端/网络恢复）
            await asyncio.sleep(1.0)
            return

        # ---- 3. 加载 chunk 并等待回放完毕 ----
        self._consecutive_failures = 0
        self._total_requests += 1

        self.replayer.load_chunk(resp.action_chunk, resp.request_id)

        logger.info(
            f"[InferenceDeployment] chunk #{self._total_requests}: "
            f"{resp.chunk_size} frames, "
            f"inference={resp.metadata.get('inference_time_ms', '?')}ms, "
            f"replaying..."
        )

        # 阻塞等待当前 chunk 回放完毕
        self.replayer.wait_idle()

        logger.info(
            f"[InferenceDeployment] chunk #{self._total_requests} replay done, "
            f"total frames sent={self.replayer.frames_sent}"
        )

    # ------------------------------------------------------------------
    # Observation 采集
    # ------------------------------------------------------------------
    def _capture_observation(self) -> Dict[str, Any]:
        """从 robot.get_observation() 抓取当前观测。

        Returns:
            obs dict，格式与 robot.py:get_observation() 一致:
              {
                "follower_left_shoulder_pitch.pos": 0.12,
                ...
                "image_head": ndarray(H,W,3),
                "image_wrist_left": ndarray(H,W,3),
                "image_wrist_right": ndarray(H,W,3),
              }
        """
        return self.robot.get_observation()

    def _extract_state_and_images(
        self, obs: Dict[str, Any]
    ) -> tuple:
        """从 observation dict 中提取 state 向量和 images 字典。

        使用 ActionChunkReplayer 的 _LITE_JOINT_NAMES 保证顺序一致。
        """
        _NAMES = ActionChunkReplayer._LITE_JOINT_NAMES

        state = np.array(
            [float(obs.get(f"follower_{name}.pos", 0.0)) for name in _NAMES],
            dtype=np.float32,
        )

        images: Dict[str, np.ndarray] = {}
        for cam_name in ("image_head", "image_wrist_left", "image_wrist_right"):
            img = obs.get(cam_name)
            if img is not None and isinstance(img, np.ndarray):
                images[cam_name] = img

        return state, images

    # ------------------------------------------------------------------
    # 推理请求 + 重试
    # ------------------------------------------------------------------
    async def _infer_with_retry(
        self,
        obs: Dict[str, Any],
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
    ) -> InferenceResponse:
        """发送推理请求，带超时重试。"""
        state, images = self._extract_state_and_images(obs)

        self._request_counter += 1
        req = InferenceRequest(
            request_id=(
                f"deepcybo-lite-{int(time.time())}-{self._request_counter:04d}"
            ),
            robot_type="deepcybo-lite-aio-ros2",
            prompt=self.prompt,
            state=state,
            images=images,
            metadata={
                "fps": self.fps,
                "chunk_size": self.chunk_size,
            },
        )

        last_error: Optional[InferenceResponse] = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = await self.client.infer(req)
                if resp.is_ok:
                    return resp
                last_error = resp
            except asyncio.TimeoutError:
                logger.warning(
                    f"[InferenceDeployment] request timeout "
                    f"(attempt {attempt}/{max_retries})"
                )
                last_error = InferenceResponse(
                    request_id=req.request_id,
                    status="error",
                    error_code="TIMEOUT",
                    error_message=f"Request timed out after {self.client._timeout.total}s",
                )
            except Exception as e:
                logger.warning(
                    f"[InferenceDeployment] request failed "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                last_error = InferenceResponse(
                    request_id=req.request_id,
                    status="error",
                    error_code="NETWORK_ERROR",
                    error_message=str(e),
                )

            if attempt < max_retries:
                await asyncio.sleep(retry_delay_s)

        return last_error  # type: ignore[return-value]
