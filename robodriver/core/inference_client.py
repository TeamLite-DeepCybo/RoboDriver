"""PI05 推理服务 HTTP 通信客户端。

职责：
  1. 管理 aiohttp HTTP 会话与超时
  2. 将 observation 序列化为推理请求（图片 Base64 编码）
  3. POST /api/v1/infer，解析响应
  4. 校验 action 维度（期望 128，实际不足时补零/截断）
  5. Health check

协议约定：
  - 请求 state 维度: 16（机器人实际输出）
  - 响应 action 维度: 128（前 16 对齐 LITE_JOINT_NAMES，后 112 预留）
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import cv2
import numpy as np

import logging_mp

logger = logging_mp.getLogger(__name__)

# ---------------------------------------------------------------------------
# 维度常量
# ---------------------------------------------------------------------------
STATE_DIM = 16            # 请求中 observation.state 的维度
ACTION_DIM = 128          # 响应中 action_chunk 每帧的期望维度
ROBOT_ACTION_DIM = 16     # 实际下发到机器人的维度（LITE_JOINT_NAMES）


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
@dataclass
class InferenceRequest:
    """推理请求 — 客户端 → 服务端。"""
    request_id: str
    robot_type: str                                    # "deepcybo-lite-aio-ros2"
    prompt: str                                        # 任务指令
    state: np.ndarray                                  # (16,) float32
    images: Dict[str, np.ndarray]                      # {"image_head": (H,W,3) RGB, ...}
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """推理响应 — 服务端 → 客户端。"""
    request_id: str
    status: str                                        # "ok" | "error"
    action_chunk: Optional[np.ndarray] = None           # (N, 128) float32 或 None
    chunk_size: int = 0
    action_dim: int = ACTION_DIM
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.status == "ok" and self.action_chunk is not None

    @property
    def is_error(self) -> bool:
        return not self.is_ok


# ---------------------------------------------------------------------------
# InferenceClient
# ---------------------------------------------------------------------------
class InferenceClient:
    """PI05 推理服务 HTTP 客户端。

    使用 aiohttp 进行异步 HTTP 通信。所有图片编码为 Base64 JPEG 嵌入 JSON。
    """

    # 服务端 action 维度容差：允许的实际维度范围
    MIN_ACTION_DIM = ROBOT_ACTION_DIM   # 至少 16 维才能控制机器人
    MAX_ACTION_DIM = 2048               # 硬上限防止异常数据

    def __init__(
        self,
        server_url: str = "http://192.168.1.100:9090",
        timeout_s: float = 10.0,
        connect_timeout_s: float = 3.0,
    ):
        self.server_url = server_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(
            total=timeout_s,
            connect=connect_timeout_s,
        )
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            logger.info(f"[InferenceClient] session started → {self.server_url}")

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
            logger.info("[InferenceClient] session closed")

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------
    async def health_check(self) -> Dict[str, Any]:
        """GET /api/v1/health → {"status": "ok", "model_loaded": "...", ...}"""
        async with self._session.get(f"{self.server_url}/api/v1/health") as resp:
            return await resp.json()

    # ------------------------------------------------------------------
    # 图片编码
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image(img: np.ndarray, quality: int = 85) -> str:
        """RGB ndarray → Base64 JPEG 字符串。"""
        _, jpeg = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        return base64.b64encode(jpeg.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------
    # 请求构造
    # ------------------------------------------------------------------
    def build_request_payload(self, req: InferenceRequest) -> Dict[str, Any]:
        """将 InferenceRequest 转为 JSON-serializable dict。"""
        return {
            "request_id": req.request_id,
            "robot_type": req.robot_type,
            "prompt": req.prompt,
            "observation": {
                "state": req.state.astype(np.float32).flatten().tolist(),
                "images": {
                    key: self._encode_image(img)
                    for key, img in req.images.items()
                },
            },
            "metadata": {
                "fps": req.metadata.get("fps", 30),
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "chunk_size": req.metadata.get("chunk_size", 50),
                "image_width": req.metadata.get("image_width", 640),
                "image_height": req.metadata.get("image_height", 480),
                "timestamp_unix": time.time(),
            },
        }

    # ------------------------------------------------------------------
    # 响应解析 + 维度校验
    # ------------------------------------------------------------------
    def parse_response(self, raw: Dict[str, Any]) -> InferenceResponse:
        """解析服务端 JSON 响应，并校验 action 维度。

        维度处理规则:
          - 期望 128 维；服务端返回不足 128 维时补零，超过时截断
          - 服务端返回 < 16 维 → 视为 error
          - 服务端返回 > MAX_ACTION_DIM → 视为 error
        """
        status = raw.get("status", "error")

        if status == "error":
            return InferenceResponse(
                request_id=raw.get("request_id", ""),
                status="error",
                error_code=raw.get("error_code", "UNKNOWN"),
                error_message=raw.get("error_message", "Unknown server error"),
                metadata=raw.get("metadata", {}),
            )

        # 解析 action_chunk
        raw_chunk = raw.get("action_chunk", [])
        if not raw_chunk:
            return InferenceResponse(
                request_id=raw.get("request_id", ""),
                status="error",
                error_code="EMPTY_CHUNK",
                error_message="Server returned empty action_chunk",
                metadata=raw.get("metadata", {}),
            )

        raw_arr = np.array(raw_chunk, dtype=np.float32)
        if raw_arr.ndim != 2:
            return InferenceResponse(
                request_id=raw.get("request_id", ""),
                status="error",
                error_code="INVALID_CHUNK_SHAPE",
                error_message=f"action_chunk must be 2D, got shape {raw_arr.shape}",
                metadata=raw.get("metadata", {}),
            )

        server_dim = raw_arr.shape[1]

        # 维度校验
        if server_dim < self.MIN_ACTION_DIM:
            return InferenceResponse(
                request_id=raw.get("request_id", ""),
                status="error",
                error_code="ACTION_DIM_TOO_SMALL",
                error_message=(
                    f"Server action dim={server_dim} < min={self.MIN_ACTION_DIM}"
                ),
                metadata=raw.get("metadata", {}),
            )

        if server_dim > self.MAX_ACTION_DIM:
            return InferenceResponse(
                request_id=raw.get("request_id", ""),
                status="error",
                error_code="ACTION_DIM_TOO_LARGE",
                error_message=(
                    f"Server action dim={server_dim} > max={self.MAX_ACTION_DIM}"
                ),
                metadata=raw.get("metadata", {}),
            )

        # 维度对齐：补零或截断到 ACTION_DIM (128)
        if server_dim < ACTION_DIM:
            logger.warning(
                f"[InferenceClient] server action dim={server_dim} < {ACTION_DIM}, "
                f"zero-padding to {ACTION_DIM}"
            )
            padded = np.zeros((raw_arr.shape[0], ACTION_DIM), dtype=np.float32)
            padded[:, :server_dim] = raw_arr
            action_chunk = padded
        elif server_dim > ACTION_DIM:
            logger.warning(
                f"[InferenceClient] server action dim={server_dim} > {ACTION_DIM}, "
                f"truncating to {ACTION_DIM}"
            )
            action_chunk = raw_arr[:, :ACTION_DIM].copy()
        else:
            action_chunk = raw_arr.copy()

        return InferenceResponse(
            request_id=raw.get("request_id", ""),
            status="ok",
            action_chunk=action_chunk,
            chunk_size=raw.get("chunk_size", action_chunk.shape[0]),
            action_dim=ACTION_DIM,
            metadata=raw.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # 推理请求
    # ------------------------------------------------------------------
    async def infer(self, req: InferenceRequest) -> InferenceResponse:
        """发送推理请求并解析响应。

        Raises:
            aiohttp.ClientError: 网络层错误
            asyncio.TimeoutError: 请求超时
        """
        payload = self.build_request_payload(req)

        logger.info(
            f"[InferenceClient] POST /api/v1/infer request_id={req.request_id} "
            f"prompt='{req.prompt[:60]}...' images={list(req.images.keys())}"
        )

        async with self._session.post(
            f"{self.server_url}/api/v1/infer",
            json=payload,
        ) as resp:
            raw = await resp.json()

        result = self.parse_response(raw)

        if result.is_ok:
            logger.info(
                f"[InferenceClient] response OK: chunk=({result.chunk_size}, {result.action_dim}) "
                f"inference={result.metadata.get('inference_time_ms', '?')}ms"
            )
        else:
            logger.error(
                f"[InferenceClient] response ERROR: {result.error_code} — "
                f"{result.error_message}"
            )

        return result


# ---------------------------------------------------------------------------
# 维度工具函数（供其他模块引用）
# ---------------------------------------------------------------------------
def server_action_to_robot(server_vec: np.ndarray) -> np.ndarray:
    """将服务端返回的 action 向量（期望 128 维）转为机器人可用的 16 维向量。

    - 维度 == ROBOT_ACTION_DIM: 直接返回
    - 维度 >  ROBOT_ACTION_DIM: 取前 16 维
    - 维度 <  ROBOT_ACTION_DIM: 后补 0
    """
    vec = np.asarray(server_vec, dtype=np.float32).flatten()
    if vec.shape[0] == ROBOT_ACTION_DIM:
        return vec.copy()
    if vec.shape[0] > ROBOT_ACTION_DIM:
        return vec[:ROBOT_ACTION_DIM].copy()
    # vec.shape[0] < ROBOT_ACTION_DIM
    padded = np.zeros(ROBOT_ACTION_DIM, dtype=np.float32)
    padded[:vec.shape[0]] = vec
    return padded
