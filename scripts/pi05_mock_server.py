#!/usr/bin/env python3
"""PI05 推理服务 Mock — 供客户端联调使用。

模拟 PI05 推理服务端行为：
  1. 接收 POST /api/v1/infer 请求
  2. 以当前 state 为基础，叠加平滑正弦运动生成 action chunk
  3. 前 16 维为有意义轨迹，后 112 维补 0
  4. 可配置模拟推理延迟

启动方式::

    pip install fastapi uvicorn
    python scripts/pi05_mock_server.py --port 9090 --delay-ms 50

协议详情见: PI05_INFERENCE_DEPLOYMENT_PIPELINE_DESIGN.md §3
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 维度常量（与客户端 inference_client.py 保持一致）
# ---------------------------------------------------------------------------
STATE_DIM = 16
ACTION_DIM = 128
ROBOT_ACTION_DIM = 16  # 前 16 维有实际运动

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="PI05 Mock Inference Server", version="0.1.0")

# 全局状态（模拟推理计数器，用于生成连续轨迹）
_tick: int = 0
_start_time: float = time.time()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class ObservationImages(BaseModel):
    image_head: str = ""
    image_wrist_left: str = ""
    image_wrist_right: str = ""


class Observation(BaseModel):
    state: List[float] = Field(..., min_length=STATE_DIM, max_length=STATE_DIM)
    images: ObservationImages = Field(default_factory=ObservationImages)


class RequestMetadata(BaseModel):
    fps: int = 30
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    chunk_size: int = 50
    image_width: int = 640
    image_height: int = 480
    timestamp_unix: float = 0.0


class InferenceRequest(BaseModel):
    request_id: str = ""
    robot_type: str = "deepcybo-lite-aio-ros2"
    prompt: str = ""
    observation: Observation
    metadata: RequestMetadata = Field(default_factory=RequestMetadata)


class ResponseMetadata(BaseModel):
    fps: int = 30
    inference_time_ms: float = 0.0
    model_version: str = "pi05-v1.0-mock"
    timestamp_unix: float = 0.0


class InferenceResponse(BaseModel):
    request_id: str = ""
    status: str = "ok"
    action_chunk: List[List[float]] = Field(default_factory=list)
    chunk_size: int = 0
    action_dim: int = ACTION_DIM
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    error_code: str | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Mock 推理逻辑
# ---------------------------------------------------------------------------
def _generate_mock_action_chunk(
    state: np.ndarray,
    chunk_size: int,
    fps: int,
    motion_scale: float = 0.1,
) -> np.ndarray:
    """以当前 state 为基础，生成平滑正弦扰动的 action chunk。

    Args:
        state: (16,) 当前关节状态
        chunk_size: 要生成的帧数
        fps: 控制频率
        motion_scale: 运动幅度缩放因子

    Returns:
        (chunk_size, 128) float32 action chunk
    """
    global _tick

    state = np.asarray(state, dtype=np.float32).flatten()
    if state.shape[0] < ROBOT_ACTION_DIM:
        padded = np.zeros(ROBOT_ACTION_DIM, dtype=np.float32)
        padded[: state.shape[0]] = state
        state = padded

    chunk = np.zeros((chunk_size, ACTION_DIM), dtype=np.float32)

    for frame_idx in range(chunk_size):
        t = (_tick + frame_idx) / max(fps, 1)

        # 前 16 维：在 state 基础上叠加平滑正弦运动
        for joint_idx in range(ROBOT_ACTION_DIM):
            phase = 0.37 * joint_idx
            slow = math.sin(2.0 * math.pi * 0.13 * t + phase)
            fast = 0.25 * math.sin(2.0 * math.pi * 0.43 * t + phase * 0.5)
            chunk[frame_idx, joint_idx] = (
                float(state[joint_idx]) + motion_scale * (slow + fast)
            )

        # 后 112 维：补 0（预留灵巧手等）
        # chunk[frame_idx, 16:128] already zeros

    _tick += chunk_size
    return chunk


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": "pi05-v1.0-mock",
        "gpu_available": False,
        "mock_mode": True,
        "uptime_s": time.time() - _start_time,
        "requests_processed": _tick,
    }


@app.post("/api/v1/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest, delay_ms: float = 0.0) -> InferenceResponse:
    """接收推理请求，生成 mock action chunk。

    查询参数 ``delay_ms`` 可用于模拟推理延迟（默认从命令行 --delay-ms 设置）。
    """
    t0 = time.perf_counter()

    # ---- 1. 校验 ----
    state = np.array(req.observation.state, dtype=np.float32)
    if state.shape[0] < STATE_DIM:
        return InferenceResponse(
            request_id=req.request_id,
            status="error",
            error_code="INVALID_STATE_DIM",
            error_message=(
                f"state dim={state.shape[0]} < required {STATE_DIM}"
            ),
            metadata=ResponseMetadata(timestamp_unix=time.time()),
        )

    # ---- 2. 模拟推理延迟 ----
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)

    # ---- 3. 生成 mock action chunk ----
    chunk_size = req.metadata.chunk_size or 50
    fps = req.metadata.fps or 30
    action_chunk = _generate_mock_action_chunk(state, chunk_size, fps)

    dt_ms = (time.perf_counter() - t0) * 1000.0

    # ---- 4. 返回 ----
    print(
        f"[mock] request_id={req.request_id} "
        f"prompt='{req.prompt[:50]}' "
        f"chunk=({chunk_size}, {ACTION_DIM}) "
        f"inference={dt_ms:.1f}ms"
    )

    return InferenceResponse(
        request_id=req.request_id,
        status="ok",
        action_chunk=action_chunk.tolist(),
        chunk_size=chunk_size,
        action_dim=ACTION_DIM,
        metadata=ResponseMetadata(
            fps=fps,
            inference_time_ms=round(dt_ms, 1),
            model_version="pi05-v1.0-mock",
            timestamp_unix=time.time(),
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PI05 Mock Inference Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9090, help="Bind port")
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=50.0,
        help="Simulated inference delay in milliseconds",
    )
    args = parser.parse_args()

    import uvicorn

    # 将 delay_ms 注入到 infer 端点（通过 FastAPI dependency）
    from fastapi import Query

    # Monkey-patch: 为 infer 端点注入默认 delay_ms
    # 简单方案：用闭包捕获
    _delay_ms = args.delay_ms

    # 替换 infer 端点以绑定默认 delay_ms
    app.router.routes.clear()

    @app.get("/api/v1/health")
    async def _health():
        return await health()

    @app.post("/api/v1/infer", response_model=InferenceResponse)
    async def _infer(
        req: InferenceRequest,
        delay_ms: float = Query(_delay_ms, alias="delay_ms"),
    ):
        return await infer(req, delay_ms=delay_ms)

    print(f"[PI05 Mock Server] starting on {args.host}:{args.port}")
    print(f"[PI05 Mock Server] mock delay: {_delay_ms}ms")
    print(f"[PI05 Mock Server] action dim: {ACTION_DIM}")
    print(f"[PI05 Mock Server] endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/api/v1/health")
    print(f"  POST http://{args.host}:{args.port}/api/v1/infer")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
