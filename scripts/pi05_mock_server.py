#!/usr/bin/env python3
"""PI05 推理服务 Mock — 回环测试用。

行为：
  1. 解码请求中的三路 Base64 JPEG 图片，用 OpenCV 在显示屏上展示
  2. 休眠 150ms 模拟推理延迟
  3. 以机械臂当前位置为起点、全身关节零位为终点，生成 32 步线性插值 chunk
  4. 前 16 维为线性插值轨迹，后 112 维补 0

启动方式::

    pip install fastapi uvicorn opencv-python
    python scripts/pi05_mock_server.py --port 9090

协议: PI05_INFERENCE_DEPLOYMENT_PIPELINE_DESIGN.md §3
"""

from __future__ import annotations

import argparse
import base64
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 维度常量
# ---------------------------------------------------------------------------
STATE_DIM = 16
ACTION_DIM = 128
ROBOT_ACTION_DIM = 16
CHUNK_SIZE = 32      # 固定 32 步线性插值
MOCK_DELAY_MS = 150  # 固定 150ms

# ---------------------------------------------------------------------------
# 图片展示窗口
# ---------------------------------------------------------------------------
_show_images: bool = True
_display_lock = threading.Lock()
_display_buffers: Dict[str, Optional[np.ndarray]] = {
    "image_head": None,
    "image_wrist_left": None,
    "image_wrist_right": None,
}


def _display_thread() -> None:
    """独立线程：持续刷新三路图片窗口。"""
    window_names = {
        "image_head": "Head Camera (头部相机)",
        "image_wrist_left": "Left Wrist Camera (左腕相机)",
        "image_wrist_right": "Right Wrist Camera (右腕相机)",
    }
    cv2.namedWindow("Mock Server — Press Q to quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mock Server — Press Q to quit", 640, 60)

    while _show_images:
        # 将三路图片拼接成一行展示
        rows = []
        with _display_lock:
            for key in ("image_head", "image_wrist_left", "image_wrist_right"):
                img = _display_buffers.get(key)
                if img is not None:
                    # 加标签
                    labeled = img.copy()
                    label = window_names.get(key, key)
                    cv2.putText(labeled, label, (8, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    rows.append(labeled)
                else:
                    rows.append(np.zeros((480, 640, 3), dtype=np.uint8))

        if rows:
            canvas = np.hstack(rows)
            cv2.imshow("PI05 Mock Server — 三路相机回显 (Press Q to quit)",
                       cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        info_panel = np.zeros((60, 640 * 3, 3), dtype=np.uint8)
        cv2.putText(info_panel,
                    f"Mock Server running | Press Q to quit | delay={MOCK_DELAY_MS}ms chunk={CHUNK_SIZE}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("Mock Server — Press Q to quit", info_panel)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            _show_images = False
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="PI05 Mock Inference Server — Loopback Test", version="0.2.0")

_start_time: float = time.time()
_request_count: int = 0
_display_started: bool = False


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
    chunk_size: int = CHUNK_SIZE
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
    model_version: str = "pi05-v1.0-mock-loopback"
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
def _decode_image(b64_str: str) -> Optional[np.ndarray]:
    """Base64 JPEG → RGB ndarray (H, W, 3)。"""
    if not b64_str:
        return None
    try:
        jpeg = base64.b64decode(b64_str)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _generate_interpolation_chunk(
    state: np.ndarray,
) -> np.ndarray:
    """当前位置 → 全身关节零位的 32 步线性插值。

    Args:
        state: (16,) float32 当前关节状态

    Returns:
        (32, 128) float32 action chunk
          - 前 16 维: 32 步线性插值轨迹
          - 后 112 维: 全零
    """
    state = np.asarray(state, dtype=np.float32).flatten()
    if state.shape[0] < ROBOT_ACTION_DIM:
        padded = np.zeros(ROBOT_ACTION_DIM, dtype=np.float32)
        padded[: state.shape[0]] = state
        state = padded

    zero_pos = np.zeros(ROBOT_ACTION_DIM, dtype=np.float32)
    chunk = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)

    for i in range(CHUNK_SIZE):
        t = i / max(CHUNK_SIZE - 1, 1)  # 0.0 → 1.0
        # 线性插值: start + (end - start) * t  =  state + (0 - state) * t
        chunk[i, :ROBOT_ACTION_DIM] = state + (zero_pos - state) * t

    # 后 112 维保持为 0
    return chunk


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": "pi05-v1.0-mock-loopback",
        "gpu_available": False,
        "mock_mode": True,
        "test_mode": "loopback",
        "uptime_s": time.time() - _start_time,
        "requests_processed": _request_count,
    }


@app.post("/api/v1/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest) -> InferenceResponse:
    global _request_count, _display_started

    t0 = time.perf_counter()
    _request_count += 1

    # ---- 0. 显示图片 ----
    if not _display_started:
        _display_started = True
        threading.Thread(target=_display_thread, daemon=True).start()

    images_raw = req.observation.images
    for key in ("image_head", "image_wrist_left", "image_wrist_right"):
        b64_str = getattr(images_raw, key, "")
        img = _decode_image(b64_str)
        with _display_lock:
            _display_buffers[key] = img

    # ---- 1. 校验 ----
    state = np.array(req.observation.state, dtype=np.float32)
    if state.shape[0] < STATE_DIM:
        return InferenceResponse(
            request_id=req.request_id,
            status="error",
            error_code="INVALID_STATE_DIM",
            error_message=f"state dim={state.shape[0]} < required {STATE_DIM}",
            metadata=ResponseMetadata(timestamp_unix=time.time()),
        )

    # ---- 2. 模拟推理延迟 150ms ----
    time.sleep(MOCK_DELAY_MS / 1000.0)

    # ---- 3. 生成 32 步线性插值 chunk ----
    action_chunk = _generate_interpolation_chunk(state)

    dt_ms = (time.perf_counter() - t0) * 1000.0

    # ---- 4. 返回 ----
    print(
        f"[mock] #{_request_count} request_id={req.request_id} "
        f"prompt='{req.prompt[:40]}' "
        f"state[:3]={[round(v,4) for v in state[:3].tolist()]}... "
        f"chunk=({CHUNK_SIZE}, {ACTION_DIM}) "
        f"inference={dt_ms:.1f}ms"
    )

    return InferenceResponse(
        request_id=req.request_id,
        status="ok",
        action_chunk=action_chunk.tolist(),
        chunk_size=CHUNK_SIZE,
        action_dim=ACTION_DIM,
        metadata=ResponseMetadata(
            fps=req.metadata.fps or 30,
            inference_time_ms=round(dt_ms, 1),
            model_version="pi05-v1.0-mock-loopback",
            timestamp_unix=time.time(),
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PI05 Mock Inference Server — Loopback Test"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9090, help="Bind port (默认 9090)")
    args = parser.parse_args()

    import uvicorn

    print(f"[PI05 Mock Server — Loopback Test]")
    print(f"  endpoint:  http://{args.host}:{args.port}")
    print(f"  delay:     {MOCK_DELAY_MS}ms")
    print(f"  chunk:     {CHUNK_SIZE} steps (linear interpolation → zero)")
    print(f"  action:    {ACTION_DIM} dims (前16插值, 后112补零)")
    print(f"  display:   三路相机图片回显 (OpenCV, Press Q to quit)")
    print(f"  endpoints:")
    print(f"    GET  http://{args.host}:{args.port}/api/v1/health")
    print(f"    POST http://{args.host}:{args.port}/api/v1/infer")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
