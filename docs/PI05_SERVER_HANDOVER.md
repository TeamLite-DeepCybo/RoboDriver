# PI05 推理服务端 — 下一代 Agent 交接文档

> 交接着：李佩泽（DeepCybo 架构师）  
> 交付分支：`feat/deployment`  
> 交接日期：2026-07-14  
> 目标：为后继 agent 提供足够的上下文，在 Mock 基础上实现真实 PI05 推理集成

---

## 1. 当前交付物总览

```
robodriver_ws/
├── PI05_INFERENCE_DEPLOYMENT_PIPELINE_DESIGN.md   ← 完整设计文档
├── PI05_SERVER_HANDOVER.md                        ← 本文档
│
├── src/RoboDriver/robodriver/core/
│   ├── inference_client.py                        ← HTTP 通信客户端（128维）
│   ├── action_chunk_replayer.py                   ← chunk 逐帧回放器（128→16）
│   └── inference_deployment.py                    ← 同步主循环
│
└── src/RoboDriver/scripts/
    └── pi05_mock_server.py                        ← Mock 推理服务端
```

---

## 2. 协议摘要

### 2.1 请求 → `POST /api/v1/infer`

```json
{
  "request_id": "deepcybo-lite-{timestamp}-{seq}",
  "robot_type": "deepcybo-lite-aio-ros2",
  "prompt": "任务自然语言指令",
  "observation": {
    "state": [16个 float32],            // LITE_JOINT_NAMES 顺序
    "images": {
      "image_head": "<base64_jpeg>",
      "image_wrist_left": "<base64_jpeg>",
      "image_wrist_right": "<base64_jpeg>"
    }
  },
  "metadata": {
    "fps": 30, "state_dim": 16, "action_dim": 128,
    "chunk_size": 50, "image_width": 640, "image_height": 480
  }
}
```

### 2.2 响应 ←

```json
{
  "request_id": "...",          // 回显
  "status": "ok",               // "ok" | "error"
  "action_chunk": [[128个f32], ...],  // (N, 128)
  "chunk_size": 50,
  "action_dim": 128,
  "metadata": {
    "fps": 30,
    "inference_time_ms": 45.2,
    "model_version": "pi05-v1.0",
    "timestamp_unix": 1720940000.623
  }
}
```

### 2.3 128 维 Action 向量分配

```
[0:14]   双臂 14 关节         ← 已使用
[14:16]  左右夹爪             ← 已使用
[16:128] 预留（灵巧手等）     ← 当前填 0
```

### 2.4 前 16 维关节顺序

```
0 left_shoulder_pitch      7 right_shoulder_pitch
1 left_shoulder_roll       8 right_shoulder_roll
2 left_shoulder_yaw        9 right_shoulder_yaw
3 left_elbow_pitch        10 right_elbow_pitch
4 left_wrist_yaw          11 right_wrist_yaw
5 left_wrist_roll         12 right_wrist_roll
6 left_wrist_pitch        13 right_wrist_pitch
                          14 left_gripper
                          15 right_gripper
```

---

## 3. 客户端行为约定（服务端需要知道的）

1. **同步模式**：客户端发送请求后**阻塞等待**响应，收到 chunk 后逐帧回放，回放完毕后立即发送**下一轮请求**（带新的 observation）
2. **超时重试**：请求超时后最多重试 3 次，间隔 1s
3. **连续失败**：连续 5 次推理失败后客户端退出主循环
4. **维度容错**：若服务端返回维度 ≠ 128，客户端会补零/截断并记录 warning；若 < 16 则丢弃
5. **无流水线**：当前不预取下一段 chunk（TODO 中有异步流水线规划）

---

## 4. Mock 服务端说明

`scripts/pi05_mock_server.py` 模拟 PI05 推理行为：

| 特性 | 当前实现 |
|------|----------|
| 框架 | FastAPI + uvicorn |
| action chunk 生成 | state + sinusoidal perturbation |
| 前 16 维 | 有意义的平滑运动轨迹 |
| 后 112 维 | 零填充 |
| 延迟模拟 | `--delay-ms` 参数（默认 50ms） |
| 图片处理 | 不解析 Base64（mock 模式忽略图片） |

启动：

```bash
pip install fastapi uvicorn
python scripts/pi05_mock_server.py --port 9090 --delay-ms 50
```

---

## 5. 真实 PI05 推理集成 TODO

### 5.1 必须实现

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 加载 PI05 模型权重 | 启动时加载到 GPU 显存 | P0 |
| 图片解码 | 将 Base64 JPEG → RGB ndarray (3, H, W)，按 PI05 规范 resize/normalize | P0 |
| 推理管线 | state(16) + images(3) + prompt(str) → action_chunk(N, 128) | P0 |
| action 维度补齐 | PI05 输出可能 < 128 维，需补齐到 128（后补 0）| P0 |
| 错误处理 | CUDA OOM、模型推理异常 → 返回 error 响应 | P0 |
| API 兼容 | 保持 POST /api/v1/infer 请求/响应 JSON schema 不变 | P0 |

### 5.2 建议实现

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 热加载模型 | 无需重启即可切换模型权重 | P1 |
| 推理耗时监控 | per-request 计时，记录到日志/metrics | P1 |
| 请求队列 | GPU 串行推理时排队，避免并发请求竞争 | P1 |
| 图片压缩质量协商 | 客户端可指定 JPEG quality（当前固定 85） | P2 |

### 5.3 参考资源

- PI05 模型仓库：（待补充）
- OpenPI 推理代码：（待补充）
- 字段映射见 `lerobot_pic_debug/DOWNSTREAM_IMAGE_PIPELINE_HANDOFF.md` §6

---

## 6. 验收测试方案

### 6.1 Mock 服务端 + 客户端联调

```bash
# 终端 1: 启动 mock 服务端
python src/RoboDriver/scripts/pi05_mock_server.py --port 9090 --delay-ms 50

# 终端 2: 用 curl 手动测试
curl -s http://127.0.0.1:9090/api/v1/health | python3 -m json.tool

curl -s -X POST http://127.0.0.1:9090/api/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "robot_type": "deepcybo-lite-aio-ros2",
    "prompt": "测试任务",
    "observation": {
      "state": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
      "images": {"image_head":"","image_wrist_left":"","image_wrist_right":""}
    },
    "metadata": {"fps":30,"state_dim":16,"action_dim":128,"chunk_size":5}
  }' | python3 -m json.tool | head -30
```

验收标准：
- health check 返回 `{"status": "ok", "mock_mode": true}`
- infer 返回 `action_chunk` 为 `(chunk_size, 128)` 的二维数组
- 前 16 维有非零值，后 112 维全为 0

### 6.2 Python 模块导入测试

```bash
cd /home/stvli/Desktop/robodriver_ws/src/RoboDriver
python3 -c "
from robodriver.core.inference_client import InferenceClient, ACTION_DIM, STATE_DIM
from robodriver.core.action_chunk_replayer import ActionChunkReplayer
print(f'ACTION_DIM={ACTION_DIM}, STATE_DIM={STATE_DIM}')
print('All imports OK')
"
```

### 6.3 维度转换单元测试

```python
import numpy as np
from robodriver.core.inference_client import server_action_to_robot
from robodriver.core.action_chunk_replayer import ActionChunkReplayer

# 128 → 16
vec_128 = np.random.randn(128).astype(np.float32)
vec_16 = server_action_to_robot(vec_128)
assert vec_16.shape == (16,), f"Expected (16,), got {vec_16.shape}"
assert np.allclose(vec_16, vec_128[:16])

# 128 → dict
action_dict = ActionChunkReplayer._action_vector_to_dict(vec_128)
assert len(action_dict) == 16
assert "leader_left_shoulder_pitch.pos" in action_dict
print("All dimension tests passed")
```

---

## 7. 已知限制与风险

| 限制 | 影响 | 缓解 |
|------|------|------|
| JSON + Base64 图片开销大 | 每帧 3 路 JPEG ~200KB Base64 → ~270KB JSON，30Hz 下 ~8MB/s | 后续升级 gRPC+Protobuf 直接传 bytes |
| 同步请求模式延迟敏感 | 推理延迟 > chunk 回放时长时机器人停顿 | TODO 中规划异步流水线 |
| 无动作平滑/插值 | chunk 间可能有跳变 | 可在 ActionChunkReplayer 中加入帧间线性插值 |
| Mock 服务端不解析图片 | 联调时无法验证图片编码正确性 | 真实 PI05 集成时需要端到端图片通路测试 |

---

## 8. 联系人

- 架构/管线设计：李佩泽（DeepCybo）
- 客户端模块代码：当前 agent 交付
- PI05 模型/推理：后继 agent 负责
- DeepCybo PR checklist：参考 `deepcybo-pr-checklist` skill
