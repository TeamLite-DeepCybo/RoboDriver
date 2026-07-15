# DeepCybo Lite — PI05 推理部署下行管线设计方案

> 设计人：李佩泽（DeepCybo 架构师）  
> 分支：`feat/deployment`  
> 最后更新：2026-07-14  
> 状态：设计+实现阶段

---

## 0. 设计目标与原则

### 0.1 核心目标

将 PI05 推理部署到 DeepCybo Lite 双臂机器人，实现"视觉-语言-动作"的端到端闭环：

```
相机 + 关节状态 ──→ PI05 推理(远端 GPU) ──→ action chunk ──→ 机器人执行
```

### 0.2 架构原则

1. **算力与控制分离**：推理服务器（A6000/云端 GPU）与机器人工控机不在同一台机器上
2. **请求-响应通信**：客户端（机器人端）主动发起推理请求，服务端（GPU 端）返回 action chunk
3. **同步主循环**：客户端遵循"采集 obs → 请求推理 → 等待响应 → 回放 chunk → 重新采集"的同步循环。异步流水线（RTC 场景）留作后续 TODO
4. **128 维 action 向量**：协议层 action 为 128 维。前 16 维对齐 `LITE_JOINT_NAMES`（双臂 14 关节 + 2 夹爪），后 112 维预留（灵巧手等），当前阶段客户端侧补 0 或截断
5. **复用现有基建**：最大化复用 `robot.send_action()`、`Daemon`、`busy_wait` 等已有链路

### 0.3 与现有模式的对比

| 维度 | 录制模式 (Record) | 回放模式 (Replay) | 推理部署模式 (Deploy) |
|------|-------------------|-------------------|------------------------|
| action 来源 | 主手遥操 (leader) | 本地 parquet | 远端 PI05 推理 |
| action 维度 | 16 | 16（parquet 中） | **128**（协议层） |
| 数据方向 | robot → 磁盘 | 磁盘 → robot | 网络双向：robot ↔ server |
| 控制模式 | 同步落盘 | 同步回放 | **同步请求→回放** |
| 控制界面 | ROS2 FSM / HMI | HMI / Server | 新增 CLI / HMI |

---

## 1. 系统架构全景

```
┌─────────────────────────────────┐      ┌──────────────────────────────────┐
│  客户端：机器人工控机            │      │  服务端：GPU 推理服务器 (A6000)    │
│  (RoboDriver)                   │      │  (PI05 Inference Server)          │
│                                 │      │                                   │
│  ┌─────────────────────────┐    │ HTTP │  ┌────────────────────────────┐   │
│  │  InferenceDeployment    │    │ POST │  │  POST /api/v1/infer        │   │
│  │  (同步主循环)           │    │─────→│  │                            │   │
│  │                         │    │      │  │  1. 解码图片                │   │
│  │  1. robot.get_obs()    │    │      │  │  2. 组装 PI05 输入         │   │
│  │  2. build & POST req   │    │      │  │  3. 推理 → action chunk    │   │
│  │  3. 阻塞等待响应        │    │      │  │     shape: (N, 128)        │   │
│  │  4. ActionChunkReplayer │    │←─────│  │  4. 返回 InferenceResp     │   │
│  │     逐帧 send_action() │    │ JSON │  └────────────────────────────┘   │
│  │  5. 重新采集 obs，循环  │    │      │                                   │
│  └─────────────────────────┘    │      └──────────────────────────────────┘
│                                 │
│  ┌─────────────────────────┐    │
│  │  DeepcyboLiteAioRos2Robot│   │
│  │  ├─ get_observation()   │    │
│  │  └─ send_action()       │    │
│  │      (16维 → MITCommand) │   │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
```

**主循环时序**（同步模式，无流水线）：

```
  while True:
    ① 采集 observation（state 16维 + 3路 image）
    ② POST /infer → 阻塞等待响应
    ③ 收到 action_chunk (N, 128)
    ④ ActionChunkReplayer 逐帧下发（前16维 → send_action）
    ⑤ chunk 耗尽 → 回到 ①
```

---

## 2. Action 向量的 128 维设计

### 2.1 维度分配

```
索引范围    长度    含义                    当前状态
─────────────────────────────────────────────────────
[0:14]      14      双臂 14 关节            已使用
[14:16]      2      左右夹爪                已使用
[16:128]   112      预留（灵巧手等）        补 0
```

### 2.2 客户端侧处理逻辑

```python
# 服务端返回 128 维 action → 客户端下发到机器人时，仅取前 16 维
ACTION_DIM = 128                           # 协议层维度
ROBOT_ACTION_DIM = 16                      # 机器人实际控制维度

def server_action_to_robot(server_vec: np.ndarray) -> np.ndarray:
    """128 维 → 16 维：取前 16 维，后 112 维忽略（当前阶段）。"""
    vec = np.asarray(server_vec, dtype=np.float32).flatten()
    if vec.shape[0] < ROBOT_ACTION_DIM:
        # 服务端返回维度不足 → 后补 0
        padded = np.zeros(ROBOT_ACTION_DIM, dtype=np.float32)
        padded[:vec.shape[0]] = vec
        return padded
    return vec[:ROBOT_ACTION_DIM].copy()
```

### 2.3 协议校验

- 客户端请求中 `state` 保持 16 维（机器人实际输出）
- 服务端响应中 `action_chunk` 每帧为 128 维
- 若服务端返回维度 ≠ 128：客户端按实际维度截断/补零，记录 warning
- 若服务端返回维度 < 16：视为错误，丢弃该 chunk

---

## 3. 通信协议

### 3.1 传输层

- 协议：HTTP/1.1 + JSON
- 编码：图片使用 Base64 编码嵌入 JSON
- 超时：请求超时 10s，连接超时 3s

### 3.2 请求格式：`POST /api/v1/infer`

```json
{
  "request_id": "deepcybo-lite-20260714-001",
  "robot_type": "deepcybo-lite-aio-ros2",
  "prompt": "将红色的方块捡起来，放进蓝色的盒子里",
  "observation": {
    "state": [0.12, -0.34, 0.05, 1.23, 0.00, 0.45, -0.67,
               0.11, -0.33, 0.06, 1.22, 0.01, 0.44, -0.66,
               0.85, 0.15],
    "images": {
      "image_head": "<base64_encoded_jpeg>",
      "image_wrist_left": "<base64_encoded_jpeg>",
      "image_wrist_right": "<base64_encoded_jpeg>"
    }
  },
  "metadata": {
    "fps": 30,
    "state_dim": 16,
    "action_dim": 128,
    "chunk_size": 50,
    "image_width": 640,
    "image_height": 480,
    "timestamp_unix": 1720940000.123
  }
}
```

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `request_id` | string | 是 | 请求唯一标识 |
| `robot_type` | string | 是 | 机器人类型，服务端据此选择推理配置 |
| `prompt` | string | 是 | 任务自然语言指令 |
| `observation.state` | float[16] | 是 | 从臂 16 维关节状态 |
| `observation.images.*` | string(base64) | 是 | 3 路 JPEG Base64 图片 |
| `metadata.action_dim` | int | 是 | **期望的 action 维度 = 128** |
| `metadata.chunk_size` | int | 否 | 期望的 action chunk 长度，默认 50 |

### 3.3 响应格式

```json
{
  "request_id": "deepcybo-lite-20260714-001",
  "status": "ok",
  "action_chunk": [
    [0.13, -0.33, ..., 0.0, 0.0],   // 128 维
    [0.14, -0.32, ..., 0.0, 0.0]    // 128 维
  ],
  "chunk_size": 50,
  "action_dim": 128,
  "metadata": {
    "fps": 30,
    "inference_time_ms": 45.2,
    "model_version": "pi05-v1.0-mock",
    "timestamp_unix": 1720940000.623
  }
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `request_id` | string | 回显 |
| `status` | "ok"\|"error" | 推理状态 |
| `action_chunk` | float[N][128] | N 帧 action，每帧 **128 维** |
| `action_dim` | int | 实际 action 维度（应为 128） |
| `chunk_size` | int | 实际 chunk 长度 |
| `metadata.inference_time_ms` | float | 推理耗时 |
| `metadata.model_version` | string | 模型版本标识 |

**错误响应**：

```json
{
  "request_id": "...",
  "status": "error",
  "error_code": "INFERENCE_FAILED",
  "error_message": "PI05 model inference failed: CUDA OOM",
  "action_chunk": [],
  "chunk_size": 0,
  "action_dim": 128,
  "metadata": {}
}
```

### 3.4 Health Check

```
GET /api/v1/health
```

响应：

```json
{"status": "ok", "model_loaded": "pi05-v1.0-mock", "gpu_available": false, "mock_mode": true}
```

---

## 4. 客户端模块实现

> 文件位于 `robodriver/core/`，全部为新增文件，零侵入现有代码。

### 4.1 文件布局

```
robodriver/core/
├── inference_client.py        ← HTTP 通信客户端
├── action_chunk_replayer.py   ← 128 维 chunk 逐帧回放器
└── inference_deployment.py    ← 推理部署同步主循环
```

### 4.2 `inference_client.py` — HTTP 通信客户端

职责：
1. 管理 `aiohttp` 会话
2. 将 observation 序列化为推理请求 JSON（图片 Base64 编码）
3. 发送 POST，解析响应
4. 校验 128 维 action_chunk
5. Health check

### 4.3 `action_chunk_replayer.py` — Chunk 逐帧回放器

职责：
1. 接收 `(N, 128)` 的 action chunk
2. 独立线程中按 FPS 节拍逐帧下发
3. 每帧取前 16 维（或按实际维度补 0/截断），映射为 `leader_*.pos` 命名 dict
4. 调用 `robot.send_action()` 发布 MITCommand

与 `replayer.py` 的对比：

| | `replayer.py` | `action_chunk_replayer.py` |
|---|---|---|
| 数据来源 | 本地 parquet | 内存 `np.ndarray` |
| action 维度 | 16（parquet schema） | **128**（协议层） |
| 下发维度 | 16 → send_action | 128 → 取前 16 → send_action |
| 控制方式 | 函数调用 | 独立线程 + start/stop |

### 4.4 `inference_deployment.py` — 推理部署主循环

职责：
1. 初始化 `InferenceClient` + `ActionChunkReplayer`
2. 同步主循环：采集 obs → 请求 → 等待 → 加载 chunk → 等待回放完成 → 再采集
3. 异常处理与降级

---

## 5. 服务端（Mock PI05 推理服务器）

> 由于开发机上无 PI05 模型参数和 GPU，先提供 mock 服务端。真实推理交由下一代 agent 实现。

### 5.1 Mock 行为

Mock 服务端接收请求后：
1. 校验 `state_dim=16`、`action_dim=128`
2. 等待 `mock_inference_delay_ms` 毫秒（模拟推理延迟，默认 50ms）
3. 以当前 state 为基准，叠加小幅度正弦扰动，生成 `(chunk_size, 128)` 的 action chunk
4. 前 16 维为有意义的运动轨迹，后 112 维填 0

### 5.2 文件位置

```
scripts/pi05_mock_server.py        ← Mock 推理服务端
```

### 5.3 启动方式

```bash
pip install fastapi uvicorn
python scripts/pi05_mock_server.py --port 9090 --delay-ms 50
```

---

## 6. Handover 文档

### 6.1 文件位置

```
PI05_SERVER_HANDOVER.md           ← 下一代 Agent 交接文档
```

### 6.2 交接内容

1. 请求/响应协议完整规范
2. 128 维 action 向量的维度分配表
3. 图片编码格式与约定
4. Mock 服务端代码说明
5. 真实 PI05 推理集成的 TODO 清单
6. 验收测试方案

---

## 7. 异常处理策略

| 场景 | 客户端行为 |
|------|-----------|
| 推理请求超时 | 记录 error，sleep 1s 后重试，最多 3 次 |
| 推理返回 error | 记录 error + error_code，sleep 1s 后重新采集 obs 重试 |
| action_chunk 维度 ≠ 128 | Warning 日志，按实际维度截断/补零，继续回放 |
| action_chunk 维度 < 16 | 丢弃 chunk，记录 error |
| 网络断开 | 等待重连，回放完当前 chunk 停止 |
| robot.send_action() 失败 | 记录 error，跳过该帧继续 |

---

## 8. TODO（后续迭代）

| 优先级 | 内容 | 说明 |
|--------|------|------|
| P0 | 真实 PI05 推理集成 | 替换 mock 模型为真实 PI05 权重 |
| P1 | 异步流水线（预取） | 回放当前 chunk 时异步请求下一段，避免机器人停顿 |
| P1 | gRPC + Protobuf | 替换 JSON+Base64，减少图片编码开销 |
| P2 | RTC（Real-Time Control） | 推理延迟 < 帧间隔的实时控制模式 |
| P2 | 灵巧手接入 | 启用 action 向量 16-127 维的实际控制 |
| P3 | HMI / ROS2 FSM 集成 | 通过 Coordinator 统一控制部署模式 |

---

## 9. 附录：关键常量

### 9.1 维度常量

```python
ACTION_DIM = 128         # 协议层 action 维度
STATE_DIM = 16           # 机器人状态/observation 维度
ROBOT_ACTION_DIM = 16    # 机器人实际控制维度（LITE_JOINT_NAMES 长度）
```

### 9.2 关节顺序 (`LITE_JOINT_NAMES`)

```
0   left_shoulder_pitch       7   right_shoulder_pitch
1   left_shoulder_roll        8   right_shoulder_roll
2   left_shoulder_yaw         9   right_shoulder_yaw
3   left_elbow_pitch         10   right_elbow_pitch
4   left_wrist_yaw           11   right_wrist_yaw
5   left_wrist_roll          12   right_wrist_roll
6   left_wrist_pitch         13   right_wrist_pitch
                             14   left_gripper
                             15   right_gripper
                             16-127  预留（零填充）
```

### 9.3 相关文件索引

| 文件 | 作用 |
|------|------|
| `robodriver/core/inference_client.py` | HTTP 通信 |
| `robodriver/core/action_chunk_replayer.py` | 128 维 chunk 逐帧下发 |
| `robodriver/core/inference_deployment.py` | 部署主循环 |
| `scripts/pi05_mock_server.py` | Mock 推理服务端 |
| `PI05_SERVER_HANDOVER.md` | 下一代 Agent 交接文档 |
| `robodriver/core/replayer.py` | 现有 replay 实现（节拍控制参考） |
| `robodriver/robots/.../robot.py` | `send_action()` 接口（16 维 MITCommand） |
| `robodriver/robots/.../node.py` | `ros_replay()` ROS2 发布 |
| `robodriver/robots/utils.py` | `busy_wait()` |

---

## 10. 与 bar_ws / cam_ros2_ws 的集成拓扑

### 10.1 现有主从遥操拓扑（采集模式）

```
cam_ros2_ws (camera_selection):
  /deepcybo/lite/camera/head/image_raw/compressed        ← video*
  /deepcybo/lite/camera/wrist_left/image_raw/compressed  ← video*
  /deepcybo/lite/camera/wrist_right/image_raw/compressed ← video*

bar_ws (dual_bilateral.launch.py):
  Master stack:  /master/lite/joint_states
  Slave stack:   /slave/lite/joint_states
                 /slave/remote_policy_controller/command (MITCommand subscriber)
  Bridge:        /master/lite/joint_states → /slave/remote_policy_controller/command
```

### 10.2 推理部署模式拓扑（只保留 slave）

```
cam_ros2_ws (camera_selection):  ← 不变
  /deepcybo/lite/camera/{head,wrist_left,wrist_right}/image_raw/compressed

bar_ws (只启动 slave stack，不启动 master/bridge):
  Slave stack:   /slave/lite/joint_states  ──→ robot.get_observation()
                 /slave/remote_policy_controller/command ←── ros_replay()

RoboDriver (推理客户端):
  DeepcyboLiteAioRos2RobotNode  订阅: 3路相机 + /slave/lite/joint_states
                                 发布: /slave/remote_policy_controller/command
  InferenceDeploymentLoop       同步主循环: obs → HTTP推理 → chunk回放
  InferenceClient               HTTP POST → GPU推理服务器
  ActionChunkReplayer           逐帧 send_action()
```

### 10.3 启动命令

```bash
# 1. 相机（现有，不变）
ros2 launch usb_cam camera_selection.launch.py

# 2. 从臂（仅 slave stack，不启动 master/bridge）
ros2 launch bar_bringup_lite real.launch.py \
  namespace:=slave \
  hardware_config:=.../lite_hardware_slave.yaml \
  calibration_file:=.../calibration_slave.yaml

# 3. 推理服务端（GPU 机器上）
python scripts/pi05_mock_server.py --port 9090

# 4. 推理客户端（本机）
python -m robodriver.scripts.deploy --server-url http://GPU_IP:9090 --prompt "任务指令"
```

---

## 11. ROS2 Action 决策记录

**决策：不引入 ROS2 Action 机制用于推理步骤。**

理由：
1. 架构原则要求算力与控制分离在不同机器，HTTP 无需 GPU 端安装 ROS2
2. PI05 推理延迟（50-200ms）是短请求-响应模式，Action 的长时间任务语义不匹配
3. Action 引入的反馈/抢占通道在当前单步推理场景中无实际收益
4. 若未来做 RTC（逐帧实时控制），可重新评估 Action 的必要性

---

## 12. 相机话题说明

`cam_ros2_ws` 通过 `camera_selection` 三摄像头选择器拉起，话题命名空间为：

```
/deepcybo/lite/camera/head/...
/deepcybo/lite/camera/wrist_left/...
/deepcybo/lite/camera/wrist_right/...
```

与 `DeepcyboLiteRos2Topics` 默认配置完全一致，无需额外 remap。

---

## 13. 从臂控制器状态机

部署模式下的从臂控制器状态由 `SlaveControllerFsm`（`slave_controller_fsm.py`）统一管理，
与数据采集 FSM（`bilateral_fsm_loop.py`）共享 `SwitchController` 服务调用模式。

### 13.1 状态图

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐  ┌────────────────┐
  │ZERO_TORQUE│→│ DAMPING │→│ STANDBY │→│ READY │→│ REMOTE_POLICY   │
  └──────────┘  │ (10s)   │  │(ramp)   │  └───┬───┘  │ (持续多chunk)  │
       ▲        └─────────┘  └─────────┘      │      └───────┬────────┘
       │                                       │              │
       │          ┌──────────────┐             │    Ctrl+C / 任务完成
       └──────────│DAMPING_SETTLE│◄────────────┘
                  │   (10s)      │
                  └──────┬───────┘
                         │ 优雅退出
                         ▼
                  ┌──────────┐
                  │ EXITING  │
                  └──────────┘
```

### 13.2 各状态行为

| 状态 | 控制器 | 持续条件 | 触发下一状态 |
|------|--------|----------|-------------|
| ZERO_TORQUE | zero_torque_controller | 瞬时 | 自动 → DAMPING |
| DAMPING | damping_controller | 10s 预置运动 | 自动 → STANDBY |
| STANDBY | standby_controller | 等待 ramp 完成 | 自动 → READY |
| READY | standby_controller | 等待推理服务端就绪 + Enter | Enter → REMOTE_POLICY |
| REMOTE_POLICY | remote_policy_controller | 持续请求 chunk 并回放 | Ctrl+C → DAMPING_SETTLE |
| DAMPING_SETTLE | damping_controller | 10s 阻尼缓冲 | 自动 → ZERO_TORQUE (下一轮) 或 EXITING |
| EXITING | zero_torque_controller | 终态 | — |

### 13.3 与数据采集 FSM 的差异

| 对比点 | 采集 FSM | 部署 FSM |
|--------|----------|----------|
| 管理范围 | 主臂 + 从臂 | 仅从臂 |
| 任务间释放 | 不经过 zero_torque | 经过 damping(10s) → zero_torque |
| 远程策略前检查 | 等待操作员 | 等待推理服务端 health check |
| 远程策略内行为 | 单段遥操 | 持续多段 action chunk |
| 退出 | damping(5s) → zero_torque | damping(10s) → zero_torque |
