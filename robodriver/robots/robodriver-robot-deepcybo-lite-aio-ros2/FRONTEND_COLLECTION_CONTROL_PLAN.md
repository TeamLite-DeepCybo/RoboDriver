# ROS2 FSM-Controlled RoboDriver Collection Plan

本文档基于 `ROBODRIVER_HANDOFF.md`、`bar_ws` 中的 Lite 双机主从遥操 FSM，以及当前 RoboDriver 数采实现，给出一份让 **ROS2 终端前端** 直接控制 RoboDriver 录制 / 结束 / 保存 / 丢弃的实现方案。

这里的“前端”特指：

```text
/home/stvli/Desktop/bar_ws/src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh
  -> bilateral_fsm_loop.py
```

它不是 RoboDriver-Server/HMI。短期为了紧急 due，DeepCybo Lite 现场数采先由这个 shell/FSM 入口统一控制；长期仍会回到 HMI 等成熟界面。RoboDriver 是优秀的开源项目，我们希望后续有 contribution，因此当前实现原则是 **新增可选能力、复用原有基建、尽量不改原有 Server/HMI 行为**。本轮先给 Lite 增加 ROS2 采集控制入口，并复用 `Coordinator`、`Record`、`DoRobotDataset` 的现有落盘能力；等 HMI 接管时，再考虑把 HMI 路径也渐进式收敛到同一个核心 handler。

## 1. 目标

目标是让 `bar_ws` 终端 FSM 通过 ROS2 Bool 话题直接驱动 RoboDriver 本地落盘：

```text
FSM START gate
  -> /to_robodriver/start_collect=true
  -> RoboDriver 创建 Record 并开始写帧

FSM FINISH gate
  -> /to_robodriver/finish_collect=true
  -> RoboDriver 停止写帧并 flush/save 成 pending episode

FSM Y
  -> /to_robodriver/affirm_to_collect=true
  -> RoboDriver 保留 pending episode

FSM N
  -> /to_robodriver/affirm_to_collect=false
  -> RoboDriver 删除 pending episode
```

短期职责边界：

```text
bar_ws ROS2 FSM：控制机器人状态机、操作员 gate、发布采集控制话题
RoboDriver Coordinator：新增 Lite ROS2 采集入口，复用原落盘流程
Record / DoRobotDataset：唯一落盘执行者
RoboDriver-Server/HMI：暂不作为紧急 due 阶段主控；Lite 可离线运行，但保留兼容入口
```

开源兼容原则：

```text
优先新增：新增 ROS2 bridge node 与 Lite 专用 Coordinator API
优先复用：仍由 Record/DoRobotDataset 执行采集、保存、删除
谨慎修改：不重写原 robot_command 大分支，不改变现有 HMI/Server 协议
局部修正：只让 Record.save() 返回已经生成的 save_data，避免重复造轮子
```

长期目标：

```text
HMI / 成熟界面：正式人机交互入口
bar_ws ROS2 FSM：机器人控制状态机与低层采集 gate
RoboDriver Coordinator：被两类入口复用的唯一采集状态机
Record / DoRobotDataset：唯一落盘执行者
```

## 2. bar_ws 侧现状

入口脚本：

```text
/home/stvli/Desktop/bar_ws/src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh
```

该脚本加载 ROS2 与 `bar_ws/install/setup.bash`，然后执行：

```text
/home/stvli/Desktop/bar_ws/src/bar_ros2/ops/lite/scripts/bilateral_fsm_loop.py
```

`bilateral_fsm_loop.py` 已经发布三类采集控制话题：

```text
/to_robodriver/start_collect
/to_robodriver/finish_collect
/to_robodriver/affirm_to_collect
```

消息类型均为：

```text
std_msgs/msg/Bool
```

QoS 为：

```text
RELIABLE + TRANSIENT_LOCAL
```

并且每个状态边沿重复发布三次。因此 RoboDriver 侧必须按状态变化去重，不能把重复消息当成多次命令。

FSM 时序：

```text
进入遥操：
  master -> zero_torque_controller
  slave  -> remote_policy_controller
  publish finish_collect=false
  publish start_collect=true

结束本段：
  publish start_collect=false
  publish finish_collect=true

人工确认：
  Y/y -> publish affirm_to_collect=true
  N/n -> publish affirm_to_collect=false
```

结论：`bar_ws` 侧基本准备好了。RoboDriver 侧缺的是订阅这三个话题并映射到 `Record` 生命周期的控制桥。

## 3. RoboDriver 当前数采现状

当前 RoboDriver 主流程：

```text
python -m robodriver.scripts.run --robot.type=deepcybo-lite-aio-ros2
  -> Daemon 持续读取 robot observation/action/status
  -> Coordinator 连接 RoboDriver-Server
  -> Coordinator 等待 HMI/Server 的 robot_command
  -> Coordinator 创建 Record
  -> Record 周期性从 Daemon 取 observation/action
  -> DoRobotDataset 写 parquet/images/meta
```

关键代码：

- `src/RoboDriver/robodriver/scripts/run.py`
  - 创建 `Daemon`
  - 创建 `Coordinator`
  - 启动 ROS2 node manager
  - 主循环持续 `daemon.update()`
- `src/RoboDriver/robodriver/core/coordinator.py`
  - 当前只处理 Server/HMI 的 `robot_command`
  - 已有 `start_collection`
  - 已有 `finish_collection`
  - 已有 `discard_collection`
  - 已有 `submit_collection`
- `src/RoboDriver/robodriver/core/recorder.py`
  - `Record.start()` 启动采集线程
  - `Record.stop()` 停止采集线程
  - `Record.save()` 调用 `DoRobotDataset.save_episode()`
  - `Record.discard()` 删除已保存 episode 或清空未保存 buffer

当前问题：

- RoboDriver 没有订阅 `/to_robodriver/*`。
- RoboDriver 主流程默认连接 RoboDriver-Server；紧急 due 阶段如果只用 ROS2 FSM，需要 Lite 在 Server 不在线时仍能启动。
- `Coordinator` 原 HMI 路径的数采状态是隐式的；Lite ROS2 入口需要显式 `IDLE/COLLECTING/WAITING_AFFIRM`，以处理 latch 和重复消息。
- `finish_collection` 当前已经执行本地 `save_episode()`，但没有把这一阶段命名为 pending。
- `submit_collection` 当前基本是占位，并不真正参与本地保存。

这些机制可以复用。短期实现不强行改造原 HMI 路径，而是在旁边新增 Lite ROS2 控制入口；长期再考虑把 HMI 路径迁移到同一套显式状态机。

## 4. 新状态机

在 `Coordinator` 内为 Lite ROS2 入口新增显式状态：

```python
class CollectionState(str, Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    WAITING_AFFIRM = "WAITING_AFFIRM"
    SAVING = "SAVING"
    DISCARDING = "DISCARDING"
    ERROR = "ERROR"
```

维护字段：

```python
self.collection_state = CollectionState.IDLE
self.record: Record | None = None
self.pending_episode_index: int | None = None
self.pending_save_data: dict | None = None
self.pending_record_cmd: dict | None = None
self.last_state_change_time = time.time()
```

状态迁移：

```text
IDLE
  start_collect=true:
    create Record
    Record.start()
    state = COLLECTING

COLLECTING
  finish_collect=true:
    Record.stop()
    Record.save()
    pending_episode_index = Record.last_record_episode_index
    pending_save_data = Record.save_data
    state = WAITING_AFFIRM

WAITING_AFFIRM
  affirm_to_collect=true:
    keep pending episode
    clear pending fields
    state = IDLE

WAITING_AFFIRM
  affirm_to_collect=false:
    Record.discard()
    clear pending fields
    state = IDLE
```

重复 / 陈旧消息处理：

- `IDLE` 收到 `finish_collect=true`：忽略，视为 stale latch。
- `IDLE` 收到 `affirm_to_collect=*`：忽略，视为没有 pending episode。
- `COLLECTING` 重复收到 `start_collect=true`：忽略。
- `WAITING_AFFIRM` 重复收到 `finish_collect=true`：忽略，不再次 `save()`。
- `WAITING_AFFIRM` 收到 `start_collect=true`：拒绝，要求先 affirm 当前 pending episode。

## 5. ROS2 Collection Bridge

新增一个 RoboDriver ROS2 node，例如：

```text
src/RoboDriver/robodriver/core/ros2_collection_bridge.py
```

职责：

- 订阅 `/to_robodriver/start_collect`
- 订阅 `/to_robodriver/finish_collect`
- 订阅 `/to_robodriver/affirm_to_collect`
- 使用 RELIABLE + TRANSIENT_LOCAL QoS
- 在 ROS2 callback 中只做轻量分发，不直接执行耗时落盘
- 将事件投递回 `Coordinator` 所在 asyncio loop

示意：

```python
class Ros2CollectionBridge(Node):
    def __init__(self, coordinator, loop):
        super().__init__("robodriver_collection_bridge")
        self.coordinator = coordinator
        self.loop = loop
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.create_subscription(
            Bool,
            "/to_robodriver/start_collect",
            self._on_start_collect,
            qos,
        )
```

callback 中使用：

```python
asyncio.run_coroutine_threadsafe(
    self.coordinator.handle_ros2_start_collect(msg.data),
    self.loop,
)
```

不要在 ROS2 callback 线程里直接 `Record.save()`，因为 `save_episode()`、图片 writer flush、metadata 更新和校验都可能耗时。

## 6. Coordinator 内部方法拆分

当前 `Coordinator.__on_robot_command_handle()` 中直接写了 HMI 命令逻辑。为了避免对开源主线造成过大的结构性差异，本轮不重写这个大分支，而是新增一组 Lite ROS2 专用方法。它们复用 `Record` 的生命周期，但不改变原 HMI/Server 的协议和返回行为。

本轮新增：

```python
async def handle_ros2_start_collect(value: bool) -> dict:
    ...

async def handle_ros2_finish_collect(value: bool) -> dict:
    ...

async def handle_ros2_affirm_to_collect(value: bool) -> dict:
    ...
```

内部再落到：

```text
start_collection_from_ros2()
finish_collection_from_ros2()
affirm_collection_from_ros2(keep=True/False)
```

ROS2 入口映射：

```text
/to_robodriver/start_collect=true
  -> start_collection_from_ros2()

/to_robodriver/finish_collect=true
  -> finish_collection_from_ros2()

/to_robodriver/affirm_to_collect=true
  -> affirm_collection_from_ros2(True)

/to_robodriver/affirm_to_collect=false
  -> affirm_collection_from_ros2(False)
```

`start_collect=false` 和 `finish_collect=false` 只用于清 latch，不触发业务动作。

后续 HMI 接管时，可以把原有 `start_collection/finish_collection/submit_collection/discard_collection` 逐步迁移到同一套内部 handler；但这不是紧急 due 阶段的必要改动。

## 7. ROS2 FSM 入口的任务 metadata

HMI 的 `start_collection` 会带：

```text
task_id
task_name
task_data_id
countdown_seconds
```

ROS2 Bool 话题没有 payload，所以 RoboDriver 侧需要自动生成 `record_cmd`。

建议规则：

```python
task_name = os.getenv("DEEPCYBO_LITE_TASK_NAME", "deepcybo_lite_bilateral")
task_id = os.getenv("DEEPCYBO_LITE_TASK_ID", datetime.now().strftime("%Y%m%d"))
task_data_prefix = os.getenv("DEEPCYBO_LITE_TASK_DATA_PREFIX", "ros2")
task_data_id = f"{task_data_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sequence:04d}"
machine_id = os.getenv("DEEPCYBO_LITE_MACHINE_ID", "deepcybo-lite-aio-ros2")
countdown_seconds = 0
source = "ros2_fsm"
```

`task_data_id` 不建议由环境变量固定为某个单值，否则连续采集容易覆盖同一段数据。ROS2 入口会自动生成唯一后缀；若目标目录已存在，则追加 `retryXX` 后缀换用新目录。

拼出的 `repo_id` 保持兼容：

```text
{task_name}_{task_id}_{task_data_id}
```

注意：ROS2 FSM 已经有自己的 START 前倒计时，所以 RoboDriver 从 ROS2 bridge 收到 `start_collect=true` 后不应再二次倒计时。

## 8. run.py 挂载 bridge

当前 `run.py` 已经在 ROS2 robot 类型下创建 `ROS2_NodeManager`，并添加 Lite robot node。

需要在 `Coordinator` 创建之后、`ros2_manager.start()` 之前加：

```python
if cfg.robot.type == "deepcybo-lite-aio-ros2":
    loop = asyncio.get_running_loop()
    collection_bridge = Ros2CollectionBridge(coordinator, loop)
    ros2_manager.add_node(collection_bridge)
```

后续 HMI 成熟后，ROS2 bridge 仍可保留为同一状态机的自动控制入口，或通过配置项启停。

同时，紧急 due 阶段允许 Lite 在 RoboDriver-Server 不在线时继续启动：

```text
cfg.robot.type == "deepcybo-lite-aio-ros2"
  Coordinator.start() 连接 Server 失败
  -> 记录 warning
  -> 继续启动 ROS2 bridge / Daemon / Record 流程
```

该离线容错只对 Lite 类型启用；其他机器人类型仍保持原先 Server 连接失败即失败的行为。

## 9. Record.save() 返回值

`Record.save()` 当前会设置：

```python
self.save_data = data
```

但没有显式 `return data`。

建议修改为：

```python
self.save_data = data
return data
```

这样 `finish_collect=true` 后，Coordinator 可以稳定拿到：

- `file_message.file_local_path`
- `file_message.file_size`
- `file_message.file_duration`
- `verification`

## 10. 数据路径

Lite ROS2 bridge 的产品环境建议：

```bash
export DEEPCYBO_LITE_DATA_ROOT=/media/stvli/0EE4-E658
```

Lite ROS2 bridge 数据路径：

```text
$DEEPCYBO_LITE_DATA_ROOT/YYYYMMDD/user/{task_name}_{task_id}/{repo_id}/
```

即：

```text
/media/stvli/0EE4-E658/YYYYMMDD/user/{task_name}_{task_id}/{repo_id}/
```

原 RoboDriver HMI/Server 路径仍由 `ROBODRIVER_HOME` 和 `DOROBOT_DATASET` 控制：

```text
$ROBODRIVER_HOME/dataset/YYYYMMDD/user/{task_name}_{task_id}/{repo_id}/
```

这样不会在核心 `constants.py` 中硬编码 Lite 现场磁盘路径，也避免影响其他机器人类型。

## 11. 七处修改清单

### 1. 新增 `CollectionState`

文件：

```text
src/RoboDriver/robodriver/core/coordinator.py
```

目标：

- 显式追踪 `IDLE/COLLECTING/WAITING_AFFIRM`
- 所有入口统一检查状态

### 2. 新增 Lite ROS2 collection handler

文件：

```text
src/RoboDriver/robodriver/core/coordinator.py
```

目标：

- `handle_ros2_start_collect(...)`
- `handle_ros2_finish_collect(...)`
- `handle_ros2_affirm_to_collect(...)`
- `start_collection_from_ros2(...)`
- `finish_collection_from_ros2(...)`
- `affirm_collection_from_ros2(...)`
- 原 Server/HMI 分支暂不重写

### 3. 新增 ROS2 collection bridge

文件：

```text
src/RoboDriver/robodriver/core/ros2_collection_bridge.py
```

目标：

- 订阅 `/to_robodriver/*`
- 同 handoff QoS
- 去重 / 过滤 false latch
- 将事件投递到 Coordinator asyncio loop

### 4. 在 `run.py` 挂载 bridge

文件：

```text
src/RoboDriver/robodriver/scripts/run.py
```

目标：

- DeepCybo Lite ROS2 启动时自动启用 bridge
- bridge 加入 `ROS2_NodeManager`
- Lite 可在 RoboDriver-Server 不在线时继续启动 ROS2 FSM 数采

### 5. 生成 ROS2 FSM record metadata

文件：

```text
src/RoboDriver/robodriver/core/coordinator.py
```

目标：

- 为 ROS2 Bool 入口自动生成 `task_name/task_id/task_data_id`
- `countdown_seconds=0`
- 路径优先走 `DEEPCYBO_LITE_DATA_ROOT`，不影响其他机器人类型

### 6. 保留 HMI 语义，新增 ROS2 affirm 语义

文件：

```text
src/RoboDriver/robodriver/core/coordinator.py
```

目标：

- 原 `submit_collection` / `discard_collection` 暂不改动，避免 HMI/Server 兼容性风险
- ROS2 `affirm_to_collect=true` 保留 pending episode
- ROS2 `affirm_to_collect=false` 删除 pending episode
- 后续 HMI 接管时再把 `submit_collection` / `discard_collection` 渐进式迁移为同一语义

### 7. `Record.save()` 返回数据

文件：

```text
src/RoboDriver/robodriver/core/recorder.py
```

目标：

- `return data`
- 让 Coordinator 不依赖隐式 `self.record.save_data`

## 12. 测试计划

### 静态检查

```bash
cd /home/stvli/Desktop/robodriver_ws/src/RoboDriver
python3 -m py_compile \
  robodriver/core/coordinator.py \
  robodriver/core/recorder.py \
  robodriver/core/ros2_collection_bridge.py \
  robodriver/scripts/run.py
```

### 话题观察

```bash
source /opt/ros/jazzy/setup.bash
source /home/stvli/Desktop/bar_ws/install/setup.bash

ros2 topic echo /to_robodriver/start_collect std_msgs/msg/Bool
ros2 topic echo /to_robodriver/finish_collect std_msgs/msg/Bool
ros2 topic echo /to_robodriver/affirm_to_collect std_msgs/msg/Bool
```

### 端到端流程

终端 A：启动 RoboDriver Lite。

```bash
cd /home/stvli/Desktop/robodriver_ws/src/RoboDriver
source /opt/ros/jazzy/setup.bash
source /home/stvli/Desktop/bar_ws/install/setup.bash
export DEEPCYBO_LITE_DATA_ROOT=/media/stvli/0EE4-E658
python -m robodriver.scripts.run --robot.type=deepcybo-lite-aio-ros2
```

终端 B：启动 FSM。

```bash
cd /home/stvli/Desktop/bar_ws
src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh --max-cycles 1 --start-delay 0.5
```

预期：

```text
Enter 开始：
  RoboDriver state -> COLLECTING
  数据目录开始写入 image/parquet buffer

Enter 结束：
  RoboDriver state -> WAITING_AFFIRM
  本地 episode 已 flush/save

Y：
  RoboDriver state -> IDLE
  数据目录保留

N：
  RoboDriver state -> IDLE
  pending episode 被删除
```

## 13. 风险与处理

| 风险 | 处理 |
|---|---|
| FSM 重复发布三次导致重复保存 | Coordinator 状态机去重 |
| RoboDriver 晚启动收到旧 latch | `finish=true` / `affirm=*` 在 IDLE 下忽略 |
| `start_collect=true` 时 observation/action 尚未 ready | start 前检查 Daemon 最新 observation/action，必要时短等待 |
| `finish_collect=true` 后进程崩溃 | pending episode 已在本地保存，后续人工扫描处理 |
| ROS2 callback 中执行 save 阻塞 executor | callback 只投递 asyncio 任务 |
| HMI 和 ROS2 FSM 同时控制 | 紧急 due 阶段默认 Lite 以 ROS2 FSM 为主控；HMI 入口保留原行为，ROS2 入口检测已有录制并拒绝冲突 start |

## 14. 推荐结论

RoboDriver 侧不需要推翻原有数采。我们要做的是：

```text
新增 ROS2 Collection Bridge
  + Lite ROS2 专用 Coordinator 状态机
  + finish 后 pending
  + affirm true/false 决定保留/删除
  + 原 Server/HMI 路径保持兼容
```

这样可以让 `bar_ws` 的 `bilateral_fsm_ready.sh` 在紧急 due 阶段成为 DeepCybo Lite 现场数采的统一控制入口，同时保留 RoboDriver 原本的 `Record` / `DoRobotDataset` 落盘能力，并为后续 HMI 接管准备同一套底层控制语义。
