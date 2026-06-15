# robodriver-robot-deepcybo-lite-aio-ros2

[![English](README-EN.md)](README-EN.md) | **简体中文**

DeepCybo Lite 双臂机器人的 **RoboDriver ROS2 接入包**（`aio`：单包同时采集 observation + action，落盘为 LeRobot / DoRobot 格式）。

完整话题约定、向量布局、调试故障树见：[Lite包使用说明.md](./Lite包使用说明.md)

---

## 概述

| 项目 | 说明 |
|------|------|
| pip 包名 | `robodriver_robot_deepcybo_lite_aio_ros2` |
| RoboDriver 类型 | `deepcybo-lite-aio-ros2` |
| 接入方式 | ROS2 **Jazzy**（团队调试版本；Humble 可参考同路径替换） |
| 臂 | 左/右各 **7 关节**（当前按 `bar_ws` arms 模式，14 维） |
| 状态/动作向量 | 各 **14 维**（`lite_hardware.yaml:joints.arm_joints` 顺序） |
| 相机 | 头部 + 左腕 + 右腕，**640×480**，30 Hz |
| 关节/采集限频 | **30 Hz**（与 Record 主循环一致） |

`bar_ws` 控制链路需发布 `/slave/lite/joint_states`（`sensor_msgs/JointState`）和 `/slave/remote_policy_controller/command`（`bar_msgs/MITCommand`）；相机话题在 `config.py` 的 `DeepcyboLiteRos2Topics` 中定义。

---

## 包结构

```text
robodriver-robot-deepcybo-lite-aio-ros2/
├── robodriver_robot_deepcybo_lite_aio_ros2/
│   ├── config.py      # 机器人类型注册、关节/相机 schema、ROS2 话题
│   ├── node.py        # 订阅 JointState/MITCommand、14 维重排、回放发布
│   ├── robot.py       # LeRobot Robot 接口
│   └── status.py      # 采集 HMI 设备状态
├── scripts/
│   ├── record_external_topics.py   # 录制外部真实 ROS2 话题
│   └── ros2_mock_lite_topics.sh   # 无真机时 mock 关节与桥接单相机
├── Lite包使用说明.md   # 详细文档 + 调试故障树（§10）
├── README.md / README-EN.md
└── pyproject.toml
```

---

## 环境要求

- Ubuntu + **ROS2 Jazzy**（已 `source /opt/ros/jazzy/setup.bash`）
- Python **≥ 3.10**
- 已安装 [RoboDriver](https://github.com/FlagOpen/RoboDriver) 主工程
- 采集 HMI（可选）：[RoboDriver-Server](https://github.com/FlagOpen/RoboDriver-Server) → `http://localhost:5805/hmi/`
- 各终端 **`ROS_DOMAIN_ID` 一致**

```bash
export ROS_DOMAIN_ID=0   # 按现场修改
```

---

## 安装

```bash
# 1. RoboDriver 主工程
cd /path/to/RoboDriver
pip install -e .    # 或 uv / conda，见主仓库 README

# 2. 本 Lite 包
cd robodriver/robots/robodriver-robot-deepcybo-lite-aio-ros2
pip install -e .
```

依赖说明：`rclpy`、`message_filters`、`bar_msgs` 由 **ROS2/bar_ws overlay** 提供，不写入 pip；`opencv-python`、`logging_mp` 由本包 `pyproject.toml` 安装。注意 Python 版本需与 ROS2 Jazzy 的 Python ABI 一致。

---

## ROS2 话题一览（默认）

### Observation（feedback，30 Hz）

| 话题 | 类型 | 维度 / 约束 |
|------|------|-------------|
| `/slave/lite/joint_states` | `sensor_msgs/JointState` | `name`/`position` 中必须包含 14 个 canonical arm joints |

### Action（command，30 Hz）

| 话题 | 类型 | 维度 / 约束 |
|------|------|-------------|
| `/slave/remote_policy_controller/command` | `bar_msgs/MITCommand` | `joint_names`/`position` 中必须包含同一 14 关节；回放也发布到此话题 |

### 相机（30 Hz，`CompressedImage`）

| 话题 | 落盘 key |
|------|----------|
| `/deepcybo/lite/camera/head/image_raw/compressed` | `image_head` |
| `/deepcybo/lite/camera/wrist_left/image_raw/compressed` | `image_wrist_left` |
| `/deepcybo/lite/camera/wrist_right/image_raw/compressed` | `image_wrist_right` |

修改话题：编辑 `config.py` 中 `DeepcyboLiteRos2Topics`，并同步 [Lite包使用说明.md](./Lite包使用说明.md)。

---

## 快速开始

### 1. 启动中台或 mock

**真机/控制链路：** 机械臂按上表发布 2 路核心话题；相机仍按下表占位话题发布。

**无真机（仅关节）：**

```bash
source /opt/ros/jazzy/setup.bash
bash scripts/ros2_mock_lite_topics.sh
```

> mock 脚本不含相机 jpeg；完整 `connect()` 需三路相机或见 [Lite包使用说明.md §8、§10.14](./Lite包使用说明.md)。

### 2. 启动 RoboDriver

```bash
source /opt/ros/jazzy/setup.bash
conda activate robodriver   # 或你的 venv
cd /path/to/RoboDriver

python -m robodriver.scripts.run --robot.type=deepcybo-lite-aio-ros2
```

连接成功日志示例：`[连接成功] 所有设备已就绪`（3 相机 + leader/follower 各 14 维）。

### 3. 采集（可选 Server + HMI）

```bash
sudo systemctl start nginx
# 按 RoboDriver-Server README 启动 8088 服务
```

- 任务平台 / HMI：`http://localhost:5805/hmi/`
- 数据目录：`$ROBODRIVER_HOME/dataset/`（默认 `~/DoRobot/dataset/`）

### 4. 回放

采集平台点击「回放」，或调用 `robot.send_action()`（内部 `node.ros_replay` 向 `/slave/remote_policy_controller/command` 发布 14 维 `MITCommand`，并补齐 velocity/effort/stiffness/damping）。

---

## 配置要点

| 文件 | 作用 |
|------|------|
| `config.py` | `leader_motors` / `follower_motors` 键序 = 14 维 `ARM_JOINT_NAMES`；`control_fps` / `camera_fps` = 30 |
| `node.py` | 缺失 canonical joint 或维度错误 → Warning + 清空缓存，恢复前不落盘臂数据 |
| `robot.py` | 臂无效时 `get_observation` / `get_action` 抛 `DeviceNotConnectedError` |
| `status.py` | HMI 显示；臂名为 `leader_arms` / `follower_arms` |

`use_videos` 在 `config.py` 中设置（默认 `False`：先存图后编码）。

---

## TODO / 路线

- 下一版接入双夹爪后，将状态 / 动作向量从当前 arms-only **14 维**扩展为 **16 维**（左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1），并同步更新 `ARM_JOINT_NAMES`、话题约定与数据 schema。

---

## 常见问题（简要）

| 现象 | 处理 |
|------|------|
| `connect` 超时 | 查 `/slave/lite/joint_states` 与 `/slave/remote_policy_controller/command` 是否有完整 14 关节；相机仍需满足当前占位配置 |
| Warning 缺少 canonical joints | JointState/MITCommand 未包含完整 `ARM_JOINT_NAMES`；恢复前不采臂数据 |
| `8088` 连接失败 | 启动 RoboDriver-Server |
| `5805` 打不开 | `sudo systemctl restart nginx` |
| `robot.type` 找不到 | `pip install -e .` 安装本包 |

**完整故障树：** [Lite包使用说明.md — 第 10 章](./Lite包使用说明.md#10-调试故障树debug-fault-tree)

---

## 相关链接

- [RoboDriver](https://github.com/FlagOpen/RoboDriver)
- [RoboDriver 文档](https://flagopen.github.io/RoboDriver-Doc)
- [RoboDriver-Server](https://github.com/FlagOpen/RoboDriver-Server)
- [LeRobot](https://github.com/huggingface/lerobot)

---

## 致谢

- [LeRobot](https://github.com/huggingface/lerobot) — 数据与 Robot 接口
- [RoboDriver](https://github.com/FlagOpen/RoboDriver) — 采集与平台链路

## 引用

若使用本包，请同时引用 RoboDriver 与 LeRobot 相关论文/仓库说明。
