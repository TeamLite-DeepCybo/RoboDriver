# DeepCybo Lite ROS2 录制操作手册

本文档用于现场拉起 DeepCybo Lite 的 RoboDriver 录制端，并通过 ROS2 指令控制开始、结束、保存或丢弃。

## 0. 分支说明

在 PR 合并前，RoboDriver 必须运行功能分支：

```bash
cd /home/stvli/Desktop/robodriver_ws/src/RoboDriver
git switch feat-deepcybo_lite_unified_cmd
```

PR 合并后，改为运行受保护主分支：

```bash
cd /home/stvli/Desktop/robodriver_ws/src/RoboDriver
git switch main-deepcybo_lite_ros2
git pull
```

## 1. 启动前检查

确认数据盘已挂载并可写：

```bash
findmnt /media/stvli/0EE4-E658
df -h /media/stvli/0EE4-E658
test -w /media/stvli/0EE4-E658 && echo OK
```

确认真机侧已经发布必要 ROS2 topic：

```bash
source /opt/ros/jazzy/setup.bash
source /home/stvli/Desktop/bar_ws/install/setup.bash
ros2 topic list | grep -E 'deepcybo|slave/lite|remote_policy_controller'
```

RoboDriver 启动前最好已经能看到：

```text
/deepcybo/lite/camera/head/image_raw/compressed
/deepcybo/lite/camera/wrist_left/image_raw/compressed
/deepcybo/lite/camera/wrist_right/image_raw/compressed
/slave/lite/joint_states
/slave/remote_policy_controller/command
```

## 2. 启动 RoboDriver 录制端

打开一个终端运行：

```bash
cd /home/stvli/Desktop/robodriver_ws/src/RoboDriver

source ~/miniconda3/etc/profile.d/conda.sh
conda activate robodriver_py312
source /opt/ros/jazzy/setup.bash
source /home/stvli/Desktop/bar_ws/install/setup.bash

export DEEPCYBO_LITE_DATA_ROOT=/media/stvli/0EE4-E658

python -m robodriver.scripts.run --robot.type=deepcybo-lite-aio-ros2
```

看到以下日志后，RoboDriver 录制端已就绪：

```text
[连接成功] 所有设备已就绪
[ROS2 Collection] Bridge node initialized
```

如果真机 topic 还没起来，`connect()` 可能超时退出。此时先拉起真机控制栈，再重新执行上述命令。

## 3. 启动 Lite 终端控制 FSM

另开一个终端运行当前 Lite 控制入口：

```bash
bash /home/stvli/Desktop/bar_ws/src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh
```

这个脚本会通过 ROS2 控制机器人状态机，并向 RoboDriver 发布录制指令。

## 4. ROS2 录制指令语义

RoboDriver 订阅以下三个 topic：

```text
/to_robodriver/start_collect
/to_robodriver/finish_collect
/to_robodriver/affirm_to_collect
```

控制语义如下：

```text
start_collect=true      开始录制
start_collect=false     清除开始录制 latch

finish_collect=true     结束当前片段，写入 pending episode
finish_collect=false    清除结束录制 latch

affirm_to_collect=true  保存 pending episode
affirm_to_collect=false 丢弃 pending episode
```

正常流程：

```text
进入遥操 -> start_collect=true
结束片段 -> finish_collect=true
确认保存 -> affirm_to_collect=true
确认丢弃 -> affirm_to_collect=false
```

## 5. 手动发布录制指令

不用 FSM 时，可手动发布：

```bash
source /opt/ros/jazzy/setup.bash
source /home/stvli/Desktop/bar_ws/install/setup.bash

ros2 topic pub --once --qos-reliability reliable --qos-durability transient_local \
  /to_robodriver/start_collect std_msgs/msg/Bool "{data: true}"

ros2 topic pub --once --qos-reliability reliable --qos-durability transient_local \
  /to_robodriver/finish_collect std_msgs/msg/Bool "{data: true}"

ros2 topic pub --once --qos-reliability reliable --qos-durability transient_local \
  /to_robodriver/affirm_to_collect std_msgs/msg/Bool "{data: true}"
```

如需丢弃，将最后一条改为：

```bash
ros2 topic pub --once --qos-reliability reliable --qos-durability transient_local \
  /to_robodriver/affirm_to_collect std_msgs/msg/Bool "{data: false}"
```

## 6. 观察录制状态

RoboDriver 终端中重点看这些日志：

```text
Collection state: IDLE -> COLLECTING
ROS2 collection started

Collection state: COLLECTING -> SAVING
save_episode succcess

Collection state: SAVING -> WAITING_AFFIRM
ROS2 collection kept
Collection state: WAITING_AFFIRM -> IDLE
```

如果看到 `ignored_false` 或 `no_pending_recording`，通常是 latch 清除或重复确认，不一定是错误。

## 7. 数据落盘位置

默认根目录：

```text
/media/stvli/0EE4-E658
```

典型路径：

```text
/media/stvli/0EE4-E658/YYYYMMDD/user/deepcybo_lite_bilateral_YYYYMMDD/deepcybo_lite_bilateral_YYYYMMDD_ros2_*
```

查看最新数据：

```bash
find /media/stvli/0EE4-E658 -maxdepth 4 -type d -name 'deepcybo_lite_bilateral_*' | sort | tail
```

## 8. 快速验收

每段保存后至少确认：

```bash
du -sh /media/stvli/0EE4-E658/YYYYMMDD/user/deepcybo_lite_bilateral_YYYYMMDD/*
```

RoboDriver 日志中应出现：

```text
动作数据形状: (..., 16)
file_integrity: pass
camera_frame_rate: pass
action_frame_rate: pass
```

采集端 parquet 默认不包含三路图片列，这是 RoboDriver 采集端的轻量格式：

```text
observation.images.image_head
observation.images.image_wrist_left
observation.images.image_wrist_right
```

验收时应确认：

```text
meta/info.json 声明三路 image feature
meta/info.json 中 image_path = images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg
parquet 中 frame_index 连续
每路 images/<image_key>/episode_xxxxxx/frame_xxxxxx.jpg 数量等于 parquet rows
```

训练或 OpenPI 导入前如果需要 Hugging Face `Image` columns，由 RoboDriver-Server 或离线转换脚本基于 `image_path` 模板补齐。不要在采集端把图片 bytes 写进 parquet。

## 9. 停止

录制结束并确认保存或丢弃后，在 RoboDriver 终端按：

```text
Ctrl+C
```

确认 RoboDriver 节点已退出：

```bash
ros2 node list | grep -E 'robodriver|deepcybo_lite_ros2_driver' || echo clean
```

## 10. 常见问题

### `connect()` 超时

说明 RoboDriver 未收到三路相机、`/slave/lite/joint_states` 或 `/slave/remote_policy_controller/command`。先确认真机控制栈已启动，再重新拉起 RoboDriver。

### Server 连接失败

Lite 紧急采集阶段允许 RoboDriver-Server 不在线。看到以下日志通常可以继续：

```text
RoboDriver-Server unavailable, continuing Lite ROS2 collection offline
```

### OpenPI 报 parquet 没有图片列

这是训练前转换阶段的问题，不是采集失败。先确认图片文件数量与 parquet rows 对齐，再运行 RoboDriver-Server / 离线转换管线，把 `frame_index` 映射为 Hugging Face `Image` columns。`lerobot_pic_debug/add_lerobot_image_paths.py` 是当前可用的最小修复逻辑。

### 真机上不要运行 mock

真机采集时不要运行 `deepcybo-lite-mock-ros2`，避免 mock topic 与真实 topic 混杂。
