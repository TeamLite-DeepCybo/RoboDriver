# Ideal LeRobot Data Format for DeepCybo Lite

本文档描述 DeepCybo Lite 在理想情况下由 RoboDriver 落盘出来的 LeRobot / DoRobot 数据格式。这里的“理想情况”指：

- `deepcybo-lite-aio-ros2` 已连接成功。
- `/slave/lite/joint_states` 提供完整 16 维 follower 状态。
- `/slave/remote_policy_controller/command` 提供完整 16 维 leader/action 指令。
- 三路相机稳定发布，并与 30 Hz 采集频率基本一致。
- `finish_collect=true` 后 `Record.save()` 正常完成，`affirm_to_collect=true` 后数据被保留。

## 1. 数据根目录

ROS2 FSM 采集入口默认写到：

```text
/media/stvli/0EE4-E658/
```

可用环境变量覆盖：

```bash
export DEEPCYBO_LITE_DATA_ROOT=/media/stvli/0EE4-E658
```

理想路径形态：

```text
$DEEPCYBO_LITE_DATA_ROOT/YYYYMMDD/user/{task_name}_{task_id}/{repo_id}/
```

其中 ROS2 FSM 自动生成的默认字段为：

```text
task_name    = deepcybo_lite_bilateral
task_id      = YYYYMMDD
task_data_id = ros2_YYYYMMDD_HHMMSS_0001
repo_id      = {task_name}_{task_id}_{task_data_id}
```

示例：

```text
/media/stvli/0EE4-E658/20260617/user/deepcybo_lite_bilateral_20260617/deepcybo_lite_bilateral_20260617_ros2_20260617_101112_0001/
```

## 2. 目录结构

`use_videos=False` 时，DeepCybo Lite 当前默认保留图片帧，理想结构如下：

```text
{repo_id}/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
├── images/
│   ├── image_head/
│   │   └── episode_000000/
│   │       ├── frame_000000.jpg
│   │       ├── frame_000001.jpg
│   │       └── ...
│   ├── image_wrist_left/
│   │   └── episode_000000/
│   │       ├── frame_000000.jpg
│   │       └── ...
│   └── image_wrist_right/
│       └── episode_000000/
│           ├── frame_000000.jpg
│           └── ...
└── meta/
    ├── common_record.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    ├── info.json
    ├── op_dataid.jsonl
    └── tasks.jsonl
```

如果未来 `use_videos=True`，图像会编码到：

```text
videos/chunk-000/{video_key}/episode_000000.mp4
```

当前 Lite 产品采集阶段建议优先使用图片帧，便于排障和抽检。

## 3. Episode Parquet

每个 episode 对应一个 parquet：

```text
data/chunk-000/episode_000000.parquet
```

采集端 parquet 的理想列包含：

| 列名 | 含义 | 理想 dtype / shape |
|---|---|---|
| `observation.state` | follower 当前状态，Lite 16 维 | `float32[16]` |
| `action` | leader / 目标控制指令，Lite 16 维 | `float32[16]` |
| `timestamp` | episode 内时间戳 | `float32` |
| `frame_index` | episode 内帧序号 | `int64` |
| `episode_index` | episode 序号 | `int64` |
| `index` | dataset 全局帧序号 | `int64` |
| `task_index` | task id 映射 | `int64` |

对于 10 秒、30 Hz 的理想 smoke recording：

```text
len(parquet) = 300
timestamp ~= frame_index / 30
frame_index = 0..299
episode_index 全部为 0
```

图像不直接进入采集端 parquet。`meta/info.json` 仍声明三路 image feature，并通过 `image_path` 模板把 `episode_index + frame_index` 映射到 `images/` 下的实际图片文件。训练前如果某个 pipeline 需要 Hugging Face `Image` columns，应在后处理阶段补齐。

## 4. 16 维向量顺序

`observation.state` 和 `action` 的 16 维顺序一致：

| index | joint |
|---:|---|
| 0 | `left_shoulder_pitch` |
| 1 | `left_shoulder_roll` |
| 2 | `left_shoulder_yaw` |
| 3 | `left_elbow_pitch` |
| 4 | `left_wrist_yaw` |
| 5 | `left_wrist_roll` |
| 6 | `left_wrist_pitch` |
| 7 | `right_shoulder_pitch` |
| 8 | `right_shoulder_roll` |
| 9 | `right_shoulder_yaw` |
| 10 | `right_elbow_pitch` |
| 11 | `right_wrist_yaw` |
| 12 | `right_wrist_roll` |
| 13 | `right_wrist_pitch` |
| 14 | `left_gripper` |
| 15 | `right_gripper` |

语义对应：

```text
observation.state[i] = follower_{joint_i}.pos
action[i]            = leader_{joint_i}.pos
```

注意：夹爪数据挂在原 14 维 arm joints 后面，顺序为左夹爪、右夹爪。

## 5. 相机数据

三路相机均属于 observation，不进入 action：

| dataset key | ROS2 topic | 理想尺寸 | 频率 |
|---|---|---:|---:|
| `image_head` | `/deepcybo/lite/camera/head/image_raw/compressed` | 640x480 RGB | 30 Hz |
| `image_wrist_left` | `/deepcybo/lite/camera/wrist_left/image_raw/compressed` | 640x480 RGB | 30 Hz |
| `image_wrist_right` | `/deepcybo/lite/camera/wrist_right/image_raw/compressed` | 640x480 RGB | 30 Hz |

图片落盘路径模板：

```text
images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg
```

理想情况下，每路相机图片数量等于 parquet 行数。例如 10 秒 30 Hz：

```text
images/image_head/episode_000000/*.jpg        -> 300 张
images/image_wrist_left/episode_000000/*.jpg  -> 300 张
images/image_wrist_right/episode_000000/*.jpg -> 300 张
```

## 6. meta/info.json

`meta/info.json` 是 dataset 的主 schema。理想关键字段应表达：

```json
{
  "fps": 30,
  "total_episodes": 1,
  "total_frames": 300,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [16],
      "names": [
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
        "right_gripper"
      ]
    },
    "action": {
      "dtype": "float32",
      "shape": [16],
      "names": [
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
        "right_gripper"
      ]
    }
  }
}
```

实际 `features` 中还会包含 `timestamp`、`frame_index`、`episode_index`、`index`、`task_index`、三路 image feature，以及 LeRobot / DoRobot 版本字段。注意：三路 image feature 出现在 `meta/info.json`，但采集端 episode parquet 默认不包含 `observation.images.*` 列。

## 7. meta/*.jsonl

### episodes.jsonl

每个 episode 一行。理想单段采集：

```json
{"episode_index": 0, "tasks": [...], "length": 300}
```

核心检查点：

```text
length == parquet 行数 == 每路相机图片数
```

### episodes_stats.jsonl

每个 episode 一行统计信息，至少应包含：

```text
stats.action.min/max/mean/std
stats.observation.state.min/max/mean/std
```

其中 `action` 和 `observation.state` 的统计数组长度应为 16。

### tasks.jsonl

任务文本映射。ROS2 FSM 默认任务名：

```text
deepcybo_lite_bilateral
```

实际可通过环境变量改：

```bash
export DEEPCYBO_LITE_TASK_NAME=your_task_name
export DEEPCYBO_LITE_TASK_ID=your_task_id
```

### op_dataid.jsonl

RoboDriver 额外记录 task data id 与 episode index 的映射：

```json
{"episode_index": 0, "dataid": "ros2_20260617_101112_0001", "machine_id": "deepcybo-lite-aio-ros2"}
```

该文件用于后续按 `task_data_id` 查询、回放、删除或上报。

### common_record.json

首个 episode 保存公共任务信息：

```json
{
  "task_id": "20260617",
  "task_name": "deepcybo_lite_bilateral",
  "machine_id": "deepcybo-lite-aio-ros2"
}
```

## 8. ROS2 FSM 保存 / 丢弃后的理想状态

### Y / 保存

当 FSM 发布：

```text
/to_robodriver/finish_collect=true
/to_robodriver/affirm_to_collect=true
```

理想结果：

- parquet 已写入 `data/chunk-000/episode_000000.parquet`
- 三路图片已写入 `images/.../episode_000000/`
- `meta/info.json`、`episodes.jsonl`、`episodes_stats.jsonl`、`tasks.jsonl` 已更新
- `meta/op_dataid.jsonl` 包含当前 `task_data_id`
- `Coordinator` 状态回到 `IDLE`

### N / 丢弃

当 FSM 发布：

```text
/to_robodriver/finish_collect=true
/to_robodriver/affirm_to_collect=false
```

理想结果：

- pending episode 被删除
- 对单 episode dataset，整个 `{repo_id}/` 目录可能被删除
- `Coordinator` 状态回到 `IDLE`
- 下一段采集会生成新的 `task_data_id`

## 9. 快速验收清单

以 10 秒 30 Hz 单段为例：

```text
meta/info.json
  fps = 30
  total_episodes = 1
  total_frames = 300
  features.action.shape = [16]
  features.observation.state.shape = [16]

data/chunk-000/episode_000000.parquet
  rows = 300
  action 每行长度 = 16
  observation.state 每行长度 = 16
  timestamp 单调递增
  frame_index = 0..299
  不包含 observation.images.* 列

images/
  image_head/episode_000000/*.jpg        = 300
  image_wrist_left/episode_000000/*.jpg  = 300
  image_wrist_right/episode_000000/*.jpg = 300

meta/op_dataid.jsonl
  包含当前 task_data_id -> episode_index 0
```

## 10. 常见异常形态

| 异常 | 说明 |
|---|---|
| `action.shape != [16]` | leader/action 维度未补夹爪或 schema 不一致 |
| `observation.state.shape != [16]` | follower 状态未补夹爪或 JointState 缺 canonical joint |
| 某路图片数少于 parquet 行数 | 相机掉帧、同步异常或 connect 后图像缓存不稳定 |
| 采集端 parquet 中出现图片 bytes | 采集端错误地 embed 了图像，应回退到帧索引 + 外部图片格式 |
| OpenPI 直接读取采集端 parquet 报缺少 image columns | 训练前转换未执行，应由 Server/离线脚本补 Hugging Face `Image` columns |
| `action` 全 0 或长时间重复 | leader command 没更新或映射错误 |
| `timestamp` 非单调 | Record loop 或数据写入异常 |
| `op_dataid.jsonl` 缺失当前 dataid | `Record.save()` 未完整完成或 save 后异常退出 |

## 11. 最小读取示例

```python
from pathlib import Path
import json
import pandas as pd

root = Path("/media/stvli/0EE4-E658/20260617/user/.../{repo_id}")

info = json.loads((root / "meta/info.json").read_text())
df = pd.read_parquet(root / "data/chunk-000/episode_000000.parquet")

print(info["fps"], info["total_frames"])
print(len(df))
print(df["action"].iloc[0])
print(df["observation.state"].iloc[0])
```

读取后应看到：

```text
len(df) == info["total_frames"]
len(df["action"].iloc[0]) == 16
len(df["observation.state"].iloc[0]) == 16
```
