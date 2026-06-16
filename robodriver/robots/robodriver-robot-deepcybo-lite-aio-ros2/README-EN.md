# robodriver-robot-deepcybo-lite-aio-ros2

**English** | [![简体中文](README.md)](README.md)

RoboDriver ROS2 integration package for the **DeepCybo Lite** dual-arm platform (`aio`: observation + action in one package, saved as LeRobot / DoRobot datasets).

For full topic contracts, vector layout, and the debug fault tree, see [Lite包使用说明.md](./Lite包使用说明.md) (Chinese; tables are self-explanatory).

---

## Overview

| Item | Value |
|------|--------|
| pip package | `robodriver_robot_deepcybo_lite_aio_ros2` |
| RoboDriver type | `deepcybo-lite-aio-ros2` |
| Stack | ROS2 **Jazzy** (team debug target; use `jazzy` paths instead of `humble`) |
| Body vector | Dual-arm **14 joints** + dual-gripper **2 values** |
| State / action vector | **16-D** each: the original 14 arm joints followed by `left_gripper`, `right_gripper` |
| Cameras | Head + left wrist + right wrist, **640×480**, 30 Hz |
| Joint / record rate limit | **30 Hz** (aligned with Record loop) |

The `bar_ws` control stack must publish `/slave/lite/joint_states` (`sensor_msgs/JointState`) and `/slave/remote_policy_controller/command` (`bar_msgs/MITCommand`). Camera topics are defined in `DeepcyboLiteRos2Topics` (`config.py`).

---

## Package layout

```text
robodriver-robot-deepcybo-lite-aio-ros2/
├── robodriver_robot_deepcybo_lite_aio_ros2/
│   ├── config.py
│   ├── node.py        # JointState/MITCommand subscribers, 16-D canonical vectors, replay publisher
│   ├── robot.py
│   └── status.py
├── scripts/
│   ├── record_external_topics.py
│   └── ros2_mock_lite_topics.sh
├── Lite包使用说明.md
├── README.md / README-EN.md
└── pyproject.toml
```

---

## Requirements

- Ubuntu + **ROS2 Jazzy** (`source /opt/ros/jazzy/setup.bash`)
- Python **≥ 3.10**
- [RoboDriver](https://github.com/FlagOpen/RoboDriver) installed
- Optional HMI: [RoboDriver-Server](https://github.com/FlagOpen/RoboDriver-Server) → `http://localhost:5805/hmi/`
- Same **`ROS_DOMAIN_ID`** on all terminals

```bash
export ROS_DOMAIN_ID=0
```

---

## Installation

```bash
cd /path/to/RoboDriver && pip install -e .

cd robodriver/robots/robodriver-robot-deepcybo-lite-aio-ros2
pip install -e .
```

`rclpy`, `message_filters`, and `bar_msgs` come from the ROS2 / `bar_ws` overlay, not pip.

---

## Default ROS2 topics

### Observation (feedback, 30 Hz)

| Topic | Type | Requirement |
|-------|------|-------------|
| `/slave/lite/joint_states` | `sensor_msgs/JointState` | `name` and `position` must contain the 16 canonical Lite joints |

### Action (command, 30 Hz)

| Topic | Type | Requirement |
|-------|------|-------------|
| `/slave/remote_policy_controller/command` | `bar_msgs/MITCommand` | `joint_names` and `position` must contain the same 16 canonical Lite joints; replay publishes to this topic |

### Cameras (30 Hz, `CompressedImage`)

| Topic | Dataset key |
|-------|-------------|
| `/deepcybo/lite/camera/head/image_raw/compressed` | `image_head` |
| `/deepcybo/lite/camera/wrist_left/image_raw/compressed` | `image_wrist_left` |
| `/deepcybo/lite/camera/wrist_right/image_raw/compressed` | `image_wrist_right` |

Override topics in `DeepcyboLiteRos2Topics` inside `config.py`.

---

## Quick start

### 1. Publish topics (robot or mock)

```bash
source /opt/ros/jazzy/setup.bash
bash scripts/ros2_mock_lite_topics.sh
```

The mock node publishes arm/gripper messages and can mirror one compressed camera stream to the three Lite camera topics. A full `connect()` still requires valid 16-D Lite data and all three camera streams.

### 2. Run RoboDriver

```bash
export DEEPCYBO_LITE_DATA_ROOT=/media/stvli/0EE4-E658
export ROBODRIVER_HOME=/media/stvli/0EE4-E658
python -m robodriver.scripts.run --robot.type=deepcybo-lite-aio-ros2
```

Expect a log like: all cameras ready + `leader_arms` / `follower_arms` 16-D vectors.

### 3. Collect (optional)

Start RoboDriver-Server + nginx, open `http://localhost:5805/hmi/`.
Lite script default data root: `/media/stvli/0EE4-E658/`.
HMI / Server data root: `$ROBODRIVER_HOME/dataset/`; in the product environment this resolves to `/media/stvli/0EE4-E658/dataset/`.

### 4. Playback

Use the HMI playback button or `robot.send_action()` → `node.ros_replay()` publishes a 16-D `MITCommand` and fills zero velocity / effort plus default stiffness / damping.

---

## Configuration

| File | Role |
|------|------|
| `config.py` | Motor/camera schema; `LITE_JOINT_NAMES`; `DeepcyboLiteRos2Topics`; `control_fps` / `camera_fps` = 30 |
| `node.py` | Topic subscriptions, 16-D canonical merge; missing joints or invalid dimensions → warning + cache cleared |
| `robot.py` | Raises if arm cache invalid when sampling |
| `status.py` | HMI device status (`leader_arms` / `follower_arms`) |

---

## Vector Layout

The current 16-D layout keeps the original 14 arm joints unchanged and appends `left_gripper`, then `right_gripper`.

---

## Troubleshooting

See **[Lite包使用说明.md §10](./Lite包使用说明.md)** — fault tree (install, ROS2, connect timeout, JointState length, cameras, Record, HMI, replay).

Quick checks:

```bash
ros2 topic list | grep -E 'slave/lite|remote_policy_controller|deepcybo/lite/camera'
ros2 topic hz /slave/lite/joint_states
ros2 topic echo /slave/remote_policy_controller/command --once --field joint_names
```

---

## Links

- [RoboDriver](https://github.com/FlagOpen/RoboDriver)
- [RoboDriver-Server](https://github.com/FlagOpen/RoboDriver-Server)
- [LeRobot](https://github.com/huggingface/lerobot)

---

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot)
- [RoboDriver](https://github.com/FlagOpen/RoboDriver)
