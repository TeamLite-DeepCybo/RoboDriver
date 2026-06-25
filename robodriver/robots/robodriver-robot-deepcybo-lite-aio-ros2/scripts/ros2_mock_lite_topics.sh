#!/usr/bin/env bash
# 无真机录制 smoke test 的 ROS2 输入节点：
#   - 50Hz 发布 /slave/lite/joint_states (JointState, 16 维关节/夹爪)
#   - 50Hz 发布 /slave/remote_policy_controller/command (bar_msgs/MITCommand, 16 维关节/夹爪)
#   - 将 /camera1/image_raw/compressed 复制到 RoboDriver 期望的三路相机话题
#
# 用法：
#   source /opt/ros/jazzy/setup.bash
#   source /home/stvli/Desktop/bar_ws/install/setup.bash
#   conda activate robodriver_py312
#   bash scripts/ros2_mock_lite_topics.sh

set -euo pipefail

python -m robodriver_robot_deepcybo_lite_aio_ros2.mock_recording "$@"
