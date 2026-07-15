#!/usr/bin/env python3
"""DeepCybo Lite — PI05 推理部署客户端入口。

用法::

    conda activate robodriver_py312
    source /opt/ros/jazzy/setup.bash
    source /home/stvli/Desktop/bar_ws/install/setup.bash
    python scripts/deploy.py --server-url http://192.168.1.100:9090 --prompt "将红色方块捡起来"

前置条件:
    1. 从臂已通过 bar_ros2 的 deploy_slave.sh 拉起
    2. 相机已通过 cam_ros2_ws 的 camera_selection 拉起
    3. PI05 推理服务端已在 GPU 机器上运行（或使用 mock server）
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


async def main(server_url: str, prompt: str, fps: int, chunk_size: int) -> None:
    # ---- ROS2 初始化 ----
    rclpy.init()
    executor = MultiThreadedExecutor()

    # ---- 机器人 ----
    # 延迟导入以避免找不到 bar_msgs 时崩溃
    from robodriver.robots.robots.robodriver_robot_deepcybo_lite_aio_ros2.robot import (
        DeepcyboLiteAioRos2Robot,
    )
    from robodriver.robots.robots.robodriver_robot_deepcybo_lite_aio_ros2.config import (
        DeepcyboLiteAioRos2RobotConfig,
    )
    from robodriver.core.inference_deployment import InferenceDeploymentLoop

    config = DeepcyboLiteAioRos2RobotConfig()
    config.control_fps = fps
    config.camera_fps = fps

    robot = DeepcyboLiteAioRos2Robot(config)
    executor.add_node(robot.get_node())

    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # ---- 连接机器人 ----
    logger.info("正在连接机器人...")
    robot.connect()
    logger.info("机器人已连接")

    # ---- 推理部署主循环 ----
    loop = InferenceDeploymentLoop(
        robot=robot,
        server_url=server_url,
        fps=fps,
        chunk_size=chunk_size,
        prompt=prompt,
    )

    try:
        await loop.run()
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        if robot.is_connected:
            robot.disconnect()
        executor.shutdown()
        if spin_thread.is_alive():
            spin_thread.join(timeout=2.0)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepCybo Lite — PI05 推理部署客户端"
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:9090",
        help="PI05 推理服务端地址（默认 http://127.0.0.1:9090）",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="任务自然语言指令",
    )
    parser.add_argument("--fps", type=int, default=30, help="控制频率")
    parser.add_argument(
        "--chunk-size", type=int, default=50, help="每段 action chunk 帧数"
    )
    args = parser.parse_args()

    asyncio.run(main(args.server_url, args.prompt, args.fps, args.chunk_size))
