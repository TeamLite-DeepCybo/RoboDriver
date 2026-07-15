#!/usr/bin/env python3
"""DeepCybo Lite — PI05 推理部署客户端入口（含从臂控制器状态机）。

状态机流程::

    ZERO_TORQUE → DAMPING(10s) → STANDBY → [服务器就绪+Enter] → REMOTE_POLICY
    REMOTE_POLICY: 持续请求 chunk → 回放 → 再请求 → ... → [Ctrl+C]
    → DAMPING_SETTLE(10s) → ZERO_TORQUE → 下一轮或优雅退出

用法::

    conda activate robodriver_py312
    source /opt/ros/jazzy/setup.bash
    source ~/Desktop/bar_ws/install/setup.bash
    python scripts/deploy.py --server-url http://192.168.1.100:9090 --prompt "任务指令"
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


async def _run_inference_loop(
    robot,
    server_url: str,
    prompt: str,
    fps: int,
    chunk_size: int,
) -> None:
    """REMOTE_POLICY 阶段的推理主循环：持续请求 chunk 并回放。

    退出条件：Ctrl+C 或连续推理失败超阈值。
    """
    from robodriver.core.inference_client import InferenceClient
    from robodriver.core.inference_deployment import InferenceDeploymentLoop

    loop_obj = InferenceDeploymentLoop(
        robot=robot,
        server_url=server_url,
        fps=fps,
        chunk_size=chunk_size,
        prompt=prompt,
    )

    # 复用 InferenceDeploymentLoop 的主循环逻辑
    await loop_obj.client.start()
    loop_obj.replayer.start()
    loop_obj._running = True

    print('\n  [推理中] 持续请求 action chunk，按 Ctrl+C 停止...\n')

    try:
        while loop_obj._running:
            await loop_obj._tick()
    except KeyboardInterrupt:
        print('\n  [推理中] 收到中断信号')
    finally:
        loop_obj._running = False
        loop_obj.replayer.stop()
        await loop_obj.client.stop()

    logger.info(
        f'[推理] 本轮完成: requests={loop_obj._total_requests} '
        f'frames={loop_obj.replayer.frames_sent}'
    )


async def main(server_url: str, prompt: str, fps: int, chunk_size: int) -> None:
    rclpy.init()
    executor = MultiThreadedExecutor()

    # ---- 机器人 ----
    from robodriver.robots.robots.robodriver_robot_deepcybo_lite_aio_ros2.robot import (
        DeepcyboLiteAioRos2Robot,
    )
    from robodriver.robots.robots.robodriver_robot_deepcybo_lite_aio_ros2.config import (
        DeepcyboLiteAioRos2RobotConfig,
    )
    from robodriver.core.slave_controller_fsm import SlaveControllerFsm

    config = DeepcyboLiteAioRos2RobotConfig()
    config.control_fps = fps
    config.camera_fps = fps

    robot = DeepcyboLiteAioRos2Robot(config)
    executor.add_node(robot.get_node())

    # ---- FSM ----
    fsm = SlaveControllerFsm(robot.get_node())

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # ---- 连接机器人 ----
    logger.info('正在连接机器人...')
    robot.connect()
    logger.info('机器人已连接')

    print('''
╔══════════════════════════════════════════════════════╗
║   DeepCybo Lite — PI05 推理部署                      ║
║   从臂控制器状态机已就绪                               ║
║   流程: DAMPING → STANDBY → READY → REMOTE_POLICY    ║
║   双 Ctrl+C 优雅退出                                  ║
╚══════════════════════════════════════════════════════╝
''')

    try:
        while True:
            # ---- 1. 推进到 READY ----
            fsm.step_to_ready()

            # ---- 2. 等待服务器 → REMOTE_POLICY ----
            result = fsm.wait_and_enter_remote(server_url)
            if result != 'REMOTE_POLICY':
                continue  # 服务器未就绪或用户跳过，回到 step_to_ready 下一轮

            # ---- 3. 推理循环 ----
            try:
                await _run_inference_loop(
                    robot, server_url, prompt, fps, chunk_size
                )
            except KeyboardInterrupt:
                pass

            # ---- 4. 任务完成 → DAMPING_SETTLE → ZERO_TORQUE ----
            fsm.settle_and_reset()

    except KeyboardInterrupt:
        print('\n\n[主循环] 收到中断，开始优雅退出...')

    finally:
        fsm.graceful_exit()
        if robot.is_connected:
            robot.disconnect()
        executor.shutdown()
        if spin_thread.is_alive():
            spin_thread.join(timeout=3.0)
        if rclpy.ok():
            rclpy.shutdown()

    print('\n推理部署已安全退出。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DeepCybo Lite — PI05 推理部署客户端'
    )
    parser.add_argument(
        '--server-url',
        default='http://127.0.0.1:9090',
        help='PI05 推理服务端地址',
    )
    parser.add_argument(
        '--prompt',
        default='',
        help='任务自然语言指令',
    )
    parser.add_argument('--fps', type=int, default=30, help='控制频率')
    parser.add_argument(
        '--chunk-size', type=int, default=50, help='每段 action chunk 帧数'
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.server_url, args.prompt, args.fps, args.chunk_size))
    except KeyboardInterrupt:
        pass
