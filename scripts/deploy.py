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
from rclpy.signals import SignalHandlerOptions

import logging_mp

logging_mp.basicConfig(level=logging_mp.INFO)
logger = logging_mp.getLogger(__name__)


async def _run_inference_loop(
    robot,
    server_url: str,
    prompt: str,
    fps: int,
    chunk_size: int,
    debug_state: bool = False,
    norm2si: bool = True,
    spin_thread=None,
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
        debug_state=debug_state,
        norm2si=norm2si,
    )

    # 复用 InferenceDeploymentLoop 的主循环逻辑
    await loop_obj.client.start()
    loop_obj.publisher.start()
    loop_obj._running = True

    print('\n  [推理中] 持续请求 action chunk，按 Ctrl+C 停止...\n')

    try:
        while loop_obj._running:
            if spin_thread and not spin_thread.is_alive():
                logger.critical('[SPIN-DEAD] executor thread died!')
                break
            await loop_obj._tick()
    except KeyboardInterrupt:
        print('\n  [推理中] 收到中断信号')
    finally:
        loop_obj._running = False
        loop_obj.publisher.stop()
        await loop_obj.client.stop()

    logger.info(
        f'[推理] 本轮完成: requests={loop_obj._total_requests} '
        f'frames={loop_obj.publisher.frames_sent}'
    )


async def main(
    server_url: str,
    prompt: str,
    fps: int,
    chunk_size: int,
    auto_mode: bool = False,
    norm2si: bool = True,
) -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    executor = MultiThreadedExecutor()

    # ---- 机器人 ----
    from robodriver.robots.utils import make_robot_from_config
    from robodriver_robot_deepcybo_lite_aio_ros2.config import (
        DeepcyboLiteAioRos2RobotConfig,
    )
    from robodriver.core.slave_controller_fsm import SlaveControllerFsm

    config = DeepcyboLiteAioRos2RobotConfig()
    config.require_leader = False  # 部署模式不需要主臂
    config.control_fps = fps
    config.camera_fps = fps

    robot = make_robot_from_config(config)
    executor.add_node(robot.get_node())

    # ---- FSM ----
    fsm = SlaveControllerFsm(robot.get_node(), auto_mode=auto_mode)

    def _spin_safe():
        try:
            executor.spin()
        except Exception as e:
            logger.error(f'[Executor] spin crashed: {type(e).__name__}: {e}', exc_info=True)
    spin_thread = threading.Thread(target=_spin_safe, daemon=True)
    spin_thread.start()

    # ---- 连接机器人 ----
    logger.info('正在连接机器人...')
    robot.connect()
    logger.info('机器人已连接')
    if args.debug_joint:
        robot.robot_ros2_node.debug_joint = True
        logger.info('[DEBUG-JOINT] 关节状态回调调试已启用')

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
                    robot, server_url, prompt, fps, chunk_size,
                    debug_state=args.debug_state,
                    norm2si=norm2si,
                    spin_thread=spin_thread,
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
    parser.add_argument(
        '--test',
        action='store_true',
        help='测试模式：控制频率降为 10Hz，避免机械臂运动过于激烈',
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='自动模式：跳过 Enter 确认，直接进入推理',
    )
    parser.add_argument(
        '--debug-state',
        action='store_true',
        help='调试模式：每次请求前打印上传的 16 维关节状态',
    )
    parser.add_argument(
        '--debug-joint',
        action='store_true',
        help='调试模式：每次收到 joint_states 时打印前 3 个关节值',
    )
    parser.add_argument(
        '--no-norm2si',
        action='store_true',
        help='关闭临时夹爪归一化补丁（仅当模型已按 SI 米制训练时使用）',
    )
    args = parser.parse_args()

    fps = 10 if args.test else args.fps
    if args.test:
        print(f'[测试模式] 控制频率: {fps}Hz (正常频率的 1/3)')
    norm2si = not args.no_norm2si
    if norm2si:
        print(
            '[TEMP 归一化补丁] 夹爪 0..1 <-> 米制换算已启用；'
            'SI 数据集重训后请用 --no-norm2si 关闭并移除补丁'
        )
    try:
        asyncio.run(
            main(
                args.server_url,
                args.prompt,
                fps,
                args.chunk_size,
                args.auto,
                norm2si,
            )
        )
    except KeyboardInterrupt:
        print('\n用户中断。')
    except TimeoutError as e:
        print(f'\n[连接超时] {e}')
        print('可能原因：ROS2/DDS 尚未从上次运行中恢复。')
        print('请等待 15-30 秒后重新拉起。')
        import sys; sys.exit(1)
    except Exception as e:
        import traceback
        print(f'\n[异常] {e}')
        traceback.print_exc()
        import sys; sys.exit(1)
