"""DeepCybo Lite — 推理部署用从臂控制器状态机。

与 `bilateral_fsm_loop.py` 的差异：
  - 只管理从臂（无主臂、无桥接）
  - 任务间加入 zero_torque 释放环节
  - 远程策略阶段前等待推理服务端就绪
  - 远程策略阶段内持续请求多段 action chunk

状态图::

    ZERO_TORQUE ──→ DAMPING ──(10s)──→ STANDBY ──(ramp完成)──→ READY
    READY ──(服务器就绪+Enter)──→ REMOTE_POLICY ──(任务完成/Ctrl+C)──→ DAMPING_SETTLE
    DAMPING_SETTLE ──(10s)──→ ZERO_TORQUE ──→ DAMPING ──→ ...   (下一轮)
    任意状态 ──(优雅退出)──→ DAMPING_SETTLE ──→ ZERO_TORQUE ──→ EXITING
"""

from __future__ import annotations

import sys
import time
from enum import Enum
from typing import Optional

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.node import Node
from bar_msgs.msg import StandbyState

import logging_mp

logger = logging_mp.getLogger(__name__)

MODE_CONTROLLERS = (
    'zero_torque_controller',
    'damping_controller',
    'standby_controller',
    'remote_policy_controller',
)

# 基础设施控制器 — 始终保持 active，不参与切换
PROTECTED_CONTROLLERS = (
    "joint_state_broadcaster",
)


STANDBY_TOPIC = '/slave/standby_controller/state'


class FsmState(str, Enum):
    ZERO_TORQUE = "ZERO_TORQUE"
    DAMPING = "DAMPING"
    STANDBY = "STANDBY"
    READY = "READY"
    REMOTE_POLICY = "REMOTE_POLICY"
    DAMPING_SETTLE = "DAMPING_SETTLE"
    EXITING = "EXITING"


class SlaveControllerFsm:
    """从臂控制器状态机。

    管理 /slave/controller_manager 下的控制器切换，
    并在 REMOTE_POLICY 阶段将控制权交给推理主循环。
    """

    CM = '/slave/controller_manager'
    DAMPING_DURATION_S = 10.0       # zero_torque → damping 预置运动时长
    DAMPING_SETTLE_DURATION_S = 10.0  # remote_policy → zero_torque 间阻尼缓冲

    def __init__(self, node: Node, auto_mode: bool = False):
        self._node = node
        self.state = FsmState.ZERO_TORQUE
        self._exiting = False
        self._cycle_count = 0
        self._auto_mode = auto_mode

        # ---- ROS2 服务客户端 ----
        self._switch_cli = node.create_client(
            SwitchController, f'{self.CM}/switch_controller'
        )
        self._list_cli = node.create_client(
            ListControllers, f'{self.CM}/list_controllers'
        )

        # ---- Standby 状态订阅 ----
        self._standby_msg: Optional[StandbyState] = None
        self._standby_activated_at: Optional[rclpy.time.Time] = None
        node.create_subscription(
            StandbyState,
            STANDBY_TOPIC,
            self._on_standby,
            rclpy.qos.QoSProfile(
                durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                depth=1,
            ),
        )

    # ------------------------------------------------------------------
    # Standby 回调
    # ------------------------------------------------------------------
    def _on_standby(self, msg: StandbyState) -> None:
        self._standby_msg = msg

    def _standby_finished_fresh(self) -> bool:
        msg = self._standby_msg
        if msg is None or not msg.is_finished:
            return False
        if self._standby_activated_at is None:
            return True
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        return stamp_ns >= self._standby_activated_at.nanoseconds

    # ------------------------------------------------------------------
    # 控制器操作
    # ------------------------------------------------------------------
    def _active_controller(self) -> Optional[str]:
        """返回当前 active 的模式控制器名。"""
        if not self._list_cli.wait_for_service(timeout_sec=2.0):
            return None
        future = self._list_cli.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=2.0)
        if not future.done():
            return None
        resp = future.result()
        for c in resp.controller:
            if c.name in MODE_CONTROLLERS and c.state == 'active':
                return c.name
        return None

    def _switch_to(self, target: str) -> None:
        """STRICT 语义切换控制器。

        当无活跃模式控制器时，列出全部活跃控制器并停用后再激活 target。
        """
        if not self._switch_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'SwitchController service not available at {self.CM}')

        active = self._active_controller()
        if active == target:
            logger.info(f'[FSM] {target} already active')
            return

        if target == 'standby_controller':
            self._standby_msg = None
            self._standby_activated_at = self._node.get_clock().now()

        if active is None:
            # 无模式控制器活跃，但 controller_manager 可能有其他活跃控制器
            # → 找出全部活跃控制器并停用，再激活 target
            logger.info(f'[FSM] (none) -> {target} (resolving all active controllers...)')
            for attempt in range(20):  # 最多等 10s（首次启动时序竞争）
                all_active = self._all_active_controllers()
                req = SwitchController.Request()
                req.deactivate_controllers = all_active
                req.activate_controllers = [target]
                req.strictness = SwitchController.Request.STRICT
                req.activate_asap = True
                future = self._switch_cli.call_async(req)
                rclpy.spin_until_future_complete(self._node, future, timeout_sec=2.0)
                if future.done():
                    resp = future.result()
                    if resp.ok:
                        logger.info(
                            f'[FSM] {all_active or "[]"} -> {target} '
                            f'(attempt {attempt + 1})'
                        )
                        return
                logger.info(
                    f'[FSM] retry {attempt + 1}/20: '
                    f'switch rejected (active={all_active}), waiting 0.5s...'
                )
                time.sleep(0.5)
            raise RuntimeError(f'[FSM] failed to activate {target} after 20 retries')
        else:
            logger.info(f'[FSM] {active} -> {target}')
            req = SwitchController.Request()
            req.deactivate_controllers = [active]
            req.activate_controllers = [target]
            req.strictness = SwitchController.Request.STRICT
            req.activate_asap = True
            future = self._switch_cli.call_async(req)
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=10.0)
            if not future.done():
                raise RuntimeError(f'[FSM] switch to {target} timed out')
            resp = future.result()
            if not resp.ok:
                raise RuntimeError(f'[FSM] switch rejected: {active} -> {target}')

    def _all_active_controllers(self) -> list:
        """返回 controller_manager 中全部 active 状态的控制器名（不限模式控制器）。"""
        if not self._list_cli.wait_for_service(timeout_sec=2.0):
            return []
        future = self._list_cli.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=2.0)
        if not future.done():
            return []
        resp = future.result()
        return [c.name for c in resp.controller if c.state == "active" and c.name not in PROTECTED_CONTROLLERS]

    def _wait_standby(self, timeout_s: float = 60.0) -> None:
        """等待 standby ramp 完成。"""
        logger.info('[FSM] waiting for standby ramp to finish...')
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._standby_finished_fresh():
                logger.info('[FSM] standby finished')
                return
        raise RuntimeError('[FSM] standby ramp timed out')

    # ------------------------------------------------------------------
    # 状态转换
    # ------------------------------------------------------------------
    def _enter_damping(self, duration: float) -> None:
        self._switch_to('damping_controller')
        print(f'\n  [DAMPING] 等待 {duration:.0f}s 预置运动...')
        for remaining in range(int(duration), 0, -1):
            print(f'  {remaining}s', end='\r', flush=True)
            time.sleep(1)
        print('  DAMPING 完成' + ' ' * 10)

    def _enter_standby(self) -> None:
        self._switch_to('standby_controller')
        self._wait_standby()

    def _wait_server_ready(self, health_url: str, timeout_s: float = 30.0) -> bool:
        """HTTP GET health check，阻塞等待服务端就绪。"""
        import urllib.request
        import json
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get('status') == 'ok':
                        print(f'\n  [READY] 推理服务端就绪: model={data.get("model_loaded", "?")}')
                        return True
                    print(f'  [READY] 服务端异常: {data}', end='\r')
            except Exception:
                print(f'  [READY] 等待服务端 {health_url} ...', end='\r')
            time.sleep(1)
        return False

    def _prompt_enter(self, message: str) -> None:
        """等待操作员按 Enter。Ctrl+C 触发优雅退出。"""
        while True:
            try:
                raw = input(message)
                if raw == '':
                    return
            except EOFError:
                raise
            except KeyboardInterrupt:
                raise

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    @property
    def is_remote_policy_active(self) -> bool:
        return self._active_controller() == 'remote_policy_controller'

    def step_to_ready(self) -> FsmState:
        """从当前状态推进到 READY（zero_torque → damping → standby → READY）。"""
        self._cycle_count += 1
        print(f'\n{"="*50}')
        print(f'  推理部署 第 {self._cycle_count} 轮')
        print(f'{"="*50}')

        # ZERO_TORQUE → DAMPING
        print('\n[ZERO_TORQUE → DAMPING]')
        self._switch_to('zero_torque_controller')
        time.sleep(0.5)
        self._enter_damping(self.DAMPING_DURATION_S)

        # DAMPING → STANDBY
        print('\n[DAMPING → STANDBY]')
        self._enter_standby()

        self.state = FsmState.READY
        print(f'\n  [READY] 从臂 standby 到位，等待推理服务端')
        return self.state

    def _publish_hold_to_remote_policy(self) -> None:
        """向 /slave/remote_policy_controller/command 发布当前位置保持指令。

        在切换控制器之前调用，防止 remote_policy_controller 激活后读到空指令
        导致机械臂垂落。
        """
        from bar_msgs.msg import MITCommand
        from robodriver_robot_deepcybo_lite_aio_ros2.config import (
            LITE_JOINT_NAMES,
        )

        node = self._node
        state_dim = len(LITE_JOINT_NAMES)

        recv = getattr(node, 'recv_follower', {})
        vec = recv.get('follower_arms')
        if vec is None:
            logger.warning('[FSM] 无 follower 关节数据，跳过 hold 指令')
            return

        msg = MITCommand()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.joint_names = list(LITE_JOINT_NAMES)
        msg.position = [float(v) for v in vec[:state_dim]]
        msg.velocity = [0.0] * state_dim
        msg.effort = [0.0] * state_dim
        msg.stiffness = [node.command_stiffness] * state_dim
        msg.damping = [node.command_damping] * state_dim
        node.publisher_command.publish(msg)

        logger.info(
            f'[FSM] hold 指令已发布 → /slave/remote_policy_controller/command '
            f'({state_dim} joints, stiffness={node.command_stiffness})'
        )

    def wait_and_enter_remote(self, server_url: str) -> FsmState:
        """READY → REMOTE_POLICY: 等待服务端就绪 + 操作员确认。"""
        health_url = f'{server_url.rstrip("/")}/api/v1/health'

        if not self._wait_server_ready(health_url):
            print('\n  [READY] 服务端未就绪，停留在 READY')
            return self.state

        if not self._auto_mode:
            try:
                self._prompt_enter(
                    f'\n  从臂 standby 到位，服务端已就绪。'
                    f'按 Enter 开始推理部署...'
                )
            except (KeyboardInterrupt, EOFError):
                print('\n  [READY] 用户取消')
                return self.state

        # ---- 保护：先切控制器再发 hold（subscriber 已存在才能收到） ----
        self._switch_to('remote_policy_controller')
        self._publish_hold_to_remote_policy()
        self.state = FsmState.REMOTE_POLICY
        print(f'\n  [REMOTE_POLICY] 从臂已就绪，开始接收推理指令\n')
        return self.state

    def settle_and_reset(self) -> FsmState:
        """REMOTE_POLICY → DAMPING_SETTLE → ZERO_TORQUE。"""
        print(f'\n[DAMPING_SETTLE]')
        self._enter_damping(self.DAMPING_SETTLE_DURATION_S)

        print('\n[DAMPING_SETTLE → ZERO_TORQUE]')
        self._switch_to('zero_torque_controller')
        self.state = FsmState.ZERO_TORQUE
        return self.state

    def graceful_exit(self) -> None:
        """安全退出：damping (10s) → zero_torque。"""
        if self._exiting:
            return
        self._exiting = True
        print('\n\n[优雅退出] 正在将双臂切到 damping ...')
        try:
            self._switch_to('damping_controller')
        except Exception as e:
            print(f'[WARN] 切 damping 失败: {e}')

        print(f'[优雅退出] 保持 damping {self.DAMPING_SETTLE_DURATION_S:.0f}s ...')
        try:
            time.sleep(self.DAMPING_SETTLE_DURATION_S)
        except KeyboardInterrupt:
            print('[WARN] 退出等待被中断，继续切 zero_torque')

        print('[优雅退出] 正在切 zero_torque ...')
        try:
            self._switch_to('zero_torque_controller')
        except Exception as e:
            print(f'[WARN] 切 zero_torque 失败: {e}')

        self.state = FsmState.EXITING
        print('[优雅退出] 完成。')
