import asyncio
from concurrent.futures import Future
from typing import Awaitable, Callable

import logging_mp
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool

from robodriver.core.coordinator import Coordinator

logger = logging_mp.getLogger(__name__)


class Ros2CollectionBridge(Node):
    def __init__(self, coordinator: Coordinator, loop: asyncio.AbstractEventLoop):
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
        self.create_subscription(
            Bool,
            "/to_robodriver/finish_collect",
            self._on_finish_collect,
            qos,
        )
        self.create_subscription(
            Bool,
            "/to_robodriver/affirm_to_collect",
            self._on_affirm_to_collect,
            qos,
        )

        logger.info("[ROS2 Collection] Bridge node initialized")

    def _submit(
        self,
        name: str,
        handler: Callable[[bool], Awaitable[dict]],
        value: bool,
    ) -> None:
        logger.info(f"[ROS2 Collection] {name}={value}")
        future = asyncio.run_coroutine_threadsafe(handler(value), self.loop)
        future.add_done_callback(lambda task: self._log_result(name, task))

    def _log_result(self, name: str, future: Future) -> None:
        try:
            result = future.result()
        except Exception as e:
            logger.exception(f"[ROS2 Collection] {name} failed: {e}")
            return
        logger.info(f"[ROS2 Collection] {name} result: {result}")

    def _on_start_collect(self, msg: Bool) -> None:
        self._submit(
            "start_collect",
            self.coordinator.handle_ros2_start_collect,
            bool(msg.data),
        )

    def _on_finish_collect(self, msg: Bool) -> None:
        self._submit(
            "finish_collect",
            self.coordinator.handle_ros2_finish_collect,
            bool(msg.data),
        )

    def _on_affirm_to_collect(self, msg: Bool) -> None:
        self._submit(
            "affirm_to_collect",
            self.coordinator.handle_ros2_affirm_to_collect,
            bool(msg.data),
        )
