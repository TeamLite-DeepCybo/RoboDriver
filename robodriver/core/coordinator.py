import asyncio
import queue
import shutil
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import aiohttp
import cv2
import logging_mp
import socketio
from lerobot.teleoperators import Teleoperator

from robodriver.core.recorder import Record, RecordConfig
from robodriver.core.replayer import DatasetReplayConfig, ReplayConfig, replay
from robodriver.core.ros2_collection_metadata import (
    build_lite_ros2_record_cmd,
    ensure_unique_ros2_record_dir,
    get_lite_record_ready_timeout,
    resolve_lite_collection_root,
)
from robodriver.dataset.dorobot_dataset import *
from robodriver.dataset.visual.visual_dataset import visualize_dataset
from robodriver.robots.daemon import Daemon
from robodriver.utils.constants import (
    DEFAULT_FPS,
    DOROBOT_DATASET,
    RERUN_WEB_PORT,
    RERUN_WS_PORT,
)
from robodriver.utils.data_file import check_disk_space, find_epindex_from_dataid_json
from robodriver.utils.utils import cameras_to_stream_json, get_current_git_branch

logger = logging_mp.get_logger(__name__)


class CollectionState(str, Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    WAITING_AFFIRM = "WAITING_AFFIRM"
    SAVING = "SAVING"
    DISCARDING = "DISCARDING"
    ERROR = "ERROR"


class Coordinator:
    def __init__(
        self,
        daemon: Daemon,
        teleop: Optional[Teleoperator],
        server_url="http://localhost:8088",
    ):
        self.server_url = server_url
        # 异步客户端
        self.sio = socketio.AsyncClient()
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=10)
        )

        self.daemon = daemon
        self.teleop = teleop

        self.running = False
        self.server_available = False
        self.last_heartbeat_time = 0
        self.heartbeat_interval = 2
        self.recording = False
        self.replaying = False
        self.saveing = False

        self.cameras = {"image_top": 1, "image_right": 2}

        # 注册异步回调
        self.sio.on("HEARTBEAT_RESPONSE", self.__on_heartbeat_response_handle)
        self.sio.on("connect", self.__on_connect_handle)
        self.sio.on("disconnect", self.__on_disconnect_handle)
        self.sio.on("robot_command", self.__on_robot_command_handle)

        self.record = None
        self.collection_state = CollectionState.IDLE
        self.pending_episode_index = None
        self.pending_save_data = None
        self.pending_record_cmd = None
        self.last_collection_state_change_time = time.time()
        self.ros2_collection_sequence = 0
        self._collection_lock = asyncio.Lock()

    ####################### Client Start/Stop ############################
    async def start(self):
        """启动客户端"""
        self.running = True
        try:
            await self.sio.connect(self.server_url)
        except Exception:
            self.running = False
            self.server_available = False
            raise
        self.server_available = True
        # 用 asyncio 任务发心跳
        asyncio.create_task(self.send_heartbeat_loop())

    async def stop(self):
        self.running = False
        if self.sio.connected:
            await self.sio.disconnect()
        await self.session.close()
        logger.info("异步客户端已停止")

    ####################### Client Handle ############################
    async def __on_heartbeat_response_handle(self, data):
        """心跳响应回调"""
        logger.info("收到心跳响应:", data)

    async def __on_connect_handle(self):
        """连接成功回调"""
        self.server_available = True
        logger.info("成功连接到服务器")

    async def __on_disconnect_handle(self):
        """断开连接回调"""
        self.server_available = False
        logger.info("与服务器断开连接")

    ####################### ROS2 Collection API ############################
    def _set_collection_state(self, state: CollectionState):
        if self.collection_state != state:
            logger.info(
                f"Collection state: {self.collection_state.value} -> {state.value}"
            )
        self.collection_state = state
        self.last_collection_state_change_time = time.time()

    def _collection_result(self, msg: str, data: Optional[dict] = None) -> dict:
        result = {
            "msg": msg,
            "state": self.collection_state.value,
        }
        if data is not None:
            result["data"] = data
        return result

    def _get_lite_collection_root(self) -> Path:
        try:
            from robodriver_robot_deepcybo_lite_aio_ros2.config import (
                DEFAULT_DATA_ROOT,
            )
        except Exception:
            DEFAULT_DATA_ROOT = DOROBOT_DATASET

        return resolve_lite_collection_root(DEFAULT_DATA_ROOT)

    def _prepare_lite_collection_root(self, root: Path) -> bool:
        root = root.expanduser()
        if str(root).startswith("/media/") and not root.exists():
            logger.error(
                f"Lite collection root does not exist, refusing to create mount path: {root}"
            )
            return False

        try:
            root.mkdir(parents=True, exist_ok=True)
            total, used, free = shutil.disk_usage(root)
        except Exception as e:
            logger.error(f"Cannot access Lite collection root {root}: {e}")
            return False

        free_gb = free // (2**30)
        if free_gb < 2:
            logger.warning(
                f"Lite collection root free space is below 2GB: {root}, free={free_gb}GB"
            )
            return False
        return True

    def _build_lite_ros2_record_cmd(self) -> dict:
        self.ros2_collection_sequence += 1
        return build_lite_ros2_record_cmd(
            sequence=self.ros2_collection_sequence,
            robot_name=getattr(self.daemon.robot, "name", "deepcybo-lite-aio-ros2"),
        )

    def _ensure_unique_ros2_record_dir(
        self, dataset_path: Path, msg: dict
    ) -> tuple[str, Path]:
        repo_id, target_dir, unique_msg = ensure_unique_ros2_record_dir(
            dataset_path, msg
        )
        msg.update(unique_msg)
        return repo_id, target_dir

    async def _wait_for_daemon_record_frame(self) -> bool:
        timeout_s = get_lite_record_ready_timeout()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if (
                self.daemon.get_observation() is not None
                and self.daemon.get_obs_action() is not None
            ):
                return True
            await asyncio.sleep(0.05)
        logger.error(
            "Timed out waiting for daemon observation/action before ROS2 collection."
        )
        return False

    async def handle_ros2_start_collect(self, value: bool) -> dict:
        if not value:
            logger.info("ROS2 start_collect=false received, latch cleared.")
            return self._collection_result("ignored_false")
        return await self.start_collection_from_ros2()

    async def handle_ros2_finish_collect(self, value: bool) -> dict:
        if not value:
            logger.info("ROS2 finish_collect=false received, latch cleared.")
            return self._collection_result("ignored_false")
        return await self.finish_collection_from_ros2()

    async def handle_ros2_affirm_to_collect(self, value: bool) -> dict:
        return await self.affirm_collection_from_ros2(keep=value)

    async def start_collection_from_ros2(self) -> dict:
        async with self._collection_lock:
            logger.info("处理 ROS2 开始采集命令...")
            if self.replaying:
                logger.warning("Replay is running, cannot start ROS2 collection.")
                return self._collection_result("replay_in_progress")

            if self.collection_state == CollectionState.COLLECTING:
                logger.info("ROS2 collection is already active, ignoring duplicate start.")
                return self._collection_result("already_collecting")

            if self.collection_state == CollectionState.WAITING_AFFIRM:
                logger.warning("Pending episode needs affirmation before next start.")
                return self._collection_result("pending_affirmation")

            if self.recording:
                logger.warning(
                    "A collection is already active outside ROS2, refusing ROS2 start."
                )
                return self._collection_result("recording_in_progress")

            dataset_path = self._get_lite_collection_root()
            if not self._prepare_lite_collection_root(dataset_path):
                return self._collection_result("storage_unavailable")

            if not await self._wait_for_daemon_record_frame():
                return self._collection_result("daemon_not_ready")

            msg = self._build_lite_ros2_record_cmd()
            repo_id, target_dir = self._ensure_unique_ros2_record_dir(
                dataset_path, msg
            )
            task_name = msg.get("task_name")
            record_cfg = RecordConfig(
                fps=DEFAULT_FPS,
                single_task=task_name,
                repo_id=repo_id,
                video=self.daemon.robot.use_videos,
                resume=False,
                root=target_dir,
            )
            self.record = Record(
                fps=DEFAULT_FPS,
                robot=self.daemon.robot,
                daemon=self.daemon,
                teleop=self.teleop,
                record_cfg=record_cfg,
                record_cmd=msg,
            )
            self.recording = True
            self.pending_episode_index = None
            self.pending_save_data = None
            self.pending_record_cmd = None
            self._set_collection_state(CollectionState.COLLECTING)
            self.record.start()

            logger.info(f"ROS2 collection started: repo_id={repo_id}, root={target_dir}")
            return self._collection_result(
                "success",
                {
                    "record_cmd": msg,
                    "repo_id": repo_id,
                    "root": str(target_dir),
                },
            )

    async def finish_collection_from_ros2(self) -> dict:
        async with self._collection_lock:
            logger.info("处理 ROS2 完成采集命令...")
            if self.replaying:
                logger.warning("Replay is running, cannot finish ROS2 collection.")
                return self._collection_result("replay_in_progress")

            if self.collection_state == CollectionState.IDLE and not self.recording:
                logger.info("No active ROS2 collection, ignoring stale finish.")
                return self._collection_result("no_active_recording")

            if self.collection_state == CollectionState.WAITING_AFFIRM:
                logger.info("ROS2 collection already saved, ignoring duplicate finish.")
                return self._collection_result("already_waiting_affirmation")

            if self.record is None:
                logger.error("ROS2 finish requested but no Record exists.")
                self.recording = False
                self._set_collection_state(CollectionState.ERROR)
                return self._collection_result("missing_record")

            self.saveing = True
            self._set_collection_state(CollectionState.SAVING)
            try:
                self.record.stop()
                save_data = await asyncio.to_thread(self.record.save)
                if save_data is None:
                    save_data = self.record.save_data

                self.recording = False
                self.pending_episode_index = self.record.last_record_episode_index
                self.pending_save_data = save_data
                self.pending_record_cmd = self.record.record_cmd
                self._set_collection_state(CollectionState.WAITING_AFFIRM)

                logger.info(
                    f"ROS2 collection saved and waiting affirmation: episode={self.pending_episode_index}"
                )
                return self._collection_result("success", save_data)
            except Exception as e:
                logger.exception(f"Failed to finish ROS2 collection: {e}")
                self.recording = False
                self._set_collection_state(CollectionState.ERROR)
                return self._collection_result("save_failed", {"error": str(e)})
            finally:
                self.saveing = False

    async def affirm_collection_from_ros2(self, keep: bool) -> dict:
        async with self._collection_lock:
            logger.info(f"处理 ROS2 采集确认命令: keep={keep}")
            if self.collection_state != CollectionState.WAITING_AFFIRM:
                logger.info("No pending ROS2 collection, ignoring stale affirmation.")
                return self._collection_result("no_pending_recording")

            if self.record is None:
                logger.error("ROS2 affirmation requested but no Record exists.")
                self._set_collection_state(CollectionState.ERROR)
                return self._collection_result("missing_record")

            data = self.pending_save_data
            if keep:
                logger.info(
                    f"ROS2 collection kept: episode={self.pending_episode_index}"
                )
                self.pending_episode_index = None
                self.pending_save_data = None
                self.pending_record_cmd = None
                self.record = None
                self._set_collection_state(CollectionState.IDLE)
                return self._collection_result("success", data)

            self._set_collection_state(CollectionState.DISCARDING)
            try:
                self.record.discard()
                logger.info("ROS2 collection discarded.")
                self.pending_episode_index = None
                self.pending_save_data = None
                self.pending_record_cmd = None
                self.record = None
                self._set_collection_state(CollectionState.IDLE)
                return self._collection_result("discarded", data)
            except Exception as e:
                logger.exception(f"Failed to discard ROS2 collection: {e}")
                self._set_collection_state(CollectionState.ERROR)
                return self._collection_result("discard_failed", {"error": str(e)})

    async def __on_robot_command_handle(self, data):
        """收到机器人命令回调"""
        logger.info("收到服务器命令:", data)
        global task_id
        global task_name
        global task_data_id
        global repo_id
        # 根据命令类型进行响应
        if data.get("cmd") == "video_list":
            logger.info("处理更新视频流命令...")
            response_data = cameras_to_stream_json(self.cameras)
            # 发送响应
            try:
                response = self.session.post(
                    f"{self.server_url}/robot/stream_info",
                    json=response_data,
                )
                logger.info(f"已发送响应 [{data.get('cmd')}]: {response_data}")
            except Exception as e:
                logger.error(f"发送响应失败 [{data.get('cmd')}]: {e}")

        elif data.get("cmd") == "start_collection":
            logger.info("处理开始采集命令...")
            msg = data.get("msg")

            if not check_disk_space(min_gb=2):
                logger.warning("存储空间不足,小于2GB,取消采集！")
                await self.send_response("start_collection", "存储空间不足,小于2GB")
                return

            if self.replaying == True:
                logger.warning("Replay is running, cannot start collection.")
                await self.send_response("start_collection", "fail")
                return

            if self.recording == True:
                self.record.stop()
                self.record.discard()
                self.recording = False

            self.recording = True

            task_id = msg.get("task_id")
            task_name = msg.get("task_name")
            task_data_id = msg.get("task_data_id")
            countdown_seconds = msg.get("countdown_seconds", 3)
            task_dir = f"{task_name}_{task_id}"
            repo_id = f"{task_name}_{task_id}_{task_data_id}"

            date_str = datetime.now().strftime("%Y%m%d")

            # 构建目标目录路径
            dataset_path = DOROBOT_DATASET

            git_branch_name = get_current_git_branch()
            target_dir = dataset_path / date_str / "user" / task_dir / repo_id
            # if "release" in git_branch_name or "main" in git_branch_name:
            #     target_dir = dataset_path / date_str / "user" / task_dir / repo_id
            # elif "dev" in git_branch_name:
            #     target_dir = dataset_path / date_str / "dev" / task_dir / repo_id
            # else:
            #     target_dir = dataset_path / date_str / "dev" / task_dir / repo_id

            # 判断是否存在对应文件夹以决定是否启用恢复模式
            resume = False

            # 检查数据集目录是否存在
            if not dataset_path.exists():
                logger.info(
                    f"Dataset directory '{dataset_path}' does not exist. Cannot resume."
                )
            else:
                # 检查目标文件夹是否存在且为目录
                if target_dir.exists() and target_dir.is_dir():
                    # resume = True
                    # logging.info(f"Found existing directory for repo_id '{repo_id}'. Resuming operation.")

                    logger.info(
                        f"Found existing directory for repo_id '{repo_id}'. Delete directory."
                    )
                    shutil.rmtree(target_dir)
                    time.sleep(0.5)  # make sure delete success.
                else:
                    logger.info(
                        f"No directory found for repo_id '{repo_id}'. Starting fresh."
                    )

            # resume 变量现在可用于后续逻辑
            logger.info(f"Resume mode: {'Enabled' if resume else 'Disabled'}")

            record_cfg = RecordConfig(
                fps=DEFAULT_FPS,
                single_task=task_name,
                repo_id=repo_id,
                video=self.daemon.robot.use_videos,
                resume=resume,
                root=target_dir,
            )
            self.record = Record(
                fps=DEFAULT_FPS,
                robot=self.daemon.robot,
                daemon=self.daemon,
                teleop=self.teleop,
                record_cfg=record_cfg,
                record_cmd=msg,
            )
            # 发送响应
            await self.send_response("start_collection", "success")
            # 开始采集倒计时
            logger.info(f"开始采集倒计时{countdown_seconds}s...")
            time.sleep(countdown_seconds)

            # 开始采集
            self.record.start()

        elif data.get("cmd") == "finish_collection":
            logger.info("处理完成采集命令...")
            if self.replaying == True:
                logger.warning("Replay is running, cannot finish collection.")
                await self.send_response("finish_collection", "fail")
                return

            if not self.saveing and self.record.save_data is None:
                # 如果不在保存状态，立即停止记录并保存
                self.saveing = True
                self.record.stop()
                self.record.save()
                self.recording = False
                self.saveing = False

            # 如果正在保存，循环等待直到 self.record.save_data 有数据
            while self.saveing:
                time.sleep(0.1)  # 避免CPU过载，适当延迟
            # 此时无论 saveing 状态如何，self.record.save_data 已有有效数据
            response_data = {
                "msg": "success",
                "data": self.record.save_data,
            }
            # 发送响应
            await self.send_response(
                "finish_collection", response_data["msg"], response_data
            )

        elif data.get("cmd") == "discard_collection":
            # 模拟处理丢弃采集
            logger.info("处理丢弃采集命令...")

            if self.replaying == True:
                logger.warning("Replay is running, cannot discard collection.")
                await self.send_response("discard_collection", "fail")
                return

            self.record.stop()
            self.record.discard()
            self.recording = False

            # 发送响应
            await self.send_response("discard_collection", "success")

        elif data.get("cmd") == "submit_collection":
            # 模拟处理提交采集
            logger.info("处理提交采集命令...")
            time.sleep(0.01)  # 模拟处理时间

            if self.replaying == True:
                logger.warning("Replay is running, cannot submit collection.")
                await self.send_response("submit_collection", "fail")
                return
            # 发送响应
            await self.send_response("submit_collection", "success")

        elif data.get("cmd") == "start_replay":
            logger.info("处理开始回放命令...")
            msg = data.get("msg")
            if self.recording == True:
                logger.warning("Recording is running, cannot start replay.")
                await self.send_response("start_replay", "fail")
                return
            if self.replaying == True:
                logger.warning("Replay is already running.")
                await self.send_response("start_replay", "fail")
                return
            self.replaying = True

            task_id = msg.get("task_id")
            task_name = msg.get("task_name")
            task_data_id = msg.get("task_data_id")
            task_dir = f"{task_name}_{task_id}"
            repo_id = f"{task_name}_{task_id}_{task_data_id}"

            date_str = datetime.now().strftime("%Y%m%d")

            # 构建目标目录路径
            dataset_path = DOROBOT_DATASET
            git_branch_name = get_current_git_branch()
            target_dir = dataset_path / date_str / "user" / task_dir / repo_id
            # if "release" in git_branch_name or "main" in git_branch_name:
            #     target_dir = dataset_path / date_str / "user" / task_dir / repo_id
            # elif "dev" in git_branch_name:
            #     target_dir = dataset_path / date_str / "dev" / task_dir / repo_id
            # else:
            #     target_dir = dataset_path / date_str / "dev" / task_dir / repo_id

            ep_index = find_epindex_from_dataid_json(target_dir, task_data_id)

            dataset = DoRobotDataset(repo_id, root=target_dir)

            logger.info(
                f"开始回放数据集: {repo_id}, 目标目录: {target_dir}, 任务数据ID: {task_data_id}, 回放索引: {ep_index}"
            )

            replay_dataset_cfg = DatasetReplayConfig(
                repo_id, ep_index, target_dir, fps=DEFAULT_FPS
            )
            replay_cfg = ReplayConfig(self.daemon.robot, replay_dataset_cfg)

            # 用于线程间通信的异常队列
            error_queue = queue.Queue()
            # 用于通知replay线程停止的事件
            stop_event = threading.Event()

            def visual_worker():
                """visual工作线程函数"""
                try:
                    # 主线程执行可视化（阻塞直到窗口关闭或超时）
                    visualize_dataset(
                        dataset,
                        mode="local",
                        episode_index=ep_index,
                        web_port=RERUN_WEB_PORT,
                        ws_port=RERUN_WS_PORT,
                        stop_event=stop_event,  # 需要replay函数支持stop_event参数
                        open_browser=False,
                    )
                except Exception as e:
                    error_queue.put(e)

            # 创建并启动replay线程
            visual_thread = threading.Thread(
                target=visual_worker,
                name="VisualThread",
                daemon=True,  # 设置为守护线程，主程序退出时自动终止
            )
            visual_thread.start()

            # 发送响应
            response_data = {
                "data": {
                    "url": f"http://127.0.0.1:{RERUN_WEB_PORT}/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A{RERUN_WS_PORT}%2Fproxy",
                },
            }
            await self.send_response("start_replay", "success", response_data)

            try:
                replay(replay_cfg)

            finally:
                # 无论可视化是否正常结束，都通知replay线程停止
                stop_event.set()
                # 等待replay线程安全退出（设置合理超时）
                visual_thread.join(timeout=5.0)

                # 检查线程是否已退出
                if visual_thread.is_alive():
                    logger.warning("Warning: Visual thread did not exit cleanly")

                # 处理子线程异常
                try:
                    error = error_queue.get_nowait()
                    raise RuntimeError(
                        f"Visual failed in thread: {str(error)}"
                    ) from error
                except queue.Empty:
                    pass
            self.replaying = False

            logger.info("=" * 20 + "Replay Complete Success!" + "=" * 20)

    ####################### Client Send to Server ############################
    async def send_heartbeat_loop(self):
        """定期发送心跳"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
                try:
                    await self.sio.emit("HEARTBEAT")
                    self.last_heartbeat_time = current_time
                except Exception as e:
                    logger.error(f"发送心跳失败: {e}")
            time.sleep(1)
            await self.sio.wait()

    # 发送回复请求
    async def send_response(self, cmd, msg, data=None):
        payload = {"cmd": cmd, "msg": msg}
        if data:
            payload.update(data)
        try:
            async with self.session.post(
                f"{self.server_url}/robot/response",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                logger.info(f"已发送响应 [{cmd}]: {payload}")
        except Exception as e:
            logger.error(f"发送响应失败 [{cmd}]: {e}")

    ####################### Robot API ############################
    def stream_info(self, info: Dict[str, int]):
        self.cameras = info.copy()
        logger.info(f"更新摄像头信息: {self.cameras}")

    def stream_info_add(self, camera_name: str, camera_id: int):
        """添加或更新单个摄像头的流信息
        
        Args:
            camera_name: 摄像头名字
            camera_id: 摄像头编号
        """
        if not hasattr(self, 'cameras'):
            self.cameras = {}
        
        self.cameras[camera_name] = camera_id
        logger.info(f"添加摄像头 {camera_name} 编号: {camera_id}")
        
        # 可选：返回更新后的总流数
        return sum(self.cameras.values())

    async def update_stream_info_to_server(self):
        if not self.server_available:
            logger.debug("Server is unavailable, skip stream info update.")
            return
        stream_info_data = cameras_to_stream_json(self.cameras)
        logger.info(f"stream_info_data: {stream_info_data}")
        try:
            # 2. 异步post加await，确保请求发送
            async with self.session.post(
                f"{self.server_url}/robot/stream_info",
                json=stream_info_data,
                timeout=aiohttp.ClientTimeout(total=2),
            ) as response:
                if response.status == 200:
                    logger.info("摄像头流信息已同步到服务器")
                else:
                    logger.warning(f"同步流信息失败: {response.status}")
        except Exception as e:
            logger.error(f"同步流信息异常: {e}")

    async def update_stream_async(self, name, frame):
        if not self.server_available:
            return
        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        url = f"{self.server_url}/robot/update_stream/{self.cameras[name]}"
        try:
            # 超时给短一点，丢几帧对视频流影响不大
            async with self.session.post(
                url, data=jpeg.tobytes(), timeout=aiohttp.ClientTimeout(total=0.2)
            ) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    logger.error(f"Server error {resp.status}: {txt}")
        except asyncio.TimeoutError:
            logger.warning("update_stream timeout")
        except Exception as e:
            logger.error("update_stream exception:", e)
