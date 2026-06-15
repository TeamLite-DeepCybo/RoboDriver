from __future__ import annotations

import argparse
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_LEROBOT_HOME", "/tmp/lerobot_home")
os.environ.setdefault(
    "HF_LEROBOT_CALIBRATION",
    str(Path(os.environ["HF_LEROBOT_HOME"]) / "calibration"),
)

import logging_mp
import rclpy
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from rclpy.executors import MultiThreadedExecutor

from robodriver.dataset.dorobot_dataset import DoRobotDataset
from robodriver.robots.utils import busy_wait
from robodriver_robot_deepcybo_lite_aio_ros2.config import (
    DeepcyboLiteAioRos2RobotConfig,
)
from robodriver_robot_deepcybo_lite_aio_ros2.robot import DeepcyboLiteAioRos2Robot


logger = logging_mp.get_logger(__name__)


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "recordings" / f"deepcybo_lite_external_ros2_{stamp}"


def _build_dataset_features(robot: DeepcyboLiteAioRos2Robot, use_videos: bool) -> dict:
    teleop_action_processor, _robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=use_videos,
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record DeepCybo Lite from external ROS2 topics. This does not publish "
            "mock arm messages; it relies on /slave/lite/joint_states, "
            "/slave/remote_policy_controller/command, and the three configured "
            "Lite camera topics already being available."
        )
    )
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repo-id", default="deepcybo/lite-ros2-external-smoke")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--task",
        default="DeepCybo Lite ROS2 external topic recording.",
    )
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging_mp.basic_config(level=logging_mp.INFO)
    args = build_arg_parser().parse_args()

    os.environ.setdefault("ROS_LOG_DIR", "/tmp/ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_LEROBOT_CALIBRATION"]).mkdir(parents=True, exist_ok=True)

    output_root = args.root or _default_output_root()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset root already exists: {output_root}. "
                "Pass --overwrite or choose a new --root."
            )
        shutil.rmtree(output_root)

    rclpy.init()
    executor = MultiThreadedExecutor()
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    robot: DeepcyboLiteAioRos2Robot | None = None
    dataset: DoRobotDataset | None = None

    try:
        cfg = DeepcyboLiteAioRos2RobotConfig()
        cfg.control_fps = int(args.fps)
        cfg.camera_fps = int(args.fps)
        cfg.use_videos = bool(args.use_videos)
        cfg.calibration_dir = (
            Path(os.environ["HF_LEROBOT_CALIBRATION"]) / "robots" / cfg.type
        )

        robot = DeepcyboLiteAioRos2Robot(cfg)
        executor.add_node(robot.get_node())
        spin_thread.start()

        logger.info("Connecting robot from external ROS2 topics...")
        robot.connect()

        features = _build_dataset_features(robot, args.use_videos)
        logger.info("Dataset features: %s", features)
        dataset = DoRobotDataset.create(
            args.repo_id,
            int(args.fps),
            root=output_root,
            robot=robot,
            features=features,
            use_videos=args.use_videos,
            use_audios=False,
            image_writer_processes=0,
            image_writer_threads=args.image_writer_threads,
        )

        frames_target = max(1, int(round(args.duration_s * args.fps)))
        logger.info(
            "Recording %s frames at %s Hz to %s",
            frames_target,
            args.fps,
            output_root,
        )

        written = 0
        while written < frames_target:
            start_t = time.perf_counter()
            observation = robot.get_observation()
            action = robot.get_action()
            frame = {
                **build_dataset_frame(dataset.features, observation, prefix=OBS_STR),
                **build_dataset_frame(dataset.features, action, prefix=ACTION),
                "task": args.task,
            }
            dataset.add_frame(frame)
            written += 1
            busy_wait(1.0 / args.fps - (time.perf_counter() - start_t))

        episode_index = dataset.save_episode()
        logger.info(
            "Saved episode %s with %s frames at %s",
            episode_index,
            written,
            output_root,
        )
        print(output_root)
    finally:
        if dataset is not None:
            dataset.stop_audio_writer()
        executor.shutdown()
        if spin_thread.is_alive():
            spin_thread.join(timeout=2.0)
        if robot is not None:
            try:
                if robot.is_connected:
                    robot.disconnect()
                else:
                    robot.get_node().destroy()
            except Exception:
                logger.exception("Failed to clean up robot node")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
