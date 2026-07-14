"""Record a short UMI-rig LeRobot smoke dataset against the mock publisher.

Mirrors the aio adapter's smoke_record.py; asserts the eef feature schema
(16 state + 7 quality + mirrored action) survives the dataset pipeline.
"""
from __future__ import annotations

import argparse
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import logging_mp

os.environ.setdefault("HF_LEROBOT_HOME", "/tmp/lerobot_home")
os.environ.setdefault(
    "HF_LEROBOT_CALIBRATION",
    str(Path(os.environ["HF_LEROBOT_HOME"]) / "calibration"),
)

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

from .config import (
    EEF_FEATURE_NAMES,
    QUALITY_FEATURE_NAMES,
    DeepcyboLiteUmiRos2RobotConfig,
)
from .mock_umi_topics import DeepcyboLiteUmiMockNode
from .robot import DeepcyboLiteUmiRos2Robot

logger = logging_mp.get_logger(__name__)


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("/tmp") / f"deepcybo_lite_umi_smoke_{stamp}"


def _build_dataset_features(robot: DeepcyboLiteUmiRos2Robot, use_videos: bool) -> dict:
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
        description="UMI-rig no-hardware recording smoke test (one episode)."
    )
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--drop-every", type=int, default=0,
                        help="inject LEFT tracking dropouts in the mock")
    parser.add_argument("--repo-id", default="deepcybo/lite-umi-ros2-smoke")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--task", default="DeepCybo Lite UMI rig smoke recording.")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=12)
    parser.add_argument("--connect-timeout-s", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _assert_expected_frame(observation: dict, action: dict) -> None:
    for name in EEF_FEATURE_NAMES:
        assert f"{name}.pos" in observation, f"missing obs {name}.pos"
        assert f"{name}.pos" in action, f"missing action {name}.pos"
        assert observation[f"{name}.pos"] == action[f"{name}.pos"], (
            f"action does not mirror observation for {name}"
        )
    for name in QUALITY_FEATURE_NAMES:
        assert f"{name}.flag" in observation, f"missing quality {name}.flag"
    for cam in ("image_head", "image_wrist_left", "image_wrist_right"):
        assert cam in observation, f"missing camera {cam}"
        assert observation[cam].shape == (480, 640, 3)


def run_smoke_record(args: argparse.Namespace) -> Path:
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
    robot: DeepcyboLiteUmiRos2Robot | None = None
    mock_node: DeepcyboLiteUmiMockNode | None = None
    dataset: DoRobotDataset | None = None

    try:
        cfg = DeepcyboLiteUmiRos2RobotConfig()
        cfg.control_fps = int(args.fps)
        cfg.camera_fps = int(args.fps)
        cfg.use_videos = bool(args.use_videos)
        cfg.calibration_dir = (
            Path(os.environ["HF_LEROBOT_CALIBRATION"]) / "robots" / cfg.type
        )

        mock_node = DeepcyboLiteUmiMockNode(
            topics=cfg.ros2_topics,
            fps=float(args.fps),
            drop_every=args.drop_every,
        )
        robot = DeepcyboLiteUmiRos2Robot(cfg)

        executor.add_node(mock_node)
        executor.add_node(robot.get_node())
        spin_thread.start()

        connect_deadline = time.perf_counter() + args.connect_timeout_s
        while True:
            try:
                robot.connect()
                break
            except TimeoutError:
                raise
            except Exception:
                if time.perf_counter() >= connect_deadline:
                    raise
                time.sleep(0.1)

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
        logger.info("Recording %s frames at %s Hz to %s",
                    frames_target, args.fps, output_root)

        written = 0
        while written < frames_target:
            start_t = time.perf_counter()
            observation = robot.get_observation()
            action = robot.get_action()
            if written == 0:
                _assert_expected_frame(observation, action)
            frame = {
                **build_dataset_frame(dataset.features, observation, prefix=OBS_STR),
                **build_dataset_frame(dataset.features, action, prefix=ACTION),
                "task": args.task,
            }
            dataset.add_frame(frame)
            written += 1
            busy_wait(1.0 / args.fps - (time.perf_counter() - start_t))

        episode_index = dataset.save_episode()
        logger.info("Saved episode %s with %s frames at %s",
                    episode_index, written, output_root)
        return output_root
    finally:
        if dataset is not None:
            dataset.stop_audio_writer()
        if executor is not None:
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
        if mock_node is not None:
            try:
                mock_node.destroy_node()
            except Exception:
                logger.exception("Failed to clean up mock node")
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: Iterable[str] | None = None) -> None:
    logging_mp.basic_config(level=logging_mp.INFO)
    args = build_arg_parser().parse_args(argv)
    output_root = run_smoke_record(args)
    print(output_root)


if __name__ == "__main__":
    main()
