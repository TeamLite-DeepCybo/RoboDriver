"""Standalone debug-overlay runner for the RViz pose-accuracy check.

Spins ONLY the adapter's ROS 2 node with ``publish_debug=True`` against the
LIVE tracker topics (``/umi/<arm>/track`` + ``/umi/world_head/pose`` +
``/lite/joint_states``). No dataset is recorded and no mock is started.

The node composes ``T_world_tcp`` exactly as it does while recording and
republishes it on:

  * ``/umi/debug/eef_left`` / ``/umi/debug/eef_right``  (geometry_msgs/PoseStamped, frame=world)
  * ``/umi/debug/markers``                             (visualization_msgs/MarkerArray)

Accuracy criterion (in RViz, Fixed Frame = ``world``): the debug pose axes /
sphere for each arm must sit on top of the tracker's ``gripper_<arm>`` TF
frame. Coincidence = the compose reproduces the tracker's own eef pose.

Run (on the rig, with the real tracker stack already up):

    python -m robodriver_robot_deepcybo_lite_umi_ros2.debug_overlay

Ctrl-C to stop.
"""
from __future__ import annotations

import argparse

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .config import DeepcyboLiteUmiRos2RobotConfig
from .node import DeepcyboLiteUmiRos2RobotNode


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Publish the /umi/debug/eef_* overlay from live tracker "
        "topics for the RViz pose-accuracy check (no recording)."
    )
    parser.add_argument(
        "--control-fps", type=int, default=None,
        help="override compose/publish rate (default: config control_fps)",
    )
    args = parser.parse_args(argv)

    cfg = DeepcyboLiteUmiRos2RobotConfig()
    control_fps = args.control_fps if args.control_fps is not None else cfg.control_fps

    rclpy.init()
    node = DeepcyboLiteUmiRos2RobotNode(
        topics=cfg.ros2_topics,
        control_fps=control_fps,
        camera_fps=cfg.camera_fps,
        publish_debug=True,
    )
    node.get_logger().info(
        f"debug-overlay runner up | consuming tracks=("
        f"{cfg.ros2_topics.track_left}, {cfg.ros2_topics.track_right}) "
        f"world={cfg.ros2_topics.world_head} | publishing "
        "/umi/debug/eef_left|right + /umi/debug/markers (frame=world). "
        "Open RViz (Fixed Frame=world) and compare against gripper_<arm> TF."
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
