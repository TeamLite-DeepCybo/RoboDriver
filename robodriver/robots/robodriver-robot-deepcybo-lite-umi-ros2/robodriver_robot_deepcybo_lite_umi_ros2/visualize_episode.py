"""Replay a recorded UMI episode into RViz (spec §8 acceptance check).

Publishes per-arm nav_msgs/Path (full trajectory in `world`) plus a moving
PoseStamped and quality-colored spheres stepping at --fps. Frames whose
present/world_fresh flags were 0 are drawn dimmed grey — dropout stretches
are visible at a glance.

Usage (RViz: Fixed Frame `world`, add the two Path + Pose + MarkerArray):
    python -m robodriver_robot_deepcybo_lite_umi_ros2.visualize_episode \
        --root /tmp/umi_smoke_drop --repo-id deepcybo/lite-umi-ros2-smoke
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node as ROS2Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .config import EEF_FEATURE_NAMES, QUALITY_FEATURE_NAMES

OBS_STATE = "observation.state"


def _state_index(names: list[str]) -> dict[str, int]:
    """Map feature name -> index into the packed observation.state vector."""
    return {name: i for i, name in enumerate(names)}


def _pose_stamped(node, pos, quat) -> PoseStamped:
    ps = PoseStamped()
    ps.header.stamp = node.get_clock().now().to_msg()
    ps.header.frame_id = "world"
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos)
    (ps.pose.orientation.x, ps.pose.orientation.y,
     ps.pose.orientation.z, ps.pose.orientation.w) = map(float, quat)
    return ps


class EpisodeViewer(ROS2Node):
    def __init__(self) -> None:
        super().__init__("umi_episode_viewer")
        self.pub_path = {
            "left": self.create_publisher(NavPath, "/umi/replay/path_left", 10),
            "right": self.create_publisher(NavPath, "/umi/replay/path_right", 10),
        }
        self.pub_pose = {
            "left": self.create_publisher(PoseStamped, "/umi/replay/eef_left", 10),
            "right": self.create_publisher(PoseStamped, "/umi/replay/eef_right", 10),
        }
        self.pub_markers = self.create_publisher(MarkerArray, "/umi/replay/markers", 10)


def _arm_slices(idx: dict[str, int]):
    def pose7(prefix: str) -> list[int]:
        return [idx[f"{prefix}_eef_{c}.pos"]
                for c in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    return {"left": pose7("left"), "right": pose7("right")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Replay a UMI episode into RViz.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", default="deepcybo/lite-umi-ros2-smoke")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)

    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    names = ds.meta.features[OBS_STATE]["names"]
    idx = _state_index(names)
    arm_idx = _arm_slices(idx)
    q_present = {
        "left": idx["left_present.flag"],
        "right": idx["right_present.flag"],
    }
    q_world = idx["world_fresh.flag"]

    rclpy.init()
    node = EpisodeViewer()

    paths = {a: NavPath() for a in ("left", "right")}
    for p in paths.values():
        p.header.frame_id = "world"

    marker_id = 0
    try:
        for frame_i in range(len(ds)):
            state = np.asarray(ds[frame_i][OBS_STATE], dtype=np.float64)
            markers = MarkerArray()
            for arm in ("left", "right"):
                v = state[arm_idx[arm]]
                pos, quat = v[:3], v[3:]
                ps = _pose_stamped(node, pos, quat)
                node.pub_pose[arm].publish(ps)
                paths[arm].header.stamp = ps.header.stamp
                paths[arm].poses.append(ps)
                node.pub_path[arm].publish(paths[arm])

                good = state[q_present[arm]] >= 1.0 and state[q_world] >= 1.0
                m = Marker()
                m.header = ps.header
                m.ns = f"replay_{arm}"
                m.id = marker_id
                marker_id += 1
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose = ps.pose
                m.scale.x = m.scale.y = m.scale.z = 0.012
                m.color = (
                    ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.9)
                    if good
                    else ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.35)   # dimmed dropout
                )
                markers.markers.append(m)
            node.pub_markers.publish(markers)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(1.0 / args.fps)
        print(f"replayed {len(ds)} frames")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
