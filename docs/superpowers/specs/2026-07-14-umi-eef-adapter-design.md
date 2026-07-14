# UMI EEF-Pose RoboDriver Adapter — Design

**Date:** 2026-07-14
**Target branch:** `feat/umi-eef-adapter` (off `feat-deepcybo_lite_unified_cmd`)
**New package:** `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/`

## 1. Goal and context

Record LeRobot datasets **live** from the DeepCybo Lite UMI handheld rig
(two ArUco-tracked handheld grippers + head/wrist cameras), with the policy
state expressed as **end-effector poses in the workspace (world-tag) frame**
instead of arm joint angles.

The existing `robodriver-robot-deepcybo-lite-aio-ros2` adapter is
joint-space (16-dim leader/follower joint vectors, `MITCommand` replay) and
targets the real two-arm robot. The UMI rig has no arm joints — only
vision-tracked gripper poses — so it gets its own sibling adapter, following
the repo convention of one package per robot variant.

Upstream data producers (already built, in `lite_aruco_umi_ros2`):

- Tracker node: per-arm `lite_aruco_umi_msgs/GripperTrack` on
  `/umi/left/track`, `/umi/right/track` — cube and TCP pose **in the head
  frame**, plus quality fields (`tracked`, `present`, `n_markers`, `reproj`,
  `has_tcp`). Gap-hold bridges tracking dropouts ≤ 0.1 s.
- World-tag node: `/umi/world_head/pose` (`PoseStamped`) — `T_world_head`
  from a fixed workspace AprilTag (world := tag frame).
- Gripper motors: `/lite/joint_states` (`JointState`) — `left_gripper` /
  `right_gripper` normalized opening in [0, 1], 50 Hz.
- Cameras: `CompressedImage` on
  `/deepcybo/lite/camera/{head,wrist_left,wrist_right}/image_raw/compressed`,
  30 Hz.

Both `GripperTrack` and the world-head pose are derived from the **same head
camera image**, so their `header.stamp` values match exactly per frame.

## 2. Non-goals

- **No replay / deployment.** `send_action()` is a guarded stub. Deploying
  eef-space actions on the real arm (online IK / MoveIt Servo) is future
  work; the joint-space route (offline MoveIt2 IK → aio adapter) is owned by
  a teammate.
- **No changes to the aio adapter** or to `lite_aruco_umi_ros2`.
- **No offline bag conversion.** The adapter ingests live topics; rosbags
  remain a raw backup, out of scope here.
- **No robot-base calibration.** Poses are stored in the world-tag frame;
  any `world_root ← world_tag` transform is a deploy-time concern.

## 3. Package layout

Mirror the aio package exactly:

```
robodriver-robot-deepcybo-lite-umi-ros2/
  pyproject.toml
  README.md
  robodriver_robot_deepcybo_lite_umi_ros2/
    __init__.py
    config.py        # topics, state layout, registration
    node.py          # rclpy node: subscribe, cache, compose
    robot.py         # LeRobot Robot subclass
    status.py        # status reporting (mirrors aio)
    se3.py           # minimal pose<->matrix helpers (numpy, no ROS)
  scripts/
    ros2_mock_umi_topics.py   # synthetic publishers for off-rig testing
    smoke_record.py           # end-to-end record smoke test
    visualize_episode.py      # post-hoc RViz episode viewer
  tests/
    test_compose.py           # pure-python unit tests (no ROS)
```

Registered as `@RobotConfig.register_subclass("deepcybo-lite-umi-ros2")`.

## 4. Ingest contract (`config.py` topics dataclass)

| Field | Default topic | Type |
|---|---|---|
| `track_left` | `/umi/left/track` | `lite_aruco_umi_msgs/GripperTrack` |
| `track_right` | `/umi/right/track` | `lite_aruco_umi_msgs/GripperTrack` |
| `world_head` | `/umi/world_head/pose` | `geometry_msgs/PoseStamped` |
| `joint_states` | `/lite/joint_states` | `sensor_msgs/JointState` |
| `camera_head` | `/deepcybo/lite/camera/head/image_raw/compressed` | `CompressedImage` |
| `camera_wrist_left` | `/deepcybo/lite/camera/wrist_left/image_raw/compressed` | `CompressedImage` |
| `camera_wrist_right` | `/deepcybo/lite/camera/wrist_right/image_raw/compressed` | `CompressedImage` |

All overridable at instantiation, matching the aio pattern.
`control_fps = camera_fps = 30`.

## 5. State schema

**Core state — 16 named float features** (galaxea-eepose ordering):

```
left_eef_x, left_eef_y, left_eef_z,
left_eef_qx, left_eef_qy, left_eef_qz, left_eef_qw,
left_gripper,
right_eef_x, right_eef_y, right_eef_z,
right_eef_qx, right_eef_qy, right_eef_qz, right_eef_qw,
right_gripper
```

- eef pose = the **gripper tip (tip_middle-aligned TCP) frame in the
  world-tag frame**, meters, quaternion `(x, y, z, w)`.
- gripper = normalized opening [0, 1] from `/lite/joint_states`.
- Raw SI values; **no normalization** applied at record time.

**Quality features — 7 named floats** appended to the observation:

```
left_tracked, left_present, left_reproj,
right_tracked, right_present, right_reproj,
world_fresh
```

Booleans stored as 0.0/1.0; `reproj` in pixels; `world_fresh` = 1.0 when a
same-stamp world pose was found for the frame's composition, else 0.0.

**Action = exact mirror of the 16 core state floats.** The handheld demo has
no leader/follower split; temporal shifting (predict pose at `t+k`) is the
training dataloader's job, keeping the stored data raw and the horizon
choice reversible.

**Cameras:** 3 image features (head, wrist_left, wrist_right), decoded to
`(H, W, 3)` uint8, as in the aio adapter.

## 6. Composition (`node.py`)

1. Cache the last N (≈30) `world_head` poses in a stamp-keyed buffer.
2. On each `GripperTrack` message for arm `X`:
   - Require `has_tcp` and `present`; take `tcp_pose` (head frame).
   - Look up the world pose with the **exact same stamp**; fall back to the
     nearest within **5 ms**; if none, mark `world_fresh = 0` and skip
     composition (hold last).
   - Compose `T_world_tcp = T_world_head · T_head_tcp` (4×4 numpy via the
     package-local `se3.py`).
   - Update the per-arm cache: pose, quaternion, quality fields, stamp.
3. On each `JointState`: parse `left_gripper` / `right_gripper` positions.
4. On each camera message: decode to BGR→RGB uint8 and cache (aio pattern).

The composition and stamp-pairing logic live in **pure functions** (numpy
in/out, no rclpy) so `tests/test_compose.py` exercises them without ROS.

## 7. Lifecycle and error handling (`robot.py`)

- **`connect()`** — gated wait, 20 s timeout, aio-style: requires (a) all
  three cameras seen, (b) at least one successfully composed world-frame
  pose per arm, (c) gripper joints seen. Timeout error lists exactly which
  conditions are unmet.
- **Per-frame after connect — never raise.** `get_observation()` always
  returns a full frame. During dropouts beyond the tracker's gap-hold, the
  last composed pose is held and quality flags are zeroed
  (`present = 0`, and/or `world_fresh = 0`). Training filters on the flags
  (INTERFACE.md §7 coverage budget: episodes should be > 90 % tracked).
- **`get_action()`** — returns the mirror of the current core state.
- **`send_action()`** — raises `NotImplementedError("UMI rig is passive;
  deploy via joint-space replay (Route B) or a future IK bridge")`.
- **`disconnect()`** — destroys the rclpy node (aio pattern).

## 8. Testing and verification

**Unit (pure python, no ROS)** — `tests/test_compose.py`:
- `T_world_head · T_head_tcp` against hand-computed values (incl. identity,
  pure translation, 90° rotations).
- Stamp pairing: exact hit, nearest-within-5 ms, miss → `world_fresh = 0`.
- Hold-last + quality-flag transitions across a simulated dropout.
- State-vector ordering and action mirroring.

**Mock integration (ROS, off-rig)** — `scripts/ros2_mock_umi_topics.py`
publishes synthetic `GripperTrack` (a known circular trajectory), matching
world poses, joint states, and test-pattern images. `scripts/smoke_record.py`
runs a short record session against the mocks and asserts a valid LeRobot
episode with the expected feature set is produced.

**RViz live overlay** — node parameter `publish_debug` (default **off**):
- Publishes `/umi/debug/eef_left|right` (`PoseStamped`, frame `world`) — the
  exact composed pose about to be recorded — plus a `MarkerArray` color-coded
  by quality (green = tracked, yellow = gap-held, red = stale/world-lost).
- **Pass criterion:** with the full tracker stack up, the debug axes
  coincide with TF2's own composed `gripper_<arm>` frames (the tracker
  publishes `world → head → cube_<arm> → gripper_<arm>`); since the adapter
  composes the same pose independently in numpy, any visible offset is a
  composition bug. Occluding a cube or the world tag must flip the marker
  colors accordingly.

**RViz post-hoc episode viewer** — `scripts/visualize_episode.py`: loads a
recorded LeRobot episode and replays it into RViz — `nav_msgs/Path` per arm
in `world`, a moving pose marker stepping at 30 Hz, gripper opening mapped to
marker scale, quality-flagged frames dimmed. Acceptance check per session:
the rendered trajectory matches the motion actually performed — smooth,
in-bounds, no teleports at dropout boundaries.

## 9. Environment prerequisites

- RoboDriver host runs on the collection machine with the ROS 2 graph up
  (`collection.launch.py` from `lite_aruco_umi_ros2`).
- The shell sourcing RoboDriver must also source the collection workspace so
  `lite_aruco_umi_msgs` (GripperTrack) is importable.
- `lerobot` + this repo's Python env per the existing aio README.

## 10. Risks / open items

- **Single-tag world pose quality:** `T_world_head` comes from one coplanar
  AprilTag (IPPE_SQUARE PnP) and is subject to planar-flip jitter near
  fronto-parallel views. It enters every composed pose. The quality flags +
  RViz overlay make it observable; if it proves too noisy, upgrading Stream C
  (multi-tag or board) is upstream work in `lite_aruco_umi_ros2`, not here.
- **`GripperTrack` availability:** the collaborator's msgs package is still
  uncommitted in their workspace snapshot; this adapter depends on it being
  merged/available.
- **Feature naming vs LeRobot tooling:** the 16+7 named float features must
  round-trip through LeRobot's dataset schema like the aio adapter's motor
  features do; verified by `smoke_record.py`.
- **Quality dims are for filtering, not for the policy.** Training configs
  must select only the 16 pose/gripper dims as policy input and exclude the
  7 quality floats, which exist to filter bad frames/episodes at load time.
  This is a training-side (dataloader/config) responsibility, out of the
  adapter's scope — documented in the package README so dataset consumers
  don't feed quality flags into the model by accident.
