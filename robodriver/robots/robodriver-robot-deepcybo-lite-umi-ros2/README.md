# robodriver-robot-deepcybo-lite-umi-ros2

RoboDriver adapter for the DeepCybo Lite **UMI handheld rig**: records LeRobot
datasets live from the ArUco tracking stack, with state = per-arm
end-effector poses in the **world-tag frame**.

Design spec: `docs/superpowers/specs/2026-07-14-umi-eef-adapter-design.md`.

## Data contract

Ingest (live ROS 2 topics, produced by `lite_aruco_umi_ros2`'s
`collection.launch.py` + `lite_umi_ros2` grippers):

| Topic | Type |
|---|---|
| `/umi/left/track`, `/umi/right/track` | `lite_aruco_umi_msgs/GripperTrack` |
| `/umi/world_head/pose` | `geometry_msgs/PoseStamped` |
| `/lite/joint_states` | `sensor_msgs/JointState` |
| `/deepcybo/lite/camera/{head,wrist_left,wrist_right}/image_raw/compressed` | `CompressedImage` |

Recorded features:

- `observation.state` — **16 pose dims** `[L eepose7, L grip, R eepose7,
  R grip]` (quat xyzw, meters, raw SI) **followed by 7 quality dims**
  `[L_tracked, L_present, L_reproj, R_tracked, R_present, R_reproj,
  world_fresh]`.
- `action` — mirror of the 16 pose dims (temporal shift is the training
  dataloader's job).
- `observation.images.{image_head,image_wrist_left,image_wrist_right}`.

> **WARNING — quality dims are for filtering, not for the policy.**
> Training configs must select only the 16 pose/gripper dims as policy
> input and use the 7 quality dims to drop bad frames/episodes. Do not feed
> quality flags into the model.

Dropout semantics: during tracking/world-tag loss the last composed pose is
**held** and flags go to 0 — poses never go NaN, but held stretches must be
filtered or the episode discarded (target: > 90 % tracked coverage).

## Environment

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash        # collection ws: lite_aruco_umi_msgs
# plus the RoboDriver python env (lerobot)
pip install -e robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
```

## Commands

```bash
# synthetic rig (off-robot), optional dropout injection
deepcybo-lite-umi-mock-ros2 --drop-every 90

# end-to-end smoke: one episode against the mock
deepcybo-lite-umi-smoke-record --duration-s 5

# live RViz overlay while recording (config: publish_debug=true):
#   /umi/debug/eef_left|right (Pose), /umi/debug/markers (MarkerArray)
#   pass criterion: axes coincide with TF gripper_<arm> frames

# post-hoc episode replay into RViz
deepcybo-lite-umi-visualize-episode --root /tmp/umi_smoke_drop
```

## Offline smoothing (post-collection)

Recorded episodes contain hold-last dropout frames (tracker + adapter tiers,
flagged by the quality dims). `umi-smooth-episodes` rebuilds every
non-measured frame by lerp+Slerp between `tracked==1` anchors, writing a NEW
dataset with an `observation.provenance` feature (0=measured, 1=interpolated,
2=unfillable) — the raw dataset is never modified:

```bash
umi-smooth-episodes --root <dataset> --out <dataset>_smoothed [--max-gap-s 0.25]
umi-smooth-episodes --root <dataset> --out /dev/null --dry-run   # report only
```

> **WARNING — provenance is for filtering, not for the policy.** Like the 7
> quality dims, `observation.provenance` must be excluded from policy inputs
> in training configs.

Design: `docs/superpowers/specs/2026-07-20-umi-offline-smoother-design.md`.

## send_action

This robot is record-only: `send_action()` raises `NotImplementedError`.
Deployment goes through joint-space replay (offline MoveIt2 IK → the
`deepcybo-lite-aio-ros2` adapter) or a future online IK bridge.
