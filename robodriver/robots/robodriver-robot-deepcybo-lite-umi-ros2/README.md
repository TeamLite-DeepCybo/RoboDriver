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
umi-smooth-episodes --root <dataset> --dry-run                   # report only
```

> **WARNING — provenance is for filtering, not for the policy.** Like the 7
> quality dims, `observation.provenance` must be excluded from policy inputs
> in training configs.

Design: `docs/superpowers/specs/2026-07-20-umi-offline-smoother-design.md`.

## Teleop pose filter (live)

The tracker publishes RAW poses; this node republishes a smoothed copy for
teleop. Recorded datasets are unaffected.

    umi-filter-bench --root <dataset> --arm right   # compare filters offline
    umi-filter-node --filter one-euro               # live, on the rig

    # tune the parameters the benchmark exists to tune, without a code edit:
    umi-filter-node --filter one-euro --min-cutoff 1.0 --beta 1.0 \
        --max-predict-frames 3 --max-predict-displacement-m 0.015 \
        --silence-timeout-s 0.2

Publishes `/umi/filtered/eef_{left,right}` (PoseStamped) and
`/umi/filtered/stale_{left,right}` (Bool).

> **`stale` means the arm must be halted, and no pose is published for it.**
> While an arm is stale (frozen, or its topic has gone silent), the node
> publishes ONLY the `Bool` on `/umi/filtered/stale_{left,right}` -- never a
> `PoseStamped` under a fresh timestamp. A consumer that never receives a
> pose cannot mistakenly act on a stale one.
>
> The filter predicts through gaps of at most `--max-predict-frames`
> (default: see `DEFAULT_MAX_PREDICT_FRAMES` in `pose_filter.py`) frames, and
> caps the predicted DISPLACEMENT at `--max-predict-displacement-m` (default:
> see `DEFAULT_MAX_PREDICT_DISPLACEMENT_M`) so that predicting is never worse
> than freezing regardless of hand speed -- extrapolating further measured
> ~2x worse than freezing when the operator reversed direction during an
> occlusion, and that ratio only grows with speed if the displacement isn't
> also capped.
>
> Independently of the filter's own gap policy, a wall-clock watchdog
> (`--silence-timeout-s`, default 0.2 s) declares an arm stale if its topic
> goes fully silent (dead tracker, dropped camera, crashed node) -- not just
> when messages keep arriving untracked.

## Collection QC (at-the-rig)

The loop while collecting: record an episode, then immediately run
`umi-qc-episode` against it, and either keep it or redo it while the object
placement, lighting and hand motion still exist to be recreated.

```bash
umi-qc-episode --root <dataset> --picking-arm right --cell B3
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | keep -- every gate (and, if prompted, the operator) passed |
| 1 | redo -- a gate failed or the operator marked the demonstration bad |
| 2 | tool error -- bad `--root`, unreadable dataset, or an interrupted prompt |

Gates (one picking arm that must grasp, one steadying arm that may hold a
container still):

- `gripper_moved` -- the picking arm's gripper actually swept (catches a
  constant-0.0 encoder stub and a demonstration where the grasp never
  happened).
- `picking_usable` / `steadying_usable` -- fraction of frames with a real or
  smoother-reconstructed pose, per arm.
- `picking_unfillable` -- picking-arm frames the smoother could not
  reconstruct at all.
- `picking_raw_tracked` / `steadying_raw_tracked` -- fraction of frames
  *genuinely measured* (not reconstructed), per arm. This floor exists so
  smoothing can never mask a rig that is quietly degrading while `usable`
  stays green.
- `cameras` -- all three camera streams present with one frame per recorded
  step.
- `duration` -- episode length within the expected window.

All thresholds are overridable from the CLI (`--gripper-range-min`,
`--picking-usable-min`, `--picking-max-unfillable`, `--picking-raw-tracked-min`,
`--steadying-usable-min`, `--steadying-raw-tracked-min`, `--duration-min-s`,
`--duration-max-s`) so a pilot session can calibrate bars that have no valid
a-priori value -- notably the gripper-range bar, whose units depend on an
encoder that isn't verified until the pilot runs.

Every run appends a JSON-lines record (timestamp, dataset root, effective
thresholds, operator, note, verdict) to a session log next to the dataset --
`<root>.qc_log.jsonl` by default, overridable with `--session-log`. The
dataset directory itself is never written to.

Use `umi-placement-cells` alongside it to dictate a balanced, shuffled object
placement per episode rather than relying on unassisted human randomization:

```bash
umi-placement-cells --rows 3 --cols 4 -n 30 --seed 0
```

## send_action

This robot is record-only: `send_action()` raises `NotImplementedError`.
Deployment goes through joint-space replay (offline MoveIt2 IK → the
`deepcybo-lite-aio-ros2` adapter) or a future online IK bridge.
