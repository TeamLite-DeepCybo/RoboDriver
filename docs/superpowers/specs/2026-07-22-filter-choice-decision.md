# Filter choice: One-Euro over EKF — decision record

Date: 2026-07-22
Status: **provisional** — see "What would make this decisive"
Applies to: `robodriver_robot_deepcybo_lite_umi_ros2.pose_filter`, default in `filter_node.py`

This record exists because the benchmark's working artifacts live in
`.superpowers/`, which is git-ignored. Without this file the merged repository
would contain the filters and the benchmark but no record of which was chosen,
why, or how provisional the choice is.

## Decision

**Ship `OneEuroPoseFilter`.** It is the default in `filter_node.py`.
`EkfPoseFilter` is retained, selectable via `--filter ekf`, and still built and
tested — the margin is not large enough to justify deleting it.

## The evidence, and its limits

### What the benchmark table shows (right arm, real recording)

| filter | jitter (real) | lag (synthetic) | overshoot (synthetic) |
|---|---|---|---|
| raw | 3.15 mm | — | — |
| one-euro mc=1.0 β=1.0 | 0.44 mm | 79 ms | 0 % |
| one-euro mc=0.3 β=0.0 | 0.14 mm | 530 ms | 0 % |
| ekf σa=0.2 | 0.71 mm | ~0 ms | 17.7 % |
| ekf σa=1.0 | 1.48 mm | ~0 ms | 14.5 % |

**Read strictly as the spec's frontier — "at equal lag, which gives less
jitter?" — this table selects the EKF**, which reaches 0.71 mm at ~0 lag while
One-Euro needs 79 ms of lag for 0.44 mm. That is not the decision taken, and the
reason follows.

### Why the table alone is not decisive

Each axis was measured in the regime that flatters a different filter:

- **Jitter** comes from the real recording, which contains only **3.5 cm of
  travel over 8 seconds** — the hand jiggled in place rather than reaching. A
  low-pass loses very little when almost nothing moves, so this flatters
  One-Euro.
- **Lag** comes from a noiseless constant-velocity ramp, which *is* the EKF's
  motion model. Its literal 0.00 ms is its best possible case, not a general
  result.

Lag had to be moved to synthetic motion because it was unobservable on the real
recording: with so little travel, the cross-correlation optimum is flat and
reports ~0 for every configuration, which collapsed the table into "lowest
jitter wins" — precisely the gameable single metric the three-metric design
exists to prevent.

### The measurement that actually decided it

On an **abrupt stop** — the operator halting at a grasp, which is the core
motion of the pick-and-place task this rig is for:

| filter | overshoot past the stop point |
|---|---|
| one-euro | **0.0 mm** |
| ekf σa=0.2 | **16.4 mm**, peaking 4 frames later |
| ekf σa=1.0 | 6.1 mm |

A constant-velocity model keeps extrapolating the velocity it just learned, so
it carries the arm past the operator's hand when they stop. A low-pass cannot
do this. An arm that overshoots 1.6 cm every time you stop is worse for
precision placement than uniform lag, which an operator adapts to within
seconds.

### Evidence that cuts the other way — recorded honestly

On a **min-jerk 30 cm reach** with the rig's real 4 mm noise, the trade inverts:

| config | jitter | peak tracking error | stop overshoot |
|---|---|---|---|
| one-euro mc=1.0 β=0.4 | 0.77 mm | **34.1 mm** | 2.5 mm |
| ekf σa=0.2 | 1.26 mm | **8.5 mm** | 7.1 mm |

Four times less tracking error for 1.6× more jitter. On sustained reaching the
EKF is materially better at following the hand; One-Euro's advantage is
concentrated at the stop.

Note also that One-Euro's good behaviour on fast *reversals* comes from its
velocity estimate lagging — a side effect, not a designed safety property. The
safety test says so in its own comment.

## What would make this decisive

**Record roughly 30 seconds of deliberate reach–grasp–place motion with the rig
and re-run `umi-filter-bench`.** Both axes would then be measured on
representative motion instead of one near-static recording and one synthetic
ramp. This is a single short recording and would convert an inferred choice into
a measured one.

Until then this record should be read as: *One-Euro is the better default for
precision placement; the EKF is better at following sustained motion; the
regimes that decided it were not equally representative.*

Beyond that, the spec already notes that whether a filter **feels** right to
teleoperate cannot be settled offline at all — jitter, lag and overshoot are
proxies, and the real verdict needs a human driving the arm through the IK
bridge.

## Parameters

`OneEuroPoseFilter(min_cutoff=1.0, beta=1.0)` — `beta` was raised from 0.4 to
1.0 because the sweep's own results showed 0.4 was dominated: identical 0.44 mm
jitter and 0 % overshoot at 106 ms lag versus 79 ms.

## Safety parameter, not a tuning parameter

`max_predict_displacement_m` (default **15 mm**) caps how far the filter may
extrapolate during a dropout, and is the reason the frame budget alone is not
sufficient.

Predicting through a gap costs at most `2*v*n/fps` if the operator reverses
direction, which **grows without bound with hand speed**: 2.5 cm at the rig's
median 0.124 m/s, but 20 cm at 1 m/s — the regime the design calls unreachable.
Capping the predicted displacement bounds that error near `2 × cap` regardless
of speed. Measured, EKF, reversal during occlusion:

| hand speed | capped (15 mm) | uncapped |
|---|---|---|
| 0.124 m/s | 2.07 cm | 2.07 cm |
| 0.3 m/s | 1.00 cm | 5.00 cm |
| 1.0 m/s | 0.00 cm | 16.67 cm |

At median speed three predicted frames cover ~12.4 mm, under the cap, so the
anti-stutter benefit for common single-frame dropouts is preserved; at high
speed the cap engages almost immediately and the filter freezes instead.
