# Online pose filter for teleop — design

Date: 2026-07-22
Package: `robodriver-robot-deepcybo-lite-umi-ros2` (RoboDriver, branch `feat/umi-eef-adapter`)
Status: design approved, pending spec review

## Purpose

Smooth the **live** ArUco gripper pose stream so it can drive a robot arm during
teleoperation. The UMI handheld rig is to be used as a teleop input device (a
capability separate from handheld data collection), which puts the ArUco pose
inside a real-time control loop for the first time.

## Why a filter is needed — and what for

Measured on the real recording `umi_real_rec_2026-07-15`, high-frequency jitter
(deviation from a 5-frame centred moving average, over consecutively-tracked
runs):

| arm | median | p95 | max |
|---|---|---|---|
| left | 2.67 mm | 7.25 mm | 11.00 mm |
| right | 3.15 mm | 9.23 mm | 13.09 mm |

At 30 Hz, commanded straight to an arm, that is visible high-frequency shake.
**Jitter rejection is the filter's job.**

**It is explicitly *not* for predicting through dropouts.** A simulated
adversarial case — move forward, occlude the markers, reverse direction —
produced **17.8 cm of error, roughly 2× worse than simply freezing the last
pose**, with the error running 4–5σ outside the filter's own covariance, and the
innovation gate then rejecting the correct measurement on reacquisition and never
recovering. During recording that corrupts a label; during teleop that is a real
arm lurching 17.8 cm the wrong way. The gap policy below is built to make that
regime unreachable.

## Scope

Three consumers of gripper pose, with different needs. Only the middle one gets
this filter:

| consumer | pose source | filtering |
|---|---|---|
| recording (training data) | ArUco, raw | none online — the offline smoother is strictly better (uses both sides of a gap) |
| **teleop** | **ArUco, live** | **this filter** |
| policy deployment | robot FK | none — no ArUco in the loop |

**The tracker keeps publishing raw measurements with honest flags.** This filter
lives downstream, so recorded datasets stay pristine and the offline smoother's
anchor set stays honest. Two consequences: the raw-vs-filtered recording question
disappears, and `GripperTrack.msg` needs **no** covariance field, because the
filter's uncertainty is consumed locally rather than published.

Out of scope: the online IK bridge, workspace mapping/clutching,
`world_root ← world_tag` calibration, and the teleop node itself. This spec
delivers the filter and a node that republishes filtered poses; nothing consumes
that topic until the IK bridge exists.

## Architecture

```
pose_filter.py      PURE MATH — stdlib + numpy + scipy only. No ROS, no I/O.
  PoseFilter          protocol: update(t, pos, quat, tracked) -> FilterOutput
  OneEuroPoseFilter   ~40 lines; params (min_cutoff, beta)
  EkfPoseFilter       ~110 lines; constant-velocity, error-state
                      -> both share ONE implementation of the gap policy

filter_bench.py     OFFLINE HARNESS — replays a recorded dataset through any
                    filter; reports jitter / lag / overshoot. Decides which
                    filter is kept.

filter_node.py      ROS NODE — subscribes to live tracker topics, republishes
                    filtered poses plus a stale flag. Thin by construction.
```

The pure/ROS split is what makes the filter provable: it is validated
exhaustively against recorded data with zero hardware, leaving the node as a
shell that only subscribes and publishes. Same rationale that keeps
`smoothing.py` importable without ROS.

### Interface

```python
@dataclass(frozen=True)
class FilterOutput:
    pos: np.ndarray | None   # (3,); None only before the first measurement
    quat: np.ndarray | None  # (4,) xyzw, unit norm; None likewise
    stale: bool              # True -> teleop must halt the arm
    n_predicted: int         # consecutive frames without a measurement
```

`stale` is driven by the tracker's `tracked` flag plus the predicted-frame
counter — **not** by a covariance. This is why the EKF's covariance is not
load-bearing here, and why One-Euro is a fair competitor despite not producing
one.

### Gap policy (shared by both filters)

```
tracked          -> update from measurement, n_predicted = 0
gap ≤ 3 frames   -> predict forward           (stale = False)
gap > 3 frames   -> FREEZE last output        (stale = True)
```

**Why 3 frames (0.1 s):** single-frame dropouts are common — 8 on the left arm
and 14 on the right in one 240-frame episode — and hard-freezing each would make
the arm visibly stutter dozens of times per episode. Over 1–3 frames at measured
hand speeds a constant-velocity guess is off by only a few mm. Beyond that the
error grows into the regime that produced the 17.8 cm result, so the filter stops
rather than guessing.

Frozen output is **bit-identical** for the duration of the freeze — no drift, so
the arm holds still rather than creeping.

`max_predict_frames` is a **constructor parameter defaulting to 3**, not a
constant. The 3-frame figure is calibrated to this rig's measured dropout
distribution and hand speeds; a different camera rate or a jumpier operator moves
it, and the benchmark can sweep it like any other parameter.

### Initialization and startup

Before any measurement arrives the filter has no state and **must not invent
one**. Behaviour:

- **First tracked measurement** — adopt it verbatim as the initial state
  (position, orientation), with zero velocity. Output equals input, `stale=False`.
  No warm-up ramp: a filter that eases toward its first measurement would command
  the arm to drift from an arbitrary origin.
- **Before any tracked measurement** — output `stale=True` and no pose. The
  teleop layer must not command the arm at all in this state. Represented as
  `FilterOutput(pos=None, quat=None, stale=True, n_predicted=0)`, so a consumer
  that ignores `stale` fails loudly rather than driving to a default pose.
- **A `reset()` method** returns the filter to the uninitialized state, for
  re-arming after a long freeze or an operator clutch event.

### Orientation

Filtered on the rotation manifold, for both filters:

```python
d = R_meas * R_filt_prev.inv()      # delta rotation
v = d.as_rotvec()                   # flat 3-vector — safe to filter
v_f = filter(v)
R_filt = Rotation.from_rotvec(v_f) * R_filt_prev
```

On-manifold by construction: no renormalisation, no hemisphere handling, and the
double-cover ambiguity cannot arise. Preferred over component-wise quaternion
blending, which is approximate and degrades as rotation speed rises.

Orientation is filtered because it drives the wrist joints — the fastest-moving
ones — so wrist wobble is as visible in teleop as position shake.

## Benchmark

The deliverable that decides which filter is kept.

| metric | computation | meaning |
|---|---|---|
| **jitter** | deviation from a 5-frame centred moving average over consecutively-tracked runs (the measure that gave the 2.67 / 3.15 mm baseline) | how much the arm stops shaking |
| **lag** | cross-correlation offset between filtered and raw position minimising error, in ms | how much delay was added |
| **overshoot** | max filtered excursion past a raw step, as % of step size | whether it rings after fast motion |

**All three are required.** Any filter can drive jitter to zero by smoothing
harder — it just becomes unusable, lagging behind the operator's hand. The design
question is the *trade*, so jitter alone would let a bad filter look excellent.

**The comparison is a parameter sweep, not a single run.** Each filter is swept
across its tuning parameters and the results reported as a **jitter-vs-lag
frontier**, so the question becomes "at equal lag, which gives less jitter?"
Comparing one arbitrary setting of each would prove nothing.

### Ground truth

There is no independent measurement of true gripper pose, so accuracy is not
directly measurable. Two ways around it:

- Jitter, lag and overshoot need no ground truth — they are properties of the
  signal, and they are what teleop feel depends on.
- For accuracy, reuse the leave-one-out method from the smoother review: drop a
  measured frame, filter across it, compare output against the held-out
  measurement. A real error number against real data.

## Testing

Synthetic-first, where truth is known.

**Filter maths:**
- zero-noise constant velocity → tracked exactly, zero lag (catches sign errors
  and broken state updates that jitter metrics would mask)
- step response: no NaN, no oscillation, converges
- known Gaussian noise on a smooth path → output variance drops materially
- quaternions unit-norm to 1e-9 over a long run; a 40° rotation converges to 40°,
  not 320°
- rotation-only motion does not perturb position, and vice versa

**Gap policy** — identical tests for both filters:
- 1, 2, 3-frame gaps → predicted, `stale=False`
- **4-frame gap → frozen, `stale=True`** (the boundary)
- 20-frame gap → frozen, output bit-identical throughout (no drift)
- **the reversal scenario** — forward, occlude, reverse — asserting bounded error
  and specifically that the 17.8 cm regime is unreachable. This test encodes why
  the policy exists.
- recovery after a long freeze converges, without the gate lock-out the EKF
  showed in simulation
- `max_predict_frames` is honoured when set to values other than 3

**Initialization:**
- before any tracked measurement → `pos`/`quat` are `None` and `stale=True`
- first tracked measurement is adopted verbatim (output equals input exactly, no
  warm-up ramp)
- an episode whose first N frames are all untracked stays uninitialized, then
  initializes cleanly on the first real measurement
- `reset()` returns the filter to the uninitialized state

**Determinism:** identical input twice → identical output. Filters carry state;
a stale-state bug is otherwise invisible.

**Against the real recording:** both filters over `umi_real_rec_2026-07-15` —
jitter below the 2.67 / 3.15 mm baseline, no NaN across 240 frames including the
real dropouts, and the 21-frame wire gap triggers freeze.

**The node** gets construction/subscribe/republish tests only; all logic lives in
the pure layer. No ROS integration test — it would need a live rig and would
cover nothing the pure tests do not.

## Known limitation

**Whether the filter feels right to teleoperate cannot be tested here.** Jitter,
lag and overshoot are proxies. The real verdict needs a human driving the arm,
which needs the IK bridge. The benchmark narrows the choice to a defensible
setting; it does not replace that trial.

## Non-goals

- Predicting through long dropouts (evidence above).
- Filtering in the tracker or in the recording path.
- Publishing covariance / changing `GripperTrack.msg`.
- IK, workspace mapping, clutching, calibration, the teleop node.
