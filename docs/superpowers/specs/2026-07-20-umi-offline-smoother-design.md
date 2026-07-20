# UMI offline episode smoother — design

Date: 2026-07-20
Package: `robodriver-robot-deepcybo-lite-umi-ros2` (RoboDriver, branch `feat/umi-eef-adapter`)
Status: design approved, pending spec review

## Problem

Recorded UMI eef datasets contain frames whose pose is a **frozen duplicate** of an
earlier pose rather than a measurement. Two hold-last layers stack:

1. **Tracker** (`aruco_umi.stream_a.StreamAGapTracker`) re-emits the last good
   `T_head_cube` for up to `max_gap_interp_s` (default 0.1 s) and flags the frame
   `tracked=0, present=1`. Despite the parameter name it performs **no**
   interpolation — it is a pure hold.
2. **Adapter** holds its last composed pose once `present=0`, flags going to 0.

Measured on `umi_real_rec_2026-07-15/` (240 frames, left arm):

| tier | frames | meaning |
|---|---|---|
| `tracked=1, present=1` | 178 (74.2%) | genuinely measured |
| `tracked=0, present=1` | 32 | tracker hold — stale duplicate |
| `present=0` | 30 | adapter hold — stale duplicate |

62 of 240 left-arm frames carry a pose that was never measured. A consumer
filtering on `present` silently accepts 32 of them. The result is a *staircase*:
the label says stationary while the gripper kept moving, then jumps on
reacquisition.

A causal filter cannot fix this — it can only extrapolate. Offline we have
measurements on **both sides** of a gap, so we can interpolate with real
information. That is strictly better for training labels and is what this
component does.

Out of scope: the online EKF for Route A deployment (separate, causal, and needs
a `GripperTrack` covariance field); physical occlusion mitigation (cube faces),
which is the actual root cause.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Offline CLI producing a **new** dataset | Non-destructive; raw poses always recoverable; output is a normal LeRobot dataset needing no trainer changes |
| 2 | Anchors = `tracked == 1` **only** | Both hold tiers are frozen duplicates containing no information; re-deriving them discards nothing real |
| 3 | Unfillable frames **kept and marked**, never dropped | Keeps `frame_index` contiguous (no re-indexing/re-chunking); coverage report lets the user decide episode-level |
| 4 | **Fill gaps only** — measured frames bit-exact | Maximum sensor fidelity, no filter tuning, trivially auditable (diff is nonzero only at gap frames) |
| 5 | Lives in the adapter package | Work is ~90% dataset-shaped; that package owns the schema constants and the canonical `DoRobotDataset` read path |

## Architecture

```
umi_real_rec_2026-07-15/            (raw — never modified)
   │  read: parquet via pandas/pyarrow, meta/*.json[l] directly
   ▼
per episode, per arm (left and right fully independent):
   anchors = state[:, <arm>_tracked] == 1
   for each maximal run of non-anchor frames bracketed by anchors a, b:
       if t[b] - t[a] <= max_gap_s:
           pos[k]  = lerp(pos[a], pos[b], w)          w = (t[k]-t[a])/(t[b]-t[a])
           quat[k] = Slerp(t[a], t[b])(t[k])
           provenance[k] = INTERPOLATED
       else:
           leave held pose; provenance[k] = UNFILLABLE
   leading/trailing non-anchor runs (no bracketing anchor) -> UNFILLABLE
   anchor frames -> copied bit-exact; provenance = MEASURED
   ▼
umi_real_rec_2026-07-15_smoothed/   (valid LeRobot v2.1) + coverage report
```

**Left/right independence.** Separate `tracked` flags and genuinely separate
occlusion events (178 vs 197 measured). Coupling them would corrupt a good arm to
match a bad one.

**Gripper columns pass through untouched.** They are not poses; lerp/Slerp do not
apply and interpolating a grasp signal would invent grasp events. (On this
dataset they are the constant-0.0 stub, but the code must not special-case that.)

**Images are never modified.** The output reuses them via hardlink (default) or
copy (`--link-images copy`). Only the parquet and `meta/` are rewritten.

**`action` is regenerated from the smoothed state** to preserve the adapter's
invariant `action == state[:16]` exactly.

**Dataset I/O is direct file manipulation, not `DoRobotDataset`.** Two landmines
found in source during planning: `DoRobotDataset.create()` dereferences
`robot.microphones` with no `None` guard (`dorobot_dataset.py:1373`), and
`add_frame()`'s presence validation would force images through the async image
writer — re-encoding JPEGs the spec requires untouched. Instead the smoother
reads parquet via pandas and writes via pyarrow, mirroring the original file
byte-conventions exactly: `fixed_size_list<float>[N]` arrow columns plus the
`huggingface` schema-metadata JSON (verified present in the recorded parquet),
with `meta/` copied and surgically patched. Consequence: the smoother runs with
only pandas/pyarrow/numpy/scipy (no torch/lerobot env needed), and
**canonical-reader compatibility is enforced by the Task 1 spike** (open the
output with `DoRobotDataset` + `visualize_episode` on Linux) rather than by
using that class at runtime — the same a1da5dc skew risk, now checked
explicitly instead of assumed.

## Rotation interpolation

Unit quaternions live on S³. Linear interpolation cuts a chord through the
sphere's interior: the result is not unit-norm and sweeps at non-constant angular
velocity. Slerp follows the great-circle arc (geodesic) at constant angular rate,
which is the correct "shortest sensible path" for an orientation across a gap.

`scipy.spatial.transform.Slerp` handles the quaternion double cover (`q` and `−q`
are the same rotation) automatically — it interpolates the relative rotation's
rotation vector, whose magnitude is always the short representation. Verified
empirically: sign-flipped anchors 40° apart produce an identical 40° path, not a
320° detour, at a uniform 8.0°/step. This behavior is pinned by a regression test
rather than assumed.

`_align_quats` (hemisphere alignment) is therefore **not** required on this path.
It is only needed for component-wise filtering, which decision 4 excludes.

## Output schema

`observation.state` stays **23 dims**, unchanged — the adapter contract, README,
and any trained config depend on that shape. Provenance is a new, separate
feature:

```
observation.provenance   float32   shape (2,)   names: [left_provenance, right_provenance]
    0.0 = MEASURED       tracked==1, copied bit-exact
    1.0 = INTERPOLATED   synthesized by lerp + Slerp between bracketing anchors
    2.0 = UNFILLABLE     gap too long or unbracketed; held pose retained
```

Additive, so existing readers are unaffected and training configs opt in
explicitly. The original 7 quality dims are preserved **byte-for-byte**: they
record what the tracker saw, provenance records what the smoother did. Keeping
both makes the output fully auditable. Overwriting `tracked`/`present` would make
that history unrecoverable.

> **Schema assumption — partially verified in source, runtime spike still Task 1.**
> The write path is name-agnostic by construction: `create()` merges the passed
> features dict verbatim (`dorobot_dataset.py:413`; the feature sanity check is a
> commented-out TODO), `validate_frame` checks presence/dtype/shape against that
> same dict rather than a hardcoded schema (`utils/dataset.py:688`), and a
> float32 (2,) feature maps to parquet through the identical
> `datasets.Sequence` branch as `observation.state` (`utils/dataset.py:338`).
> What source reading cannot prove is the **runtime round-trip** (stats
> aggregation over the new key, reader behavior, downstream consumers) — the
> a1da5dc v2.1/v3.0 bug lived exactly in that between-components space, hence
> the spike. Fallback if it fails: a sidecar `meta/provenance.jsonl` — same
> information, no schema risk, no redesign.
>
> Two consequences found while verifying:
>
> 1. **Declared ⇒ mandatory.** `validate_frame` checks feature *presence*: once
>    `observation.provenance` is declared, every `add_frame()` must supply it.
>    The smoother always can; the feature just can't be optional.
> 2. **Training configs must exclude it** — same warning class as the 7 quality
>    dims. A non-image `observation.*` key is liable to be auto-classified as a
>    STATE input by LeRobot's feature→policy mapping. The spike must confirm how
>    `lerobot_lite`'s `dataset_to_policy_features` treats the key, and the README
>    warning extends to it: provenance is for filtering, never policy input.

## CLI

```bash
umi-smooth-episodes --root <in> --out <out>
                    [--max-gap-s 0.25] [--link-images hard|copy]
                    [--dry-run] [--overwrite]
```

`--dry-run` computes and prints the report without writing, so `--max-gap-s` can
be tuned against real numbers.

`--max-gap-s` default **0.25 s** (~7 frames @30 Hz). The tracker's causal 0.1 s is
conservative because it extrapolates blindly; bracketed interpolation has
information on both sides and tolerates longer gaps safely. Longest left gap in
the reference episode is ~0.23 s.

**Definition:** `max_gap_s` is measured **anchor to anchor** (`t[b] - t[a]`), not
across the missing frames alone — so a gap of N missing frames spans
`(N+1)/fps` seconds by this measure. This matches the existing
`interpolate_gaps` implementation and is the boundary the tests pin.

Coverage report, per episode per arm:

```
episode_000000
  left    measured 178/240 (74.2%)  interpolated 62  unfillable 0
          gaps: 3x1f, 2x4f, 1x7f    longest 0.233s
  right   measured 197/240 (82.1%)  interpolated 43  unfillable 0
  -> usable 240/240 (100.0%)   KEEP
```

The gap histogram is the actionable part — it shows whether `max_gap_s` is
generous or stingy on this data.

## Error handling

Refuse rather than guess:

- `--out` exists and `--overwrite` not given -> `FileExistsError`.
- Fewer than 2 anchors in an episode -> emit untouched, all frames `UNFILLABLE`,
  log a prominent warning (nothing to bracket).
- Non-monotonic or duplicate timestamps -> abort with the offending index.
- Post-condition assert: an episode must never finish with more unfillable frames
  than it began with held frames. Violation is a bug; fail loudly.

## Testing

Synthetic-first, where ground truth is known:

1. **Round-trip spike** — extra feature survives write -> read (gates the schema).
2. **Known-truth reconstruction** — generate a smooth trajectory, knock out
   frames, verify interpolation recovers them within tolerance. This is the test
   that proves the component works.
3. **Bit-exactness** — measured frames identical to raw (decision 4 guarantee).
4. **Rotation** — quaternions unit-norm after Slerp; sign-flipped anchors take the
   short arc; constant angular rate.
5. **Boundaries** — gap exactly at `max_gap_s`; leading and trailing gaps;
   single-frame gaps.
6. **Pass-through** — gripper columns unmodified; `action == state[:16]` holds.
7. **Independence** — corrupting left leaves right bit-exact.
8. **End-to-end** — on the real 240-frame episode, output re-passes the full
   parquet validation (shapes, no NaN/inf, unit quats, action mirrors state,
   30 Hz, contiguous indices, complete v2.1 meta).

## Non-goals

- Online/causal filtering (EKF) — belongs in the tracker, serves deployment.
- Jitter smoothing of measured frames — deliberately excluded (decision 4); can be
  added later as a separate, measurable step.
- Modifying `GripperTrack.msg` or the tracker.
- Fixing the misleading `max_gap_interp_s` name in the tracker (worth doing;
  separate change).
