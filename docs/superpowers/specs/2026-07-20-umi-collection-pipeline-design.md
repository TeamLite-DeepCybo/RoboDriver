# UMI data collection pipeline — design

Date: 2026-07-20
Rig: DeepCybo Lite bimanual UMI handheld rig (2 ArUco-cube grippers, head cam + 2 wrist cams, world tag)
Recording: `robodriver-robot-deepcybo-lite-umi-ros2` adapter, LeRobot v2.1, 30 Hz
Status: design approved, pending spec review

## Goal

Collect 100–150 demonstrations of a single narrow manipulation task, of sufficient
quality to train a Diffusion Policy, without discovering a rig or schema defect
after the human hours are already spent.

## Task definition

**Pick an object and place it in a container.**

- **Right gripper** — picks: reach, grasp, lift, transport, release.
- **Left gripper** — steadies the container (near-static role).

The bimanual split is deliberate, for two reasons: it makes the recorded 16-dim
two-arm state meaningful rather than half-dead filler, and it assigns the
*weaker-tracking* left arm a *near-static* role. The left arm measured 74.2%
tracked coverage on the 2026-07-15 recording versus the right's 82.1%; a gripper
held nearly still in clear view of the head cam tracks far better than one
performing fast reaches.

Episode length: ~10 s (gate range 5–20 s).

## Pipeline

```
[0] SESSION PRE-FLIGHT      once per sitting, ~3 min
      gripper sweep · 3 camera rates · tracking sanity (RViz overlay) · world tag
                             │  gate: do not record until all green
                             ▼
[1] RECORD ONE EPISODE      RoboDriver record, umi adapter, 30 Hz
                             │
                             ▼
[2] IMMEDIATE QC            ~2 s, at the rig, before the next episode
      coverage · gripper actually moved · 3 streams · length
                             │  FAIL → redo NOW
                             ▼
[3] BATCH PROCESS           end of session
      umi-smooth-episodes → coverage report → per-episode accept/reject
                             │
                             ▼
[4] PILOT GATE              ONCE, after the first 10 episodes
      must reach a real training run before bulk collection is permitted
                             │
                             ▼
[5] BULK COLLECT            100–150 episodes, repeating [0]–[3]
```

### Why QC is at the rig (stage 2)

This is the pipeline's one load-bearing structural decision. An episode found
bad *between episodes* costs ~30 s to redo. The same episode found at the end of
the week is simply lost: the object placement, lighting, and hand motion cannot
be recreated. The 2026-07-15 recording is the cautionary case — the gripper
channel was constant 0.0 for all 240 frames, and a five-second check at the rig
would have caught it.

The check is cheap because the signals already exist: the adapter writes
`tracked` / `present` / `reproj` per frame, and the smoother's coverage
computation runs in milliseconds on a single episode.

### Why the pilot gate is hard (stage 4)

Two links between "recorded data" and "trained policy" are **unverified**, and
neither fails until training is actually attempted:

1. The Linux `DoRobotDataset` canonical-reader gate — the smoother's own
   declared STOP condition, still never run.
2. Whether `lerobot_lite` registers a diffusion policy type at all.

No quantity of collected data works around either. Ten episodes cost about an
hour and settle both.

## Session pre-flight (stage 0)

| Check | Method | Failure it catches |
|---|---|---|
| **Gripper sweep** | fully open→close→open each gripper; watch `/lite/joint_states` | the constant-0.0 stub; also confirms direction and units (encoders are connected but **unverified**) |
| **3 camera rates** | all publishing; wrists at ~30 Hz | the 18–20 Hz wrist USB-bandwidth problem (fix: MJPEG `pixel_format`) |
| **Tracking sanity** | `deepcybo-lite-umi-debug-overlay` + RViz; move each gripper through the work zone | mis-tracking and dead spots, found *before* recording 20 episodes into one |
| **World tag** | visible, static, unoccluded | silent `world_fresh = 0` |

The debug overlay built for the RViz pose-accuracy test is reused here as the
pre-flight instrument; nothing new is needed.

## Per-episode protocol (stage 1)

1. Both grippers at a defined home pose.
2. Left gripper steadies the container.
3. Right gripper: reach → grasp → lift → transport → release into container.
4. Return to home.

Start recording just before motion; stop just after release. Long idle heads and
tails actively teach the policy to remain still.

### Variation

With 100–150 episodes the budget allows **one axis varied well**, not three
varied sparsely:

1. **Object start position — primary.** The difference between a policy and a
   replayed trajectory. Cover it **systematically**: a helper prints a target
   cell ("place at B3") each episode. Unassisted human randomization clumps, and
   the bias only becomes visible after training.
2. **Container position** — fixed, or 2–3 discrete positions. Not continuous.
3. **Object identity** — one object. Object generalization is a separate axis
   needing its own data budget.
4. **Lighting / background** — fixed.

### Workspace layout (occlusion-aware)

**Use a shallow, open container. Not a tall or deep one.** Reaching into a deep
container rotates the cube away from the head cam and hides it behind the rim at
the moment of release — the *task-critical frame*, where grasp state and pose
matter most. A deep container yields a dataset that is clean everywhere except
the one moment the policy most needs.

Additionally: keep the pick zone in the head cam's clear field, and park the left
gripper so its cube faces the head cam.

## Quality gates

### Per-episode, immediate (stage 2)

Run the smoother's coverage computation on the just-written parquet and gate on
**usable-after-smoothing**, not raw tracked count.

| Check | Bar | Rationale |
|---|---|---|
| **Gripper moved** | value range > threshold | catches the 0.0 stub **and** a failed grasp — highest-value single check |
| Picking arm (right) usable | **≥ 95%, zero unfillable** | the right arm already reached 100% usable post-smoothing on real data |
| Steadying arm (left) usable | ≥ 90% | near-static role should comfortably beat its previous 74% |
| 3 camera streams | present, frame counts match | the wrist-bandwidth failure mode |
| Episode length | 5–20 s | runaway or aborted recordings |

Any failure → **redo the episode immediately.**

The gripper-range threshold is calibrated during the pilot: record one
deliberate full open→close, measure the observed range, and set the bar at a
conservative fraction of it. It cannot be set a priori because the encoder units
are unverified.

### Batch, end of session (stage 3)

Run `umi-smooth-episodes`, review the coverage report, reject episodes failing
the per-episode bars. Keep raw and smoothed datasets side by side — the smoother
never modifies its input.

### Pilot gate — all six required before bulk collection (stage 4)

1. **Canonical reader** — smoothed output opens with `DoRobotDataset` and
   `deepcybo-lite-umi-visualize-episode` on the Linux rig.
2. **Diffusion Policy available** — confirmed present in `lerobot_lite`'s
   registered policy types.
3. **A training run starts and loss decreases** over a few hundred steps. Not a
   good policy — proof that shapes match, data loads, and gradients flow.
4. **Gripper signal real** across all 10 episodes, not merely one.
5. **Quality dims and `observation.provenance` excluded** from policy input in
   the training config (both are filtering-only; see the adapter README warning).
6. **Rotation representation decided** — raw quaternion versus 6D rotation.

If any gate fails, fix before collecting further.

## Dataset organization and labels

One dataset per **session**, merged at the end of collection rather than
appending to a single growing dataset — a corrupt or aborted session then costs
one session, not the whole corpus.

```
umi_pickplace_2026-07-2X_sNN/            raw, never modified
umi_pickplace_2026-07-2X_sNN_smoothed/   smoother output
umi_pickplace_merged/                    final training corpus
```

`repo_id`: `deepcybo/umi-pickplace`. Task string, identical for every episode of
this single-task collection, written to `meta/tasks.jsonl` and
`meta/episodes.jsonl`:

> `"Pick up the object and place it in the container."`

This replaces the placeholder text used in the pipeline-test recording
("DeepCybo Lite UMI real-rig gripperless pipeline test"), which is not a task
description and would be meaningless to a language-conditioned policy later.

Record per session, outside the dataset: date, object, container position,
lighting, operator, and any rig change. A dataset whose provenance is unknown
cannot be debugged after the fact.

## Components: existing vs new

**Reuse as-is:** the tracker, the RoboDriver umi adapter, `umi-smooth-episodes`
and its coverage report, `deepcybo-lite-umi-debug-overlay`,
`deepcybo-lite-umi-visualize-episode`.

**New, small:**

- an immediate per-episode QC checker (reads the just-written parquet, applies
  the stage-2 bars, prints PASS/FAIL) — the pipeline's only genuinely new
  critical component;
- a placement-cell prompter for systematic object-position coverage;
- a session pre-flight checker (gripper sweep + camera rates), which may simply
  be a checklist plus a topic-rate script rather than code.

## Rotation representation

Diffusion Policy predicts continuous values and has no notion that a quaternion
must remain unit-norm, so a 6D rotation representation is the usual choice.

Two findings bound this risk:

- The recorded quaternions are **already continuous**: the 2026-07-15 episode has
  **zero hemisphere flips** across all consecutive tracked frames (minimum
  consecutive dot product 0.9992, consistent `qw` sign) on both arms. No
  sign-alignment preprocessing is required. Re-confirm across the pilot's 10
  episodes.
- Converting quaternion → 6D is a **preprocessing pass over existing parquets**,
  not a re-collection. It is cheap to apply retroactively.

Therefore the representation decision is a pilot-time item for tidiness, **not a
collection risk**.

## Known blockers carried in

1. **Gripper encoders connected but unverified** — resolved by pre-flight sweep +
   pilot gate #4. Grasp width is the one channel unrecoverable in post.
2. **Wrist cams ~18–20 Hz vs head ~28 Hz** — USB bandwidth. Fix is MJPEG
   `pixel_format` in usb_cam (config only). Confirm topology with `lsusb -t`;
   the working hypothesis is both wrist cams sharing one host controller.
3. **Left-arm occlusion** — the 2026-07-15 recording contains a 0.70 s (21-frame)
   unfillable dropout at frames 86–106. The head-cam JPEGs for those exact frames
   are in the dataset; inspect them to identify the cause (true occlusion vs
   grazing view angle vs motion blur) before assuming a fix. The steadying-arm
   role plus workspace layout are expected to mitigate it; no filter recovers
   0.7 s of missing information.

## Non-goals

- Online EKF / causal filtering — a deployment-side component; the offline
  smoother already covers training-label repair better (see the
  2026-07-20 offline-smoother spec).
- Multi-task or language-conditioned collection.
- Object or lighting generalization.
- Deployment, IK, and Route A execution.
