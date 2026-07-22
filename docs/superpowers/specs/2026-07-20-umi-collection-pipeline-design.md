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
| **Wire routing** | cables physically clipped/routed out of the head cam's line of sight to both cubes | **the confirmed root cause of the 0.70 s dropout** in the 2026-07-15 recording — wires blocked the only visible marker face |
| **Gripper sweep** | fully open→close→open each gripper; watch `/lite/joint_states` | the constant-0.0 stub; also confirms direction and units (encoders are connected but **unverified**) |
| **3 camera rates** | all publishing; wrists at ~30 Hz | the 18–20 Hz wrist USB-bandwidth problem (fix: MJPEG `pixel_format`) |
| **Face redundancy** | sweep each gripper through the working volume; confirm **≥ 2 marker faces detected** throughout, not merely that tracking succeeds | single-face states, where one obstruction ends tracking |
| **Tracking sanity** | `deepcybo-lite-umi-debug-overlay` + RViz | mis-tracking and dead spots, found *before* recording 20 episodes into one |
| **World tag** | visible, static, unoccluded | silent `world_fresh = 0` |

The debug overlay built for the RViz pose-accuracy test is reused here as the
pre-flight instrument; nothing new is needed.

**Wire routing must be structural, not remembered.** The operator is
concentrating on the manipulation, not the cabling, and "be careful" will not
survive 150 episodes. Clip or route the cables so they physically cannot enter
the sight line, then verify it as a pre-flight item.

**Face redundancy is the deeper fix.** The wires were the trigger; the
underlying fragility was that the cube had exactly one face visible at that
moment, so a single obstruction ended tracking for 21 frames. With the wires
fixed, the next obstruction — forearm, object, container rim — reproduces the
same hole. Checking for ≥ 2 detected faces through the working volume is
therefore a stronger pre-flight condition than "tracking works."

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
| Picking arm **raw tracked** | **≥ 90%** | see "raw floor" below — smoothing must not become a crutch |
| Steadying arm (left) usable | ≥ 90% | near-static role should comfortably beat its previous 74% |
| 3 camera streams | present, frame counts match | the wrist-bandwidth failure mode |
| Episode length | 5–20 s | runaway or aborted recordings |
| **Manual review flag** | operator marks the episode good/bad | UMI's `check_result.txt` equivalent — catches task-level failures (dropped object, botched grasp) that no metric sees |

Any failure → **redo the episode immediately.**

**The raw-tracked floor exists so smoothing cannot mask a degrading rig.**
Gating only on usable-after-smoothing would let raw coverage rot from 90% to 60%
while the gate stayed green, because the smoother keeps recovering frames. The
raw floor makes tracking degradation visible on the episode it starts, which is
the point of at-the-rig QC. For reference, the 2026-07-15 recording had 82.1%
raw on the picking arm — below this bar, and its worst dropout was a routing
problem now fixed.

**Manual review is not optional.** Every metric here is about *tracking* quality;
none can tell whether the demonstration itself was any good. UMI drops episodes
whose `check_result.txt != true`, and a policy trained on fumbled grasps learns
to fumble.

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
  the stage-2 bars including the raw-tracked floor, prompts for the manual
  review flag, prints PASS/FAIL) — the pipeline's only genuinely new critical
  component;
- a placement-cell prompter for systematic object-position coverage;
- a session pre-flight checker: camera rates and gripper sweep in script form;
  wire routing and face redundancy as operator checks, with the face-redundancy
  sweep read off the existing debug overlay;
- a one-off latency measurement script (`GripperTrack` header stamp vs
  wall-clock publish time) — run once during the pilot, not per session.

The manual review flag needs somewhere to live. Store it per episode in the
session log alongside the placement cell, not inside the dataset — it is
collection metadata, and the dataset schema is already settled.

## Rotation representation

Diffusion Policy predicts continuous values and has no notion that a quaternion
must remain unit-norm, so a 6D rotation representation is the usual choice —
and it is what UMI trains on (see the reference table above).

One finding bounds the risk further: the recorded quaternions are **already
continuous**. The 2026-07-15 episode has **zero hemisphere flips** across all
consecutive tracked frames (minimum consecutive dot product 0.9992, consistent
`qw` sign) on both arms, so no sign-alignment preprocessing is required.
Re-confirm across the pilot's 10 episodes.

Since both this and the absolute→relative question are preprocessing passes over
recorded parquets, neither is a **collection** risk. Both are decided before the
first real training run.

## Reference: what the original UMI does

Read from `real-stanford/universal_manipulation_interface` (2026-07-20). Useful
as a check on our decisions, not as a template — our tracking architecture
differs on purpose.

| Aspect | UMI | Us |
|---|---|---|
| Tracking | inside-out visual-**inertial** SLAM, localizing against a **prebuilt map** | outside-in ArUco from a fixed head cam, no IMU |
| Dropout handling | **reject** the episode (> 10 lost frames); no filtering, no interpolation | smooth offline, reject only what is unrecoverable |
| Lost-frame value | identity-quaternion sentinel — deliberately invalid | hold-last, flagged |
| Data retained | **99%** | 74–82% raw tracked |
| Deployment | chunk-wise, 10 Hz, **6 steps ≈ 0.6 s per inference**, no temporal ensembling | undecided (chunk-wise is now the strong default) |
| Rotation | **6D** | quaternion |
| Pose frame | **relative to the current gripper pose** | absolute, world-tag frame |
| Latency | measured and explicitly compensated | **never measured** |

Three consequences for this pipeline:

1. **UMI's robustness techniques do not transfer.** Prebuilt maps,
   relocalization, IMU dead-reckoning, and SLAM masking are all inside-out
   methods. Our coverage problem is ours alone to solve — via wire routing, face
   redundancy, and camera placement. This is why the pre-flight now checks both.
2. **Rejecting rather than salvaging is not available to us at their threshold.**
   UMI can discard episodes with > 10 lost frames because they achieve 99%. At
   74–82% that rule would discard nearly everything, so the offline smoother is
   the correct adaptation to a weaker tracking architecture — not a workaround.
   The raw-tracked floor above is what keeps that honest.
3. **Their quality-gate posture is worth copying even where their tracking is
   not** — hence the manual-review flag and the reported statistics.

### Latency (new, previously unconsidered)

UMI treats latency as first-class: collection latency and deployment latency
must match, or the policy encounters a different world than it trained on. Our
camera → ArUco detect → compose → publish path has a real, **unmeasured** delay.

Action: during the pilot, measure it — compare the `GripperTrack` header stamp
(which carries the image capture time) against wall-clock publish time. Record
the figure in the session log. It costs an afternoon and is needed for
deployment regardless.

### Pose representation (decide before training, not before collecting)

UMI trains on **6D rotations, relative to the current gripper pose**. We record
absolute quaternions. Both differences are **preprocessing passes over recorded
parquets** — neither requires re-collection — but they change what the policy
learns and should be settled before the first real training run. Raised with the
mentor; not a collection blocker.

## Known blockers carried in

1. **Gripper encoders connected but unverified** — resolved by pre-flight sweep +
   pilot gate #4. Grasp width is the one channel unrecoverable in post.
2. **Wrist cams ~18–20 Hz vs head ~28 Hz** — USB bandwidth. Fix is MJPEG
   `pixel_format` in usb_cam (config only). Confirm topology with `lsusb -t`;
   the working hypothesis is both wrist cams sharing one host controller.
3. **Left-arm occlusion — root cause identified.** The 2026-07-15 recording
   contains a 0.70 s (21-frame) unfillable dropout at frames 86–106. Inspection
   of those head-cam frames showed **the wires blocked the only visible marker
   face**. Two independent fixes follow, both in the pre-flight above: structural
   wire routing (the trigger) and face redundancy (the underlying fragility — one
   visible face means one obstruction ends tracking). No filter recovers 0.7 s of
   missing information; this was always a physical problem.

   **Re-measure coverage after the fix.** If the pilot's raw tracked coverage
   rises to ~95%+, we are in UMI's regime, where rare bad episodes are simply
   rejected and the smoother becomes a safety net rather than a dependency.

## Non-goals

- Online EKF / causal filtering — a deployment-side component; the offline
  smoother already covers training-label repair better (see the
  2026-07-20 offline-smoother spec).
- Multi-task or language-conditioned collection.
- Object or lighting generalization.
- Deployment, IK, and Route A execution.
