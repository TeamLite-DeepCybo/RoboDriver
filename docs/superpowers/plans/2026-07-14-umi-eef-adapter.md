# UMI EEF-Pose RoboDriver Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new RoboDriver robot adapter `deepcybo-lite-umi-ros2` that records LeRobot datasets live from the UMI handheld rig, with state = per-arm end-effector poses in the world-tag frame (+ gripper opening + quality flags).

**Architecture:** Sibling package to the existing joint-space `robodriver-robot-deepcybo-lite-aio-ros2` adapter, mirroring its file layout and host contract (`robot.get_node()` is spun by `robodriver/core/ros2thread.py`). All pose math and dropout/hold logic live in two pure-numpy modules (`se3.py`, `compose.py`) tested without ROS; `node.py` is thin rclpy glue; `robot.py` implements the LeRobot `Robot` interface.

**Tech Stack:** Python ≥3.10, rclpy (ROS 2 Jazzy), numpy, opencv-python, lerobot (Robot base + dataset pipeline), `lite_aruco_umi_msgs/GripperTrack` (from the collection workspace).

**Spec:** `docs/superpowers/specs/2026-07-14-umi-eef-adapter-design.md`. One layout deviation from spec §3, following the aio package convention: the spec's `scripts/*.py` are implemented as package modules with `[project.scripts]` console entry points (like aio's `mock_recording.py`/`smoke_record.py`), and unit tests live in `tests/` at the package root.

## Global Constraints

- Branch: `feat/umi-eef-adapter`. Commits: **no Claude/AI attribution lines** (company repo policy). Do not push without the user's say-so.
- Do **not** modify `robodriver-robot-deepcybo-lite-aio-ros2` or any file outside the new package + docs.
- New package root: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/` (inner python package `robodriver_robot_deepcybo_lite_umi_ros2`).
- Registration names: config `"deepcybo-lite-umi-ros2"`, status `"deepcybo-lite-umi-ros2"`, node name `"deepcybo_lite_umi_ros2_driver"`.
- State = 16 floats `[L eepose7, L grip, R eepose7, R grip]`, quaternion order **(x, y, z, w)**, raw SI units, no normalization. Quality = 7 floats. Action mirrors the 16 state floats exactly.
- Stamp pairing tolerance: **5 ms**. Never raise from `get_observation()` after `connect()`; hold last pose with flags zeroed.
- `send_action()` raises `NotImplementedError("UMI rig is passive; deploy via joint-space replay (Route B) or a future IK bridge")`.
- **Environments:** `[PURE]` steps run anywhere (Windows dev box OK: `python -m pytest`). `[ROS]` steps need the Linux collection machine with ROS 2 Jazzy + the collection workspace (`lite_aruco_umi_msgs`) + the RoboDriver env (lerobot) sourced. Every task ends with at least a `[PURE]` `python -m py_compile` gate so work can proceed on Windows; `[ROS]` verification is batched in Tasks 8–10.
- All `pytest` commands run from the package root: `cd robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2`.

## File Structure

```
robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/
  pyproject.toml                                # Task 1
  README.md                                     # Task 1 (stub), Task 11 (full)
  robodriver_robot_deepcybo_lite_umi_ros2/
    __init__.py                                 # Task 1
    se3.py            # quat/matrix/pose math, pure numpy        Task 1
    compose.py        # WorldBuffer, EefComposer, vectors, pure  Tasks 2–3
    config.py         # topics, feature names, RobotConfig       Task 4
    status.py         # RobotStatus subclass                     Task 4
    node.py           # rclpy node: subscribe/cache/compose      Task 5, 9
    robot.py          # LeRobot Robot subclass                   Task 6
    mock_umi_topics.py  # synthetic rig publisher                Task 7
    smoke_record.py     # end-to-end record smoke test           Task 8
    visualize_episode.py # post-hoc RViz episode viewer          Task 10
  tests/
    test_se3.py                                  # Task 1
    test_compose.py                              # Tasks 2–3
```

---

### Task 1: Package scaffold + `se3.py`

**Files:**
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/pyproject.toml`
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/README.md`
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/__init__.py`
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/se3.py`
- Test: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_se3.py`

**Interfaces:**
- Produces: `se3.pos_quat_to_T(pos, quat_xyzw) -> np.ndarray (4,4) float64`; `se3.T_to_pos_quat(T) -> (np.ndarray(3,), np.ndarray(4,))` quat xyzw, w ≥ 0; both used by `compose.py` (Task 2–3) and `node.py` (Task 5).

- [ ] **Step 1: Write the failing tests** `[PURE]`

```python
# tests/test_se3.py
import numpy as np
import pytest

from robodriver_robot_deepcybo_lite_umi_ros2 import se3

SQ2 = np.sqrt(2.0) / 2.0


def test_identity_round_trip():
    T = se3.pos_quat_to_T([0, 0, 0], [0, 0, 0, 1])
    assert np.allclose(T, np.eye(4))
    pos, quat = se3.T_to_pos_quat(T)
    assert np.allclose(pos, [0, 0, 0])
    assert np.allclose(quat, [0, 0, 0, 1])


def test_translation_only():
    T = se3.pos_quat_to_T([0.1, -0.2, 0.5], [0, 0, 0, 1])
    assert np.allclose(T[:3, 3], [0.1, -0.2, 0.5])
    assert np.allclose(T[:3, :3], np.eye(3))


def test_90deg_about_z():
    # quat (0,0,sin45,cos45) == +90 deg about z: x-axis -> y-axis
    T = se3.pos_quat_to_T([0, 0, 0], [0, 0, SQ2, SQ2])
    expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(T[:3, :3], expected_R, atol=1e-12)


def test_round_trip_pose():
    pos = [0.3, -0.1, 0.9]
    quat = [0.18257419, 0.36514837, 0.54772256, 0.73029674]  # normalized (1,2,3,4)
    T = se3.pos_quat_to_T(pos, quat)
    pos2, quat2 = se3.T_to_pos_quat(T)
    assert np.allclose(pos2, pos, atol=1e-9)
    assert np.allclose(quat2, quat, atol=1e-9)  # w>0 branch preserved


def test_quat_normalized_on_input():
    T = se3.pos_quat_to_T([0, 0, 0], [0, 0, 2 * SQ2, 2 * SQ2])  # unnormalized
    expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(T[:3, :3], expected_R, atol=1e-12)


def test_compose_world_head_times_head_tcp():
    # T_world_head: +1m z translation, 180 deg about x  (quat (1,0,0,0))
    T_wh = se3.pos_quat_to_T([0.0, 0.0, 1.0], [1, 0, 0, 0])
    # T_head_tcp: pure translation
    T_ht = se3.pos_quat_to_T([0.1, 0.2, 0.5], [0, 0, 0, 1])
    T_wt = T_wh @ T_ht
    pos, quat = se3.T_to_pos_quat(T_wt)
    # R(180x) @ (0.1,0.2,0.5) = (0.1,-0.2,-0.5); + (0,0,1) => (0.1,-0.2,0.5)
    assert np.allclose(pos, [0.1, -0.2, 0.5], atol=1e-12)
    assert np.allclose(np.abs(quat), [1, 0, 0, 0], atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail** `[PURE]`

Run: `cd robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2 && python -m pytest tests/test_se3.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: robodriver_robot_deepcybo_lite_umi_ros2`

- [ ] **Step 3: Write the scaffold + implementation** `[PURE]`

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "robodriver_robot_deepcybo_lite_umi_ros2"
version = "0.1.0"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
authors = [
    {name = "DeepCybo - Team Lite", email = "rayyz25@uw.edu"},
]
keywords = ["robotics", "lerobot", "deepcybo", "umi", "ros2"]
dependencies = [
  # ROS2 Jazzy system deps (rclpy, message_filters) via apt/rosdep, not pip.
  # lite_aruco_umi_msgs comes from the sourced collection workspace overlay.
  "logging_mp",
  "opencv-python",
  "numpy",
]

[project.scripts]
deepcybo-lite-umi-mock-ros2 = "robodriver_robot_deepcybo_lite_umi_ros2.mock_umi_topics:main"
deepcybo-lite-umi-smoke-record = "robodriver_robot_deepcybo_lite_umi_ros2.smoke_record:main"
deepcybo-lite-umi-visualize-episode = "robodriver_robot_deepcybo_lite_umi_ros2.visualize_episode:main"

[tool.setuptools.packages.find]
include = ["robodriver_robot_deepcybo_lite_umi_ros2"]
```

`README.md` (stub; completed in Task 11):

```markdown
# robodriver-robot-deepcybo-lite-umi-ros2

RoboDriver adapter for the DeepCybo Lite **UMI handheld rig**: records LeRobot
datasets with state = per-arm end-effector poses in the world-tag frame.
See `docs/superpowers/specs/2026-07-14-umi-eef-adapter-design.md`.
```

`robodriver_robot_deepcybo_lite_umi_ros2/__init__.py`:

```python
__version__ = "0.1.0"
```

`robodriver_robot_deepcybo_lite_umi_ros2/se3.py`:

```python
"""Minimal SE(3) helpers (pure numpy, no ROS imports).

Quaternions are (x, y, z, w). `T_to_pos_quat` returns w >= 0 so round trips
are stable (q and -q are the same rotation).
"""
from __future__ import annotations

import numpy as np


def pos_quat_to_T(pos, quat_xyzw) -> np.ndarray:
    """Position (3,) + quaternion xyzw (4,) -> 4x4 homogeneous transform."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = q / n
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def T_to_pos_quat(T) -> tuple[np.ndarray, np.ndarray]:
    """4x4 transform -> (pos (3,), quat xyzw (4,) with w >= 0)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    if quat[3] < 0.0:
        quat = -quat
    quat /= np.linalg.norm(quat)
    return T[:3, 3].copy(), quat
```

- [ ] **Step 4: Run tests to verify they pass** `[PURE]`

Run: `python -m pytest tests/test_se3.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Scaffold deepcybo-lite-umi-ros2 adapter package with SE(3) helpers"
```

---

### Task 2: `compose.py` — stamp utils + `WorldBuffer`

**Files:**
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/compose.py`
- Test: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_compose.py`

**Interfaces:**
- Consumes: `se3.pos_quat_to_T`, `se3.T_to_pos_quat` (Task 1).
- Produces: `stamp_to_ns(sec: int, nanosec: int) -> int`; `STAMP_TOLERANCE_NS = 5_000_000`; `class WorldBuffer` with `add(stamp_ns: int, T_world_head: np.ndarray) -> None` and `lookup(stamp_ns: int, tol_ns: int = STAMP_TOLERANCE_NS) -> np.ndarray | None`. Used by Task 3 and `node.py` (Task 5).

- [ ] **Step 1: Write the failing tests** `[PURE]`

```python
# tests/test_compose.py
import numpy as np
import pytest

from robodriver_robot_deepcybo_lite_umi_ros2 import se3
from robodriver_robot_deepcybo_lite_umi_ros2.compose import (
    STAMP_TOLERANCE_NS,
    WorldBuffer,
    stamp_to_ns,
)


def test_stamp_to_ns():
    assert stamp_to_ns(0, 0) == 0
    assert stamp_to_ns(1, 500_000_000) == 1_500_000_000
    assert stamp_to_ns(1700000000, 123) == 1_700_000_000_000_000_123


def test_world_buffer_exact_hit():
    buf = WorldBuffer()
    T = se3.pos_quat_to_T([0, 0, 1], [1, 0, 0, 0])
    buf.add(1_000_000_000, T)
    got = buf.lookup(1_000_000_000)
    assert got is not None
    assert np.allclose(got, T)


def test_world_buffer_nearest_within_tolerance():
    buf = WorldBuffer()
    T = se3.pos_quat_to_T([0, 0, 1], [0, 0, 0, 1])
    buf.add(1_000_000_000, T)
    assert buf.lookup(1_000_000_000 + STAMP_TOLERANCE_NS) is not None
    assert buf.lookup(1_000_000_000 - STAMP_TOLERANCE_NS) is not None


def test_world_buffer_miss_beyond_tolerance():
    buf = WorldBuffer()
    buf.add(1_000_000_000, np.eye(4))
    assert buf.lookup(1_000_000_000 + STAMP_TOLERANCE_NS + 1) is None
    assert buf.lookup(2_000_000_000) is None


def test_world_buffer_picks_nearest_of_several():
    buf = WorldBuffer()
    T_near = se3.pos_quat_to_T([1, 0, 0], [0, 0, 0, 1])
    buf.add(1_000_000_000, np.eye(4))
    buf.add(1_000_004_000, T_near)  # 4 us closer to query
    got = buf.lookup(1_000_003_000)
    assert np.allclose(got, T_near)


def test_world_buffer_evicts_oldest():
    buf = WorldBuffer(maxlen=3)
    for i in range(5):
        buf.add(i * 1_000_000_000, np.eye(4))
    assert buf.lookup(0) is None          # evicted
    assert buf.lookup(1_000_000_000) is None  # evicted
    assert buf.lookup(4_000_000_000) is not None
```

- [ ] **Step 2: Run tests to verify they fail** `[PURE]`

Run: `python -m pytest tests/test_compose.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` on `compose`

- [ ] **Step 3: Write minimal implementation** `[PURE]`

```python
# robodriver_robot_deepcybo_lite_umi_ros2/compose.py
"""Pure composition logic: stamp pairing, world-pose buffer, per-arm hold-last.

No ROS imports so tests/test_compose.py runs anywhere.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

STAMP_TOLERANCE_NS = 5_000_000  # 5 ms — spec §6


def stamp_to_ns(sec: int, nanosec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nanosec)


class WorldBuffer:
    """Last-N T_world_head poses keyed by stamp (ns), nearest-lookup."""

    def __init__(self, maxlen: int = 30) -> None:
        self._maxlen = maxlen
        self._buf: OrderedDict[int, np.ndarray] = OrderedDict()

    def add(self, stamp_ns: int, T_world_head: np.ndarray) -> None:
        self._buf[int(stamp_ns)] = np.asarray(T_world_head, dtype=np.float64)
        while len(self._buf) > self._maxlen:
            self._buf.popitem(last=False)

    def lookup(
        self, stamp_ns: int, tol_ns: int = STAMP_TOLERANCE_NS
    ) -> np.ndarray | None:
        if not self._buf:
            return None
        stamp_ns = int(stamp_ns)
        exact = self._buf.get(stamp_ns)
        if exact is not None:
            return exact
        best = min(self._buf, key=lambda s: abs(s - stamp_ns))
        if abs(best - stamp_ns) <= tol_ns:
            return self._buf[best]
        return None
```

- [ ] **Step 4: Run tests to verify they pass** `[PURE]`

Run: `python -m pytest tests/test_compose.py tests/test_se3.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add world-pose stamp buffer with 5ms nearest pairing"
```

---

### Task 3: `compose.py` — `EefComposer` hold-last + state/quality vectors

**Files:**
- Modify: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/compose.py` (append)
- Test: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_compose.py` (append)

**Interfaces:**
- Consumes: `WorldBuffer` (Task 2), `se3.T_to_pos_quat` (Task 1).
- Produces (used by `node.py` Task 5 and `robot.py` Task 6):
  - `@dataclass EefState`: `pose7: np.ndarray(7,) float32` (x,y,z,qx,qy,qz,qw), `valid: bool`, `tracked: float`, `present: float`, `reproj: float`, `world_fresh: float`.
  - `class EefComposer` with `update(stamp_ns: int, T_head_tcp: np.ndarray | None, tracked: bool, present: bool, reproj: float, world: WorldBuffer) -> EefState` (hold-last semantics; `world_fresh=1.0` only when a fresh compose happened this call).
  - `build_state_vector(left: EefState, right: EefState, left_grip: float, right_grip: float) -> np.ndarray(16,) float32`.
  - `build_quality_vector(left: EefState, right: EefState) -> np.ndarray(7,) float32` ordered `[L_tracked, L_present, L_reproj, R_tracked, R_present, R_reproj, world_fresh]` with `world_fresh = max(left.world_fresh, right.world_fresh)`.
  - `REPROJ_INVALID = 999.0` (clamp for non-finite reproj so datasets never contain inf/nan).

- [ ] **Step 1: Append the failing tests** `[PURE]`

```python
# append to tests/test_compose.py
from robodriver_robot_deepcybo_lite_umi_ros2.compose import (
    REPROJ_INVALID,
    EefComposer,
    build_quality_vector,
    build_state_vector,
)

T_WH = se3.pos_quat_to_T([0.0, 0.0, 1.0], [1, 0, 0, 0])  # 180deg about x, 1m up
T_HT = se3.pos_quat_to_T([0.1, 0.2, 0.5], [0, 0, 0, 1])  # pure translation
STAMP = 1_000_000_000


def _fresh_world() -> WorldBuffer:
    buf = WorldBuffer()
    buf.add(STAMP, T_WH)
    return buf


def test_composer_fresh_compose():
    comp = EefComposer()
    st = comp.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                     world=_fresh_world())
    assert st.valid
    assert st.tracked == 1.0 and st.present == 1.0 and st.world_fresh == 1.0
    assert st.reproj == pytest.approx(0.4)
    assert np.allclose(st.pose7[:3], [0.1, -0.2, 0.5], atol=1e-6)
    assert np.allclose(np.abs(st.pose7[3:]), [1, 0, 0, 0], atol=1e-6)


def test_composer_holds_pose_on_tracking_loss():
    comp = EefComposer()
    comp.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                world=_fresh_world())
    held = comp.update(STAMP + 33_000_000, None, tracked=False, present=False,
                       reproj=float("inf"), world=_fresh_world())
    assert held.valid                       # still have a usable pose
    assert held.tracked == 0.0 and held.present == 0.0
    assert held.world_fresh == 0.0          # no fresh compose this frame
    assert held.reproj == REPROJ_INVALID    # inf clamped
    assert np.allclose(held.pose7[:3], [0.1, -0.2, 0.5], atol=1e-6)  # held


def test_composer_world_miss_holds_and_flags():
    comp = EefComposer()
    comp.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                world=_fresh_world())
    empty = WorldBuffer()  # no world pose at all
    st = comp.update(STAMP + 33_000_000, T_HT, tracked=True, present=True,
                     reproj=0.5, world=empty)
    assert st.tracked == 1.0 and st.present == 1.0
    assert st.world_fresh == 0.0
    assert np.allclose(st.pose7[:3], [0.1, -0.2, 0.5], atol=1e-6)  # held


def test_composer_recovers_after_dropout():
    comp = EefComposer()
    comp.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                world=_fresh_world())
    comp.update(STAMP + 1, None, tracked=False, present=False,
                reproj=float("nan"), world=_fresh_world())
    buf = WorldBuffer()
    buf.add(STAMP + 2, T_WH)
    T_HT2 = se3.pos_quat_to_T([0.2, 0.0, 0.5], [0, 0, 0, 1])
    st = comp.update(STAMP + 2, T_HT2, tracked=True, present=True, reproj=0.3,
                     world=buf)
    assert st.world_fresh == 1.0
    assert np.allclose(st.pose7[:3], [0.2, 0.0, 0.5], atol=1e-6)


def test_composer_invalid_before_first_compose():
    comp = EefComposer()
    st = comp.update(STAMP, None, tracked=False, present=False, reproj=0.0,
                     world=_fresh_world())
    assert not st.valid


def test_state_vector_ordering():
    left, right = EefComposer(), EefComposer()
    ls = left.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                     world=_fresh_world())
    T_HT_R = se3.pos_quat_to_T([-0.1, 0.2, 0.5], [0, 0, 0, 1])
    rs = right.update(STAMP, T_HT_R, tracked=True, present=True, reproj=0.6,
                      world=_fresh_world())
    vec = build_state_vector(ls, rs, left_grip=0.25, right_grip=0.75)
    assert vec.shape == (16,) and vec.dtype == np.float32
    assert np.allclose(vec[0:7], ls.pose7)
    assert vec[7] == pytest.approx(0.25)
    assert np.allclose(vec[8:15], rs.pose7)
    assert vec[15] == pytest.approx(0.75)


def test_quality_vector_ordering_and_world_fresh_max():
    left, right = EefComposer(), EefComposer()
    ls = left.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                     world=_fresh_world())
    rs = right.update(STAMP, None, tracked=False, present=False, reproj=2.0,
                      world=_fresh_world())
    q = build_quality_vector(ls, rs)
    assert q.shape == (7,) and q.dtype == np.float32
    assert list(q[:3]) == [1.0, 1.0, pytest.approx(0.4)]
    assert list(q[3:6]) == [0.0, 0.0, pytest.approx(2.0)]
    assert q[6] == 1.0  # left composed fresh => world was available
```

- [ ] **Step 2: Run tests to verify the new ones fail** `[PURE]`

Run: `python -m pytest tests/test_compose.py -v`
Expected: prior 6 pass; new 7 FAIL with `ImportError: cannot import name 'EefComposer'`

- [ ] **Step 3: Append the implementation to `compose.py`** `[PURE]`

```python
# append to robodriver_robot_deepcybo_lite_umi_ros2/compose.py
from dataclasses import dataclass, field, replace

from . import se3

REPROJ_INVALID = 999.0


def _finite_or_invalid(v: float) -> float:
    v = float(v)
    return v if np.isfinite(v) else REPROJ_INVALID


@dataclass
class EefState:
    """Per-arm composed eef state with quality flags (spec §5)."""

    pose7: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    valid: bool = False        # a pose has been composed at least once
    tracked: float = 0.0
    present: float = 0.0
    reproj: float = REPROJ_INVALID
    world_fresh: float = 0.0   # 1.0 only when composed fresh this frame


class EefComposer:
    """Hold-last composer for one arm (spec §6–7)."""

    def __init__(self) -> None:
        self._state = EefState()

    def update(
        self,
        stamp_ns: int,
        T_head_tcp: np.ndarray | None,
        *,
        tracked: bool,
        present: bool,
        reproj: float,
        world: WorldBuffer,
    ) -> EefState:
        world_fresh = 0.0
        if present and T_head_tcp is not None:
            T_world_head = world.lookup(stamp_ns)
            if T_world_head is not None:
                T_world_tcp = T_world_head @ np.asarray(T_head_tcp, np.float64)
                pos, quat = se3.T_to_pos_quat(T_world_tcp)
                self._state.pose7 = np.concatenate([pos, quat]).astype(np.float32)
                self._state.valid = True
                world_fresh = 1.0
        # tracking loss or world miss: pose7 held from last success
        self._state.tracked = 1.0 if tracked else 0.0
        self._state.present = 1.0 if present else 0.0
        self._state.reproj = _finite_or_invalid(reproj)
        self._state.world_fresh = world_fresh
        return replace(self._state, pose7=self._state.pose7.copy())


def build_state_vector(
    left: EefState, right: EefState, left_grip: float, right_grip: float
) -> np.ndarray:
    """16-dim [L eepose7, L grip, R eepose7, R grip] (spec §5)."""
    return np.concatenate(
        [
            left.pose7,
            np.array([left_grip], dtype=np.float32),
            right.pose7,
            np.array([right_grip], dtype=np.float32),
        ]
    ).astype(np.float32)


def build_quality_vector(left: EefState, right: EefState) -> np.ndarray:
    """[L_tracked, L_present, L_reproj, R_tracked, R_present, R_reproj, world_fresh]."""
    return np.array(
        [
            left.tracked,
            left.present,
            left.reproj,
            right.tracked,
            right.present,
            right.reproj,
            max(left.world_fresh, right.world_fresh),
        ],
        dtype=np.float32,
    )
```

- [ ] **Step 4: Run tests to verify they pass** `[PURE]`

Run: `python -m pytest tests/ -v`
Expected: 19 passed

- [ ] **Step 5: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add per-arm hold-last eef composer and state/quality vectors"
```

---

### Task 4: `config.py` + `status.py`

**Files:**
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/config.py`
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/status.py`

**Interfaces:**
- Consumes: `lerobot.cameras.CameraConfig`, `lerobot.cameras.opencv.OpenCVCameraConfig`, `lerobot.robots.config.RobotConfig` (same imports as the aio adapter's `config.py`); aio's `status.py` dataclasses are re-created locally (they are package-local there too).
- Produces (used by Tasks 5–8):
  - `EEF_FEATURE_NAMES: tuple[str, ...]` — exactly 16 names in order: `left_eef_x, left_eef_y, left_eef_z, left_eef_qx, left_eef_qy, left_eef_qz, left_eef_qw, left_gripper, right_eef_x, right_eef_y, right_eef_z, right_eef_qx, right_eef_qy, right_eef_qz, right_eef_qw, right_gripper`.
  - `QUALITY_FEATURE_NAMES: tuple[str, ...]` — exactly 7: `left_tracked, left_present, left_reproj, right_tracked, right_present, right_reproj, world_fresh`.
  - `GRIPPER_JOINTS = ("left_gripper", "right_gripper")`.
  - `@dataclass DeepcyboLiteUmiRos2Topics` with fields `track_left, track_right, world_head, joint_states, camera_head, camera_wrist_left, camera_wrist_right` (defaults per spec §4).
  - `DeepcyboLiteUmiRos2RobotConfig(RobotConfig)` registered `"deepcybo-lite-umi-ros2"` with `cameras` (3× `OpenCVCameraConfig` 640×480@30 — keys `image_head`, `image_wrist_left`, `image_wrist_right`), `control_fps=30`, `camera_fps=30`, `publish_debug: bool = False`, `use_videos: bool = False`, `microphones: Dict[str, int]`, `ros2_topics: DeepcyboLiteUmiRos2Topics`.
  - `DeepcyboLiteUmiRos2RobotStatus` registered `"deepcybo-lite-umi-ros2"`.

- [ ] **Step 1: Write `config.py`** `[PURE]` (verification of lerobot imports happens in Step 3 `[ROS]` and Task 8)

```python
# robodriver_robot_deepcybo_lite_umi_ros2/config.py
"""DeepCybo Lite UMI handheld rig — RoboDriver config and ROS2 topic contract.

State vector (16-dim, spec §5):
  left eef pose7 (x,y,z,qx,qy,qz,qw) | left gripper |
  right eef pose7                    | right gripper
Quality vector (7-dim): flags for filtering only — never feed to the policy.
"""
from dataclasses import dataclass, field
from typing import Dict

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig


@dataclass
class DeepcyboLiteUmiRos2Topics:
    """All live inputs of the UMI rig (spec §4). Overridable at instantiation."""

    track_left: str = "/umi/left/track"        # lite_aruco_umi_msgs/GripperTrack
    track_right: str = "/umi/right/track"      # lite_aruco_umi_msgs/GripperTrack
    world_head: str = "/umi/world_head/pose"   # geometry_msgs/PoseStamped
    joint_states: str = "/lite/joint_states"   # sensor_msgs/JointState
    camera_head: str = "/deepcybo/lite/camera/head/image_raw/compressed"
    camera_wrist_left: str = "/deepcybo/lite/camera/wrist_left/image_raw/compressed"
    camera_wrist_right: str = "/deepcybo/lite/camera/wrist_right/image_raw/compressed"


EEF_FEATURE_NAMES: tuple[str, ...] = (
    "left_eef_x", "left_eef_y", "left_eef_z",
    "left_eef_qx", "left_eef_qy", "left_eef_qz", "left_eef_qw",
    "left_gripper",
    "right_eef_x", "right_eef_y", "right_eef_z",
    "right_eef_qx", "right_eef_qy", "right_eef_qz", "right_eef_qw",
    "right_gripper",
)

QUALITY_FEATURE_NAMES: tuple[str, ...] = (
    "left_tracked", "left_present", "left_reproj",
    "right_tracked", "right_present", "right_reproj",
    "world_fresh",
)

GRIPPER_JOINTS: tuple[str, ...] = ("left_gripper", "right_gripper")

STATE_DIM = len(EEF_FEATURE_NAMES)      # 16
QUALITY_DIM = len(QUALITY_FEATURE_NAMES)  # 7


@RobotConfig.register_subclass("deepcybo-lite-umi-ros2")
@dataclass
class DeepcyboLiteUmiRos2RobotConfig(RobotConfig):
    """DeepCybo Lite UMI handheld rig eef-pose collection config."""

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_head": OpenCVCameraConfig(
                index_or_path=0, fps=30, width=640, height=480
            ),
            "image_wrist_left": OpenCVCameraConfig(
                index_or_path=1, fps=30, width=640, height=480
            ),
            "image_wrist_right": OpenCVCameraConfig(
                index_or_path=2, fps=30, width=640, height=480
            ),
        }
    )

    control_fps: int = 30
    camera_fps: int = 30

    # RViz verification overlay (spec §8): off for real sessions.
    publish_debug: bool = False

    use_videos: bool = False
    microphones: Dict[str, int] = field(default_factory=dict)

    ros2_topics: DeepcyboLiteUmiRos2Topics = field(
        default_factory=DeepcyboLiteUmiRos2Topics
    )
```

- [ ] **Step 2: Write `status.py`** `[PURE]`

```python
# robodriver_robot_deepcybo_lite_umi_ros2/status.py
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import abc

import draccus


@dataclass
class CameraInfo:
    name: str = ""
    chinese_name: str = ""
    type: str = ""
    width: int = 0
    height: int = 0
    is_connect: bool = False


@dataclass
class CameraStatus:
    number: int = 0
    information: List[CameraInfo] = field(default_factory=list)

    def __post_init__(self):
        self.number = len(self.information) if self.information else 0


@dataclass
class ArmInfo:
    name: str = ""
    type: str = ""
    start_pose: List[float] = field(default_factory=list)
    joint_p_limit: List[float] = field(default_factory=list)
    joint_n_limit: List[float] = field(default_factory=list)
    is_connect: bool = False


@dataclass
class ArmStatus:
    number: int = 0
    information: List[ArmInfo] = field(default_factory=list)

    def __post_init__(self):
        self.number = len(self.information) if self.information else 0


@dataclass
class Specifications:
    end_type: str = "Default"
    fps: int = 30
    camera: Optional[CameraStatus] = None
    arm: Optional[ArmStatus] = None


@dataclass
class RobotStatus(draccus.ChoiceRegistry, abc.ABC):
    device_name: str = "Default"
    device_body: str = "Default"
    specifications: Specifications = field(default_factory=Specifications)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


_CAMERAS = (
    ("image_head", "头部相机"),
    ("image_wrist_left", "左腕相机"),
    ("image_wrist_right", "右腕相机"),
)

LEFT_EEF = "left_eef"
RIGHT_EEF = "right_eef"


RobotStatus.register_subclass("deepcybo-lite-umi-ros2")


@dataclass
class DeepcyboLiteUmiRos2RobotStatus(RobotStatus):
    device_name: str = "DeepCybo Lite UMI"
    device_body: str = "DeepCybo"

    def __post_init__(self):
        self.specifications.end_type = "UMI 双手持夹爪（世界系 eef 位姿）"
        self.specifications.fps = 30
        self.specifications.camera = CameraStatus(
            information=[
                CameraInfo(
                    name=name,
                    chinese_name=cname,
                    type="RGB 相机",
                    width=640,
                    height=480,
                    is_connect=False,
                )
                for name, cname in _CAMERAS
            ]
        )
        self.specifications.arm = ArmStatus(
            information=[
                ArmInfo(
                    name=LEFT_EEF,
                    type="左手持夹爪 eef pose7 + 开合（ArUco 视觉跟踪）",
                    is_connect=False,
                ),
                ArmInfo(
                    name=RIGHT_EEF,
                    type="右手持夹爪 eef pose7 + 开合（ArUco 视觉跟踪）",
                    is_connect=False,
                ),
            ]
        )
```

- [ ] **Step 3: Syntax gate now, import gate on the ROS machine**

Run `[PURE]`: `python -m py_compile robodriver_robot_deepcybo_lite_umi_ros2/config.py robodriver_robot_deepcybo_lite_umi_ros2/status.py`
Expected: exit 0.
Run `[ROS]` (RoboDriver env with lerobot): `python -c "from robodriver_robot_deepcybo_lite_umi_ros2.config import DeepcyboLiteUmiRos2RobotConfig, EEF_FEATURE_NAMES; from robodriver_robot_deepcybo_lite_umi_ros2.status import DeepcyboLiteUmiRos2RobotStatus; assert len(EEF_FEATURE_NAMES) == 16; print(DeepcyboLiteUmiRos2RobotConfig().type)"`
Expected: `deepcybo-lite-umi-ros2`

- [ ] **Step 4: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add UMI adapter config (topics, 16+7 feature schema) and status"
```

---

### Task 5: `node.py` — rclpy glue

**Files:**
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/node.py`

**Interfaces:**
- Consumes: `DeepcyboLiteUmiRos2Topics`, `GRIPPER_JOINTS` (Task 4); `WorldBuffer`, `EefComposer`, `build_state_vector`, `build_quality_vector`, `stamp_to_ns` (Tasks 2–3); `se3.pos_quat_to_T` (Task 1); `lite_aruco_umi_msgs.msg.GripperTrack` (collection workspace overlay).
- Produces (used by `robot.py` Task 6; debug topics extended in Task 9):
  - `class DeepcyboLiteUmiRos2RobotNode(ROS2Node)` with constructor `(topics=None, control_fps=30, camera_fps=30, publish_debug=False)`.
  - Attributes: `recv_images: Dict[str, np.ndarray]`, `recv_images_status: Dict[str, int]`, `lock: threading.Lock`.
  - `left_valid() -> bool`, `right_valid() -> bool`, `grippers_valid() -> bool` — connect gating.
  - `state_vector() -> np.ndarray(16,) | None`, `quality_vector() -> np.ndarray(7,) | None` — None until both arms valid and grippers seen.
  - `destroy() -> None`.

- [ ] **Step 1: Write `node.py`** `[PURE]` (syntax gate here; behavior verified in Tasks 7–8)

```python
# robodriver_robot_deepcybo_lite_umi_ros2/node.py
"""DeepCybo Lite UMI rig — ROS2 subscribe / stamp-pair / compose / cache.

Data flow per head-camera frame (all stamps identical — they derive from the
same image):  GripperTrack (head frame)  +  world_head PoseStamped
              -> T_world_tcp = T_world_head @ T_head_tcp  (compose.py)
Gripper opening comes from /lite/joint_states; cameras via approx-time sync.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node as ROS2Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, JointState

import logging_mp

from . import se3
from .compose import (
    EefComposer,
    WorldBuffer,
    build_quality_vector,
    build_state_vector,
    stamp_to_ns,
)
from .config import GRIPPER_JOINTS, DeepcyboLiteUmiRos2Topics

try:
    from lite_aruco_umi_msgs.msg import GripperTrack
except ImportError as exc:  # pragma: no cover - needs collection ws overlay
    GripperTrack = None
    _MSGS_IMPORT_ERROR = exc
else:
    _MSGS_IMPORT_ERROR = None

CONNECT_TIMEOUT_FRAME = 10
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

logger = logging_mp.get_logger(__name__)


def _pose_to_T(pose) -> np.ndarray:
    """geometry_msgs/Pose -> 4x4 (pure floats in, se3 does the math)."""
    p, q = pose.position, pose.orientation
    return se3.pos_quat_to_T([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])


class DeepcyboLiteUmiRos2RobotNode(ROS2Node):
    def __init__(
        self,
        topics: Optional[DeepcyboLiteUmiRos2Topics] = None,
        control_fps: int = 30,
        camera_fps: int = 30,
        publish_debug: bool = False,
    ):
        if GripperTrack is None:
            raise ImportError(
                "Cannot import lite_aruco_umi_msgs.msg.GripperTrack. Source the "
                "collection workspace overlay before starting RoboDriver, e.g. "
                "`source ~/ros2_ws/install/setup.bash`."
            ) from _MSGS_IMPORT_ERROR

        super().__init__("deepcybo_lite_umi_ros2_driver")
        self.topics = topics or DeepcyboLiteUmiRos2Topics()
        self.control_fps = control_fps
        self.camera_fps = camera_fps
        self.publish_debug = bool(publish_debug)

        self.qos = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.qos_best_effort = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        t = self.topics
        self.create_subscription(
            GripperTrack, t.track_left,
            lambda msg: self._track_callback("left", msg), self.qos,
        )
        self.create_subscription(
            GripperTrack, t.track_right,
            lambda msg: self._track_callback("right", msg), self.qos,
        )
        self.create_subscription(
            PoseStamped, t.world_head, self._world_callback, self.qos,
        )
        self.create_subscription(
            JointState, t.joint_states, self._joint_state_callback,
            self.qos_best_effort,
        )

        self.last_image_recv_time_ns = 0
        self.min_camera_interval_ns = int(1e9 / max(camera_fps, 1))

        self._world = WorldBuffer()
        self._composer = {"left": EefComposer(), "right": EefComposer()}
        self._eef_state = {"left": None, "right": None}   # latest EefState
        self._grippers: Dict[str, float] = {}             # joint -> opening

        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}

        self.lock = threading.Lock()

        self._init_image_message_filters()
        self._init_debug_publishers()   # no-op unless publish_debug (Task 9)

        logger.info(
            "[DeepCybo Lite UMI] node ready | tracks=(%s, %s) world=%s "
            "joints=%s control_fps=%s camera_fps=%s debug=%s",
            t.track_left, t.track_right, t.world_head, t.joint_states,
            control_fps, camera_fps, self.publish_debug,
        )

    # ------------------------------------------------------------------
    # Stream A + C -> composed world-frame eef state
    # ------------------------------------------------------------------
    def _world_callback(self, msg: PoseStamped) -> None:
        try:
            ns = stamp_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            with self.lock:
                self._world.add(ns, _pose_to_T(msg.pose))
        except Exception as e:
            self.get_logger().error(f"world_head callback error: {e}")

    def _track_callback(self, arm: str, msg) -> None:
        try:
            ns = stamp_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            usable = bool(msg.present) and bool(msg.has_tcp)
            T_head_tcp = _pose_to_T(msg.tcp_pose) if usable else None
            with self.lock:
                state = self._composer[arm].update(
                    ns,
                    T_head_tcp,
                    tracked=bool(msg.tracked),
                    present=bool(msg.present),
                    reproj=float(msg.reproj),
                    world=self._world,
                )
                self._eef_state[arm] = state
            self._publish_debug_pose(arm, state, msg.header.stamp)  # Task 9
        except Exception as e:
            self.get_logger().error(f"track[{arm}] callback error: {e}")

    # ------------------------------------------------------------------
    # Stream B — gripper opening
    # ------------------------------------------------------------------
    def _joint_state_callback(self, msg: JointState) -> None:
        try:
            index = {name: i for i, name in enumerate(msg.name)}
            if not all(j in index for j in GRIPPER_JOINTS):
                return
            with self.lock:
                for joint in GRIPPER_JOINTS:
                    i = index[joint]
                    if i < len(msg.position):
                        self._grippers[joint] = float(msg.position[i])
        except Exception as e:
            self.get_logger().error(f"JointState callback error: {e}")

    # ------------------------------------------------------------------
    # Cameras — 3x CompressedImage @ camera_fps (aio pattern)
    # ------------------------------------------------------------------
    def _init_image_message_filters(self) -> None:
        t = self.topics
        sub_head = Subscriber(self, CompressedImage, t.camera_head)
        sub_wrist_l = Subscriber(self, CompressedImage, t.camera_wrist_left)
        sub_wrist_r = Subscriber(self, CompressedImage, t.camera_wrist_right)
        self.image_sync = ApproximateTimeSynchronizer(
            [sub_head, sub_wrist_l, sub_wrist_r], queue_size=5, slop=0.05
        )
        self.image_sync.registerCallback(self._image_synchronized_callback)

    def _image_synchronized_callback(self, head, wrist_left, wrist_right) -> None:
        try:
            now = time.time_ns()
            if now - self.last_image_recv_time_ns < self.min_camera_interval_ns:
                return
            self.last_image_recv_time_ns = now
            self._images_recv(head, "image_head")
            self._images_recv(wrist_left, "image_wrist_left")
            self._images_recv(wrist_right, "image_wrist_right")
        except Exception as e:
            self.get_logger().error(f"Image synchronized callback error: {e}")

    def _images_recv(self, msg: CompressedImage, event_id: str) -> None:
        try:
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] != CAMERA_HEIGHT or frame.shape[1] != CAMERA_WIDTH:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            with self.lock:
                self.recv_images[event_id] = frame
                self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"recv image error ({event_id}): {e}")

    # ------------------------------------------------------------------
    # Accessors for robot.py (connect gating + per-frame vectors)
    # ------------------------------------------------------------------
    def left_valid(self) -> bool:
        with self.lock:
            s = self._eef_state["left"]
            return s is not None and s.valid

    def right_valid(self) -> bool:
        with self.lock:
            s = self._eef_state["right"]
            return s is not None and s.valid

    def grippers_valid(self) -> bool:
        with self.lock:
            return all(j in self._grippers for j in GRIPPER_JOINTS)

    def state_vector(self) -> Optional[np.ndarray]:
        with self.lock:
            left, right = self._eef_state["left"], self._eef_state["right"]
            if (
                left is None or right is None
                or not left.valid or not right.valid
                or not all(j in self._grippers for j in GRIPPER_JOINTS)
            ):
                return None
            return build_state_vector(
                left, right,
                self._grippers[GRIPPER_JOINTS[0]],
                self._grippers[GRIPPER_JOINTS[1]],
            )

    def quality_vector(self) -> Optional[np.ndarray]:
        with self.lock:
            left, right = self._eef_state["left"], self._eef_state["right"]
            if left is None or right is None:
                return None
            return build_quality_vector(left, right)

    # ------------------------------------------------------------------
    # Debug overlay — implemented in Task 9; keep no-op stubs until then
    # ------------------------------------------------------------------
    def _init_debug_publishers(self) -> None:
        self._debug_pubs = None

    def _publish_debug_pose(self, arm, state, stamp) -> None:
        return

    def destroy(self) -> None:
        super().destroy_node()
```

- [ ] **Step 2: Syntax gate** `[PURE]`

Run: `python -m py_compile robodriver_robot_deepcybo_lite_umi_ros2/node.py`
Expected: exit 0. (Behavioral verification: Tasks 7–8 on the ROS machine.)

- [ ] **Step 3: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add UMI adapter ROS2 node: subscribe, stamp-pair, compose, cache"
```

---

### Task 6: `robot.py` — LeRobot `Robot` subclass

**Files:**
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/robot.py`

**Interfaces:**
- Consumes: `DeepcyboLiteUmiRos2RobotConfig`, `EEF_FEATURE_NAMES`, `QUALITY_FEATURE_NAMES` (Task 4); `DeepcyboLiteUmiRos2RobotNode` (Task 5); `DeepcyboLiteUmiRos2RobotStatus` (Task 4); `lerobot.robots.robot.Robot`, `lerobot.cameras.make_cameras_from_configs`, `lerobot.utils.errors.{DeviceAlreadyConnectedError, DeviceNotConnectedError}` (as in aio `robot.py`).
- Produces: `class DeepcyboLiteUmiRos2Robot(Robot)` with `name = "deepcybo-lite-umi-ros2"`; features contract used by Task 8:
  - `observation_features`: `{f"{n}.pos": float}` for the 16 names + `{f"{n}.flag": float}` for the 7 quality names + camera shape dict.
  - `action_features`: `{f"{n}.pos": float}` for the 16 names.
  - `connect()`, `get_observation()`, `get_action()`, `send_action()` (raises), `update_status()`, `disconnect()`, `get_node()`.

- [ ] **Step 1: Write `robot.py`** `[PURE]`

```python
# robodriver_robot_deepcybo_lite_umi_ros2/robot.py
import time
from functools import cached_property
from typing import Any

import logging_mp
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import (
    EEF_FEATURE_NAMES,
    QUALITY_FEATURE_NAMES,
    DeepcyboLiteUmiRos2RobotConfig,
)
from .node import DeepcyboLiteUmiRos2RobotNode
from .status import LEFT_EEF, RIGHT_EEF, DeepcyboLiteUmiRos2RobotStatus

logger = logging_mp.get_logger(__name__)


class DeepcyboLiteUmiRos2Robot(Robot):
    config_class = DeepcyboLiteUmiRos2RobotConfig
    name = "deepcybo-lite-umi-ros2"

    def __init__(self, config: DeepcyboLiteUmiRos2RobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.connect_excluded_cameras: list[str] = []

        self.status = DeepcyboLiteUmiRos2RobotStatus()
        self.robot_ros2_node = DeepcyboLiteUmiRos2RobotNode(
            topics=config.ros2_topics,
            control_fps=config.control_fps,
            camera_fps=config.camera_fps,
            publish_debug=config.publish_debug,
        )

        self.connected = False
        self.logs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Feature contract (spec §5)
    # ------------------------------------------------------------------
    @property
    def _state_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in EEF_FEATURE_NAMES}

    @property
    def _quality_ft(self) -> dict[str, type]:
        # Filtering-only flags — training configs must EXCLUDE these from
        # the policy input (spec §10).
        return {f"{name}.flag": float for name in QUALITY_FEATURE_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._quality_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Mirror of the 16 core state floats (spec §5) — no quality flags.
        return {f"{name}.pos": float for name in EEF_FEATURE_NAMES}

    @property
    def is_connected(self) -> bool:
        return self.connected

    # ------------------------------------------------------------------
    def _missing_cameras(self) -> list[str]:
        return [
            name
            for name in self.cameras
            if name not in self.connect_excluded_cameras
            and name not in self.robot_ros2_node.recv_images
        ]

    def _received_cameras(self) -> list[str]:
        return [
            name
            for name in self.cameras
            if name not in self.connect_excluded_cameras
            and name in self.robot_ros2_node.recv_images
        ]

    def connect(self) -> None:
        timeout = 20
        start_time = time.perf_counter()
        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        node = self.robot_ros2_node
        while True:
            cameras_ok = len(self._missing_cameras()) == 0
            left_ok = node.left_valid()
            right_ok = node.right_valid()
            grippers_ok = node.grippers_valid()
            if cameras_ok and left_ok and right_ok and grippers_ok:
                break
            if time.perf_counter() - start_time > timeout:
                parts: list[str] = []
                if not cameras_ok:
                    parts.append(
                        f"cameras missing [{', '.join(self._missing_cameras())}]; "
                        f"received [{', '.join(self._received_cameras())}]"
                    )
                if not left_ok:
                    parts.append(
                        "left eef never composed (needs GripperTrack with "
                        "present+has_tcp AND a stamp-matched world_head pose)"
                    )
                if not right_ok:
                    parts.append("right eef never composed (same requirements)")
                if not grippers_ok:
                    parts.append(
                        "gripper joints not seen on joint_states "
                        "(need left_gripper + right_gripper)"
                    )
                raise TimeoutError("connect timeout, unmet: " + "; ".join(parts))
            time.sleep(0.01)

        logger.info(
            "[connected] cameras=%s | left/right eef composed | grippers ok | %.2fs",
            ", ".join(self._received_cameras()),
            time.perf_counter() - start_time,
        )
        for i in range(self.status.specifications.camera.number):
            self.status.specifications.camera.information[i].is_connect = True
        for i in range(self.status.specifications.arm.number):
            self.status.specifications.arm.information[i].is_connect = True
        self.connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    def _state_and_quality(self) -> dict[str, float]:
        node = self.robot_ros2_node
        state = node.state_vector()
        quality = node.quality_vector()
        if state is None or quality is None:
            # Post-connect this cannot regress to None (hold-last keeps the
            # vectors); guard anyway rather than KeyError downstream.
            raise DeviceNotConnectedError(
                f"{self}: eef state unavailable — was connect() successful?"
            )
        out: dict[str, float] = {}
        for i, name in enumerate(EEF_FEATURE_NAMES):
            out[f"{name}.pos"] = float(state[i])
        for i, name in enumerate(QUALITY_FEATURE_NAMES):
            out[f"{name}.flag"] = float(quality[i])
        return out

    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        obs_dict: dict[str, Any] = self._state_and_quality()
        for cam_key in self.cameras:
            if cam_key in self.robot_ros2_node.recv_images:
                obs_dict[cam_key] = self.robot_ros2_node.recv_images[cam_key]
        return obs_dict

    def get_action(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        state = self._state_and_quality()
        # Action mirrors the 16 core floats exactly (spec §5) — no flags.
        return {
            f"{name}.pos": state[f"{name}.pos"] for name in EEF_FEATURE_NAMES
        }

    def send_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "UMI rig is passive; deploy via joint-space replay (Route B) "
            "or a future IK bridge"
        )

    def update_status(self) -> str:
        node = self.robot_ros2_node
        for cam in self.status.specifications.camera.information:
            cam.is_connect = node.recv_images_status.get(cam.name, 0) > 0
        for arm in self.status.specifications.arm.information:
            if arm.name == LEFT_EEF:
                arm.is_connect = node.left_valid()
            elif arm.name == RIGHT_EEF:
                arm.is_connect = node.right_valid()
        return self.status.to_json()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. Run `robot.connect()` before disconnecting."
            )
        if hasattr(self, "robot_ros2_node"):
            self.robot_ros2_node.destroy()
        self.connected = False

    def __del__(self) -> None:
        try:
            if getattr(self, "connected", False):
                self.disconnect()
        except Exception:
            pass

    def get_node(self) -> DeepcyboLiteUmiRos2RobotNode:
        return self.robot_ros2_node
```

- [ ] **Step 2: Syntax gate** `[PURE]`

Run: `python -m py_compile robodriver_robot_deepcybo_lite_umi_ros2/robot.py`
Expected: exit 0.

- [ ] **Step 3: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add UMI adapter LeRobot Robot: eef features, connect gating, mirrored action"
```

---

### Task 7: `mock_umi_topics.py` — synthetic rig publisher

**Files:**
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/mock_umi_topics.py`

**Interfaces:**
- Consumes: `DeepcyboLiteUmiRos2Topics`, `GRIPPER_JOINTS` (Task 4); `GripperTrack` msg.
- Produces: `class DeepcyboLiteUmiMockNode(ROS2Node)` — constructor `(topics=None, fps=30.0, joint_rate_hz=50.0, drop_every=0, drop_len=15)`; publishes all 7 rig topics with a **single shared stamp per frame**; `main()` console entry (`deepcybo-lite-umi-mock-ros2`). Used by `smoke_record.py` (Task 8) and manual RViz checks (Task 9).

- [ ] **Step 1: Write the mock node** `[PURE]`

```python
# robodriver_robot_deepcybo_lite_umi_ros2/mock_umi_topics.py
"""Synthetic UMI rig publisher for off-rig testing (spec §8).

Per camera-frame tick (fps): ONE stamp shared by both GripperTrack msgs, the
world_head pose, and all three camera images — exactly like the real rig,
where they all derive from the same head image.

--drop-every N: every N frames, publish drop_len frames of "tracking lost"
GripperTrack (present=False) for the LEFT arm to exercise hold-last + flags.
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node as ROS2Node
from sensor_msgs.msg import CompressedImage, JointState

from .config import GRIPPER_JOINTS, DeepcyboLiteUmiRos2Topics

try:
    from lite_aruco_umi_msgs.msg import GripperTrack
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lite_aruco_umi_msgs not on the ROS overlay; source the collection "
        "workspace first."
    ) from exc

# Fixed T_world_head used by every frame: head 1 m above the tag, looking
# straight down (180 deg about x  => quat (1,0,0,0)).
WORLD_HEAD_POS = (0.2, 0.3, 1.0)
WORLD_HEAD_QUAT = (1.0, 0.0, 0.0, 0.0)


def _set_pose(msg: Pose, pos, quat) -> None:
    msg.position.x, msg.position.y, msg.position.z = map(float, pos)
    (msg.orientation.x, msg.orientation.y,
     msg.orientation.z, msg.orientation.w) = map(float, quat)


def _test_image(frame_idx: int, label: str) -> bytes:
    img = np.full((480, 640, 3), 32, dtype=np.uint8)
    cv2.putText(img, f"{label} {frame_idx}", (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class DeepcyboLiteUmiMockNode(ROS2Node):
    def __init__(
        self,
        topics: DeepcyboLiteUmiRos2Topics | None = None,
        fps: float = 30.0,
        joint_rate_hz: float = 50.0,
        drop_every: int = 0,
        drop_len: int = 15,
    ):
        super().__init__("deepcybo_lite_umi_mock")
        t = topics or DeepcyboLiteUmiRos2Topics()
        self.topics = t
        self.drop_every = int(drop_every)
        self.drop_len = int(drop_len)

        self.pub_track_left = self.create_publisher(GripperTrack, t.track_left, 10)
        self.pub_track_right = self.create_publisher(GripperTrack, t.track_right, 10)
        self.pub_world = self.create_publisher(PoseStamped, t.world_head, 10)
        self.pub_joints = self.create_publisher(JointState, t.joint_states, 10)
        self.pub_cam = {
            "head": self.create_publisher(CompressedImage, t.camera_head, 10),
            "wl": self.create_publisher(CompressedImage, t.camera_wrist_left, 10),
            "wr": self.create_publisher(CompressedImage, t.camera_wrist_right, 10),
        }

        self._frame = 0
        self.create_timer(1.0 / fps, self._tick_frame)
        self.create_timer(1.0 / joint_rate_hz, self._tick_joints)
        self.get_logger().info(
            f"UMI mock up @ {fps} Hz (drop_every={drop_every}, drop_len={drop_len})"
        )

    def _dropping_left(self) -> bool:
        if self.drop_every <= 0:
            return False
        return (self._frame % self.drop_every) < self.drop_len

    def _track_msg(self, stamp, phase: float, dropped: bool) -> GripperTrack:
        msg = GripperTrack()
        msg.header.stamp = stamp
        msg.header.frame_id = "head"
        msg.timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if dropped:
            msg.tracked = False
            msg.present = False
            msg.has_tcp = False
            msg.n_markers = 0
            msg.reproj = float("inf")
            return msg
        # circle r=0.15 m at z=0.5 m in the HEAD frame
        pos = (0.15 * math.cos(phase), 0.15 * math.sin(phase), 0.5)
        _set_pose(msg.tcp_pose, pos, (0.0, 0.0, 0.0, 1.0))
        _set_pose(msg.cube_pose, pos, (0.0, 0.0, 0.0, 1.0))
        msg.tracked = True
        msg.present = True
        msg.has_tcp = True
        msg.n_markers = 3
        msg.reproj = 0.5
        return msg

    def _tick_frame(self) -> None:
        stamp = self.get_clock().now().to_msg()
        phase = 2.0 * math.pi * (self._frame / 90.0)  # 3 s per revolution

        self.pub_track_left.publish(
            self._track_msg(stamp, phase, dropped=self._dropping_left())
        )
        self.pub_track_right.publish(
            self._track_msg(stamp, phase + math.pi, dropped=False)
        )

        world = PoseStamped()
        world.header.stamp = stamp
        world.header.frame_id = "world"
        _set_pose(world.pose, WORLD_HEAD_POS, WORLD_HEAD_QUAT)
        self.pub_world.publish(world)

        for key, label in (("head", "HEAD"), ("wl", "WRIST-L"), ("wr", "WRIST-R")):
            img = CompressedImage()
            img.header.stamp = stamp
            img.format = "jpeg"
            img.data = _test_image(self._frame, label)
            self.pub_cam[key].publish(img)

        self._frame += 1

    def _tick_joints(self) -> None:
        now = self.get_clock().now()
        t_sec = now.nanoseconds * 1e-9
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = list(GRIPPER_JOINTS)
        msg.position = [
            0.5 + 0.5 * math.sin(t_sec),          # left in [0,1]
            0.5 + 0.5 * math.sin(t_sec + 1.0),    # right in [0,1]
        ]
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]
        self.pub_joints.publish(msg)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Publish synthetic UMI rig topics.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--joint-rate-hz", type=float, default=50.0)
    parser.add_argument("--drop-every", type=int, default=0,
                        help="every N frames, drop LEFT tracking for --drop-len frames")
    parser.add_argument("--drop-len", type=int, default=15)
    args = parser.parse_args(argv)

    rclpy.init()
    node = DeepcyboLiteUmiMockNode(
        fps=args.fps,
        joint_rate_hz=args.joint_rate_hz,
        drop_every=args.drop_every,
        drop_len=args.drop_len,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax gate** `[PURE]`

Run: `python -m py_compile robodriver_robot_deepcybo_lite_umi_ros2/mock_umi_topics.py`
Expected: exit 0.

- [ ] **Step 3: Manual check on the ROS machine** `[ROS]`

```bash
# shell 1 (collection ws + robodriver env sourced)
python -m robodriver_robot_deepcybo_lite_umi_ros2.mock_umi_topics
# shell 2
ros2 topic hz /umi/left/track            # ~30 Hz
ros2 topic hz /lite/joint_states         # ~50 Hz
ros2 topic echo /umi/world_head/pose --once
```
Expected: both rates present; world pose = (0.2, 0.3, 1.0) quat (1,0,0,0).

- [ ] **Step 4: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add synthetic UMI rig mock publisher with dropout injection"
```

---

### Task 8: `smoke_record.py` — end-to-end acceptance

**Files:**
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/smoke_record.py`

**Interfaces:**
- Consumes: `DeepcyboLiteUmiRos2RobotConfig` (Task 4), `DeepcyboLiteUmiRos2Robot` (Task 6), `DeepcyboLiteUmiMockNode` (Task 7); RoboDriver dataset pipeline exactly as aio's `smoke_record.py` (`DoRobotDataset`, `build_dataset_frame`, `make_default_processors`, `busy_wait`).
- Produces: `main()` console entry (`deepcybo-lite-umi-smoke-record`) that records one episode from the mock and asserts the feature schema; the plan's ROS acceptance gate.

- [ ] **Step 1: Write `smoke_record.py`** `[PURE]`

```python
# robodriver_robot_deepcybo_lite_umi_ros2/smoke_record.py
"""Record a short UMI-rig LeRobot smoke dataset against the mock publisher.

Mirrors the aio adapter's smoke_record.py; asserts the eef feature schema
(16 state + 7 quality + mirrored action) survives the dataset pipeline.
"""
from __future__ import annotations

import argparse
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import logging_mp

os.environ.setdefault("HF_LEROBOT_HOME", "/tmp/lerobot_home")
os.environ.setdefault(
    "HF_LEROBOT_CALIBRATION",
    str(Path(os.environ["HF_LEROBOT_HOME"]) / "calibration"),
)

import rclpy
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from rclpy.executors import MultiThreadedExecutor

from robodriver.dataset.dorobot_dataset import DoRobotDataset
from robodriver.robots.utils import busy_wait

from .config import (
    EEF_FEATURE_NAMES,
    QUALITY_FEATURE_NAMES,
    DeepcyboLiteUmiRos2RobotConfig,
)
from .mock_umi_topics import DeepcyboLiteUmiMockNode
from .robot import DeepcyboLiteUmiRos2Robot

logger = logging_mp.get_logger(__name__)


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("/tmp") / f"deepcybo_lite_umi_smoke_{stamp}"


def _build_dataset_features(robot: DeepcyboLiteUmiRos2Robot, use_videos: bool) -> dict:
    teleop_action_processor, _robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=use_videos,
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UMI-rig no-hardware recording smoke test (one episode)."
    )
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--drop-every", type=int, default=0,
                        help="inject LEFT tracking dropouts in the mock")
    parser.add_argument("--repo-id", default="deepcybo/lite-umi-ros2-smoke")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--task", default="DeepCybo Lite UMI rig smoke recording.")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=12)
    parser.add_argument("--connect-timeout-s", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _assert_expected_frame(observation: dict, action: dict) -> None:
    for name in EEF_FEATURE_NAMES:
        assert f"{name}.pos" in observation, f"missing obs {name}.pos"
        assert f"{name}.pos" in action, f"missing action {name}.pos"
        assert observation[f"{name}.pos"] == action[f"{name}.pos"], (
            f"action does not mirror observation for {name}"
        )
    for name in QUALITY_FEATURE_NAMES:
        assert f"{name}.flag" in observation, f"missing quality {name}.flag"
    for cam in ("image_head", "image_wrist_left", "image_wrist_right"):
        assert cam in observation, f"missing camera {cam}"
        assert observation[cam].shape == (480, 640, 3)


def run_smoke_record(args: argparse.Namespace) -> Path:
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_LEROBOT_CALIBRATION"]).mkdir(parents=True, exist_ok=True)

    output_root = args.root or _default_output_root()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset root already exists: {output_root}. "
                "Pass --overwrite or choose a new --root."
            )
        shutil.rmtree(output_root)

    rclpy.init()
    executor = MultiThreadedExecutor()
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    robot: DeepcyboLiteUmiRos2Robot | None = None
    mock_node: DeepcyboLiteUmiMockNode | None = None
    dataset: DoRobotDataset | None = None

    try:
        cfg = DeepcyboLiteUmiRos2RobotConfig()
        cfg.control_fps = int(args.fps)
        cfg.camera_fps = int(args.fps)
        cfg.use_videos = bool(args.use_videos)
        cfg.calibration_dir = (
            Path(os.environ["HF_LEROBOT_CALIBRATION"]) / "robots" / cfg.type
        )

        mock_node = DeepcyboLiteUmiMockNode(
            topics=cfg.ros2_topics,
            fps=float(args.fps),
            drop_every=args.drop_every,
        )
        robot = DeepcyboLiteUmiRos2Robot(cfg)

        executor.add_node(mock_node)
        executor.add_node(robot.get_node())
        spin_thread.start()

        connect_deadline = time.perf_counter() + args.connect_timeout_s
        while True:
            try:
                robot.connect()
                break
            except TimeoutError:
                raise
            except Exception:
                if time.perf_counter() >= connect_deadline:
                    raise
                time.sleep(0.1)

        features = _build_dataset_features(robot, args.use_videos)
        logger.info("Dataset features: %s", features)
        dataset = DoRobotDataset.create(
            args.repo_id,
            int(args.fps),
            root=output_root,
            robot=robot,
            features=features,
            use_videos=args.use_videos,
            use_audios=False,
            image_writer_processes=0,
            image_writer_threads=args.image_writer_threads,
        )

        frames_target = max(1, int(round(args.duration_s * args.fps)))
        logger.info("Recording %s frames at %s Hz to %s",
                    frames_target, args.fps, output_root)

        written = 0
        while written < frames_target:
            start_t = time.perf_counter()
            observation = robot.get_observation()
            action = robot.get_action()
            if written == 0:
                _assert_expected_frame(observation, action)
            frame = {
                **build_dataset_frame(dataset.features, observation, prefix=OBS_STR),
                **build_dataset_frame(dataset.features, action, prefix=ACTION),
                "task": args.task,
            }
            dataset.add_frame(frame)
            written += 1
            busy_wait(1.0 / args.fps - (time.perf_counter() - start_t))

        episode_index = dataset.save_episode()
        logger.info("Saved episode %s with %s frames at %s",
                    episode_index, written, output_root)
        return output_root
    finally:
        if dataset is not None:
            dataset.stop_audio_writer()
        if executor is not None:
            executor.shutdown()
        if spin_thread.is_alive():
            spin_thread.join(timeout=2.0)
        if robot is not None:
            try:
                if robot.is_connected:
                    robot.disconnect()
                else:
                    robot.get_node().destroy()
            except Exception:
                logger.exception("Failed to clean up robot node")
        if mock_node is not None:
            try:
                mock_node.destroy_node()
            except Exception:
                logger.exception("Failed to clean up mock node")
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: Iterable[str] | None = None) -> None:
    logging_mp.basic_config(level=logging_mp.INFO)
    args = build_arg_parser().parse_args(argv)
    output_root = run_smoke_record(args)
    print(output_root)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax gate** `[PURE]`

Run: `python -m py_compile robodriver_robot_deepcybo_lite_umi_ros2/smoke_record.py`
Expected: exit 0.

- [ ] **Step 3: Run the smoke on the ROS machine** `[ROS]`

```bash
cd robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
pip install -e .
python -m robodriver_robot_deepcybo_lite_umi_ros2.smoke_record --duration-s 5
# then with dropouts:
python -m robodriver_robot_deepcybo_lite_umi_ros2.smoke_record --duration-s 5 \
    --drop-every 60 --root /tmp/umi_smoke_drop --overwrite
```
Expected: both runs print a dataset root and exit 0; no assertion failures. Inspect the dropout run: `left_present.flag` must contain both 0.0 and 1.0 values across frames while `left_eef_*.pos` stays finite (held pose, no NaN).

- [ ] **Step 4: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add UMI adapter end-to-end smoke recording against the mock rig"
```

---

### Task 9: RViz live debug overlay

**Files:**
- Modify: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/node.py` (replace the two Task-5 stub methods `_init_debug_publishers` / `_publish_debug_pose`)

**Interfaces:**
- Consumes: `EefState` (Task 3), `publish_debug` config flag (Task 4).
- Produces: when `publish_debug=True` — `/umi/debug/eef_left`, `/umi/debug/eef_right` (`PoseStamped`, frame `world`) and `/umi/debug/markers` (`visualization_msgs/MarkerArray`, one sphere per arm, quality-colored). Pass criterion per spec §8: debug axes coincide with TF2's `gripper_<arm>` frames.

- [ ] **Step 1: Replace the stubs in `node.py`** `[PURE]`

Add to the imports at the top of `node.py`:

```python
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
```

Replace `_init_debug_publishers` and `_publish_debug_pose`:

```python
    def _init_debug_publishers(self) -> None:
        if not self.publish_debug:
            self._debug_pubs = None
            self._debug_marker_pub = None
            return
        t = self.topics
        self._debug_pubs = {
            "left": self.create_publisher(PoseStamped, "/umi/debug/eef_left", 10),
            "right": self.create_publisher(PoseStamped, "/umi/debug/eef_right", 10),
        }
        self._debug_marker_pub = self.create_publisher(
            MarkerArray, "/umi/debug/markers", 10
        )
        self.get_logger().info("debug overlay ON: /umi/debug/eef_* + /umi/debug/markers")

    @staticmethod
    def _quality_color(state) -> ColorRGBA:
        # green = fresh compose; yellow = held (tracking or world gap);
        # red = never composed / long stale
        if state.world_fresh >= 1.0 and state.present >= 1.0:
            return ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.9)
        if state.valid:
            return ColorRGBA(r=0.9, g=0.8, b=0.1, a=0.9)
        return ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.9)

    def _publish_debug_pose(self, arm, state, stamp) -> None:
        if self._debug_pubs is None or not state.valid:
            return
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = "world"
        p7 = state.pose7
        ps.pose.position.x = float(p7[0])
        ps.pose.position.y = float(p7[1])
        ps.pose.position.z = float(p7[2])
        ps.pose.orientation.x = float(p7[3])
        ps.pose.orientation.y = float(p7[4])
        ps.pose.orientation.z = float(p7[5])
        ps.pose.orientation.w = float(p7[6])
        self._debug_pubs[arm].publish(ps)

        marker = Marker()
        marker.header = ps.header
        marker.ns = "umi_eef"
        marker.id = 0 if arm == "left" else 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = ps.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.color = self._quality_color(state)
        arr = MarkerArray()
        arr.markers.append(marker)
        self._debug_marker_pub.publish(arr)
```

- [ ] **Step 2: Syntax gate** `[PURE]`

Run: `python -m py_compile robodriver_robot_deepcybo_lite_umi_ros2/node.py`
Expected: exit 0.

- [ ] **Step 3: Verify against the mock in RViz** `[ROS]`

```bash
# shell 1: mock with dropouts
python -m robodriver_robot_deepcybo_lite_umi_ros2.mock_umi_topics --drop-every 90
# shell 2: node alone with debug on
python -c "
import rclpy
from robodriver_robot_deepcybo_lite_umi_ros2.node import DeepcyboLiteUmiRos2RobotNode
rclpy.init()
n = DeepcyboLiteUmiRos2RobotNode(publish_debug=True)
rclpy.spin(n)
"
# shell 3
rviz2   # Fixed Frame: world; add Pose displays for /umi/debug/eef_left|right
        # and MarkerArray /umi/debug/markers
```
Expected: two pose axes moving on opposite sides of a circle; left marker flips green→yellow during injected dropouts (pose freezes) and back. **On the real rig** (later, with `collection.launch.py` up): the debug axes must coincide with the TF `gripper_left`/`gripper_right` frames — any offset is a composition bug.

- [ ] **Step 4: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add quality-colored RViz debug overlay for composed eef poses"
```

---

### Task 10: `visualize_episode.py` — post-hoc RViz episode viewer

**Files:**
- Create: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/visualize_episode.py`

**Interfaces:**
- Consumes: a recorded dataset (Task 8 output); `lerobot.datasets.lerobot_dataset.LeRobotDataset`; feature name lists (Task 4).
- Produces: `main()` console entry (`deepcybo-lite-umi-visualize-episode`) publishing `/umi/replay/path_left|right` (`nav_msgs/Path`), `/umi/replay/eef_left|right` (`PoseStamped`), `/umi/replay/markers` (`MarkerArray`, dimmed where quality flags were 0).

- [ ] **Step 1: Write the viewer** `[PURE]`

```python
# robodriver_robot_deepcybo_lite_umi_ros2/visualize_episode.py
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
```

- [ ] **Step 2: Syntax gate** `[PURE]`

Run: `python -m py_compile robodriver_robot_deepcybo_lite_umi_ros2/visualize_episode.py`
Expected: exit 0.

- [ ] **Step 3: Verify against the dropout smoke dataset** `[ROS]`

```bash
python -m robodriver_robot_deepcybo_lite_umi_ros2.visualize_episode \
    --root /tmp/umi_smoke_drop
# RViz: Fixed Frame `world`; add /umi/replay/path_left|right (Path),
# /umi/replay/eef_left|right (Pose), /umi/replay/markers (MarkerArray)
```
Expected: two circular trajectories on opposite phases; the left arm's path shows grey dimmed stretches (injected dropouts, frozen pose) while the right stays green; prints `replayed N frames`.

Note: if `ds.meta.features[OBS_STATE]["names"]` differs in this lerobot version (e.g. names nested under a different key), print `ds.meta.features[OBS_STATE]` once and adjust `_state_index` accordingly — the names list is the contract; the packing order comes from `observation_features` insertion order (16 state, then 7 quality).

- [ ] **Step 4: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add post-hoc RViz episode viewer with dropout dimming"
```

---

### Task 11: README + final spec cross-check

**Files:**
- Modify: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/README.md` (replace stub)

**Interfaces:**
- Consumes: everything above.
- Produces: the package's user-facing doc, including the spec-§10 quality-exclusion warning.

- [ ] **Step 1: Write the full README** `[PURE]`

```markdown
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

## send_action

This robot is record-only: `send_action()` raises `NotImplementedError`.
Deployment goes through joint-space replay (offline MoveIt2 IK → the
`deepcybo-lite-aio-ros2` adapter) or a future online IK bridge.
```

- [ ] **Step 2: Spec cross-check** `[PURE]`

Re-read `docs/superpowers/specs/2026-07-14-umi-eef-adapter-design.md` §3–§10 against the package. Verify: every §3 file exists (with the documented scripts→modules deviation), §4 topics match `config.py`, §5 names/order match `EEF_FEATURE_NAMES`/`QUALITY_FEATURE_NAMES`, §6 tolerance is 5 ms, §7 behaviors present, §8 all four test layers exist, §9 documented in README, §10 warning present in README. Fix any drift found; run the full pure suite once more:

Run: `python -m pytest tests/ -v`
Expected: 19 passed

- [ ] **Step 3: Commit** `[PURE]`

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Complete UMI adapter README with schema and quality-dim warning"
```

---

## Self-Review (done at plan time)

- **Spec coverage:** §3 layout → Tasks 1–10 (deviation documented in header); §4 topics → Task 4; §5 schema → Tasks 3, 4, 6; §6 composition → Tasks 2, 3, 5; §7 lifecycle → Task 6; §8 tests → Tasks 1–3 (unit), 7–8 (mock/smoke), 9 (live overlay), 10 (episode viewer); §9 env → Task 8 Step 3 + README; §10 risks → GripperTrack import guard (Task 5), schema round-trip assert (Task 8), quality-exclusion warning (Task 11).
- **Placeholder scan:** none — every code step carries full code; the two Task-5 debug stubs are explicitly replaced by Task 9.
- **Type consistency:** `EefState` fields, `EefComposer.update` signature, `state_vector()/quality_vector()` optionality, feature suffixes `.pos`/`.flag`, and node accessor names (`left_valid/right_valid/grippers_valid`) are used identically across Tasks 3, 5, 6, 8, 10.
```
