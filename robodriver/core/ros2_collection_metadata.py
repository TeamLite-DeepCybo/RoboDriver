import os
from datetime import datetime
from pathlib import Path
from typing import Mapping


DEFAULT_LITE_TASK_NAME = "deepcybo_lite_bilateral"
DEFAULT_LITE_TASK_DATA_PREFIX = "ros2"
DEFAULT_LITE_COLLECTOR_ID = "ros2_fsm"
DEFAULT_LITE_MACHINE_ID = "deepcybo-lite-aio-ros2"
DEFAULT_RECORD_READY_TIMEOUT = 15.0

# Robot types that drive data collection through the ROS2 FSM collection bridge
# (``/to_robodriver/start_collect`` etc.). These robots mount
# ``Ros2CollectionBridge`` in ``robodriver.scripts.run`` and keep collecting
# offline when RoboDriver-Server is unavailable.
ROS2_COLLECTION_ROBOT_TYPES = frozenset(
    {
        "deepcybo-lite-aio-ros2",
        "deepcybo-lite-umi-ros2",
    }
)


def uses_ros2_collection_bridge(robot_type: str) -> bool:
    """Whether ``robot_type`` records via the ROS2 FSM collection bridge."""
    return robot_type in ROS2_COLLECTION_ROBOT_TYPES


def _env_value(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    if value is None or value == "":
        return default
    return value


def env_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    value = env.get(name)
    if value is None or value == "":
        return default

    try:
        parsed = float(value)
    except ValueError:
        return default

    if minimum is not None and parsed < minimum:
        return default
    return parsed


def get_lite_record_ready_timeout(env: Mapping[str, str] | None = None) -> float:
    return env_float(
        os.environ if env is None else env,
        "DEEPCYBO_LITE_RECORD_READY_TIMEOUT",
        DEFAULT_RECORD_READY_TIMEOUT,
        minimum=0.0,
    )


def resolve_lite_collection_root(
    default_root: str | Path,
    env: Mapping[str, str] | None = None,
) -> Path:
    env = os.environ if env is None else env
    env_root = env.get("DEEPCYBO_LITE_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path(default_root).expanduser()


def build_lite_ros2_record_cmd(
    *,
    sequence: int,
    robot_name: str | None = None,
    now: datetime | None = None,
    env: Mapping[str, str] | None = None,
) -> dict:
    env = os.environ if env is None else env
    now = datetime.now() if now is None else now
    stamp = now.strftime("%Y%m%d_%H%M%S")

    task_name = _env_value(env, "DEEPCYBO_LITE_TASK_NAME", DEFAULT_LITE_TASK_NAME)
    task_id = _env_value(env, "DEEPCYBO_LITE_TASK_ID", now.strftime("%Y%m%d"))
    task_data_prefix = _env_value(
        env, "DEEPCYBO_LITE_TASK_DATA_PREFIX", DEFAULT_LITE_TASK_DATA_PREFIX
    )
    machine_id = _env_value(
        env,
        "DEEPCYBO_LITE_MACHINE_ID",
        robot_name or DEFAULT_LITE_MACHINE_ID,
    )

    return {
        "task_id": str(task_id),
        "task_name": str(task_name),
        "task_data_id": f"{task_data_prefix}_{stamp}_{sequence:04d}",
        "machine_id": str(machine_id),
        "collector_id": _env_value(
            env, "DEEPCYBO_LITE_COLLECTOR_ID", DEFAULT_LITE_COLLECTOR_ID
        ),
        "source": "ros2_fsm",
        "countdown_seconds": 0,
        "created_at": now.isoformat(timespec="seconds"),
    }


def build_collection_repo_id(record_cmd: Mapping[str, str]) -> str:
    return (
        f"{record_cmd.get('task_name')}_"
        f"{record_cmd.get('task_id')}_"
        f"{record_cmd.get('task_data_id')}"
    )


def build_collection_target_dir(
    dataset_path: str | Path,
    record_cmd: Mapping[str, str],
    *,
    now: datetime | None = None,
) -> Path:
    now = datetime.now() if now is None else now
    task_dir = f"{record_cmd.get('task_name')}_{record_cmd.get('task_id')}"
    return (
        Path(dataset_path)
        / now.strftime("%Y%m%d")
        / "user"
        / task_dir
        / build_collection_repo_id(record_cmd)
    )


def ensure_unique_ros2_record_dir(
    dataset_path: str | Path,
    record_cmd: Mapping[str, str],
    *,
    now: datetime | None = None,
) -> tuple[str, Path, dict]:
    unique_cmd = dict(record_cmd)
    base_task_data_id = unique_cmd["task_data_id"]
    target_dir = build_collection_target_dir(dataset_path, unique_cmd, now=now)
    collision_count = 0

    while target_dir.exists():
        collision_count += 1
        unique_cmd["task_data_id"] = f"{base_task_data_id}_retry{collision_count:02d}"
        target_dir = build_collection_target_dir(dataset_path, unique_cmd, now=now)

    return build_collection_repo_id(unique_cmd), target_dir, unique_cmd
