from datetime import datetime

from robodriver.core.ros2_collection_metadata import (
    DEFAULT_RECORD_READY_TIMEOUT,
    build_collection_repo_id,
    build_collection_target_dir,
    build_lite_ros2_record_cmd,
    ensure_unique_ros2_record_dir,
    get_lite_record_ready_timeout,
    resolve_lite_collection_root,
)


def test_build_lite_ros2_record_cmd_defaults():
    now = datetime(2026, 6, 17, 10, 11, 12)

    cmd = build_lite_ros2_record_cmd(
        sequence=3,
        robot_name="lite-robot",
        now=now,
        env={},
    )

    assert cmd == {
        "task_id": "20260617",
        "task_name": "deepcybo_lite_bilateral",
        "task_data_id": "ros2_20260617_101112_0003",
        "machine_id": "lite-robot",
        "collector_id": "ros2_fsm",
        "source": "ros2_fsm",
        "countdown_seconds": 0,
        "created_at": "2026-06-17T10:11:12",
    }


def test_build_lite_ros2_record_cmd_env_overrides():
    now = datetime(2026, 6, 17, 10, 11, 12)
    env = {
        "DEEPCYBO_LITE_TASK_NAME": "fold_towel",
        "DEEPCYBO_LITE_TASK_ID": "task42",
        "DEEPCYBO_LITE_TASK_DATA_PREFIX": "trial",
        "DEEPCYBO_LITE_MACHINE_ID": "lite-a",
        "DEEPCYBO_LITE_COLLECTOR_ID": "operator-a",
    }

    cmd = build_lite_ros2_record_cmd(
        sequence=1,
        robot_name="ignored-robot",
        now=now,
        env=env,
    )

    assert cmd["task_name"] == "fold_towel"
    assert cmd["task_id"] == "task42"
    assert cmd["task_data_id"] == "trial_20260617_101112_0001"
    assert cmd["machine_id"] == "lite-a"
    assert cmd["collector_id"] == "operator-a"


def test_collection_path_and_repo_id_are_compatible():
    now = datetime(2026, 6, 17, 10, 11, 12)
    cmd = {
        "task_name": "fold_towel",
        "task_id": "task42",
        "task_data_id": "trial_0001",
    }

    assert build_collection_repo_id(cmd) == "fold_towel_task42_trial_0001"
    assert build_collection_target_dir("/data", cmd, now=now).as_posix() == (
        "/data/20260617/user/fold_towel_task42/fold_towel_task42_trial_0001"
    )


def test_ensure_unique_ros2_record_dir_avoids_existing_path(tmp_path):
    now = datetime(2026, 6, 17, 10, 11, 12)
    cmd = {
        "task_name": "fold_towel",
        "task_id": "task42",
        "task_data_id": "trial_0001",
    }
    existing = build_collection_target_dir(tmp_path, cmd, now=now)
    existing.mkdir(parents=True)

    repo_id, target_dir, unique_cmd = ensure_unique_ros2_record_dir(
        tmp_path,
        cmd,
        now=now,
    )

    assert unique_cmd["task_data_id"] == "trial_0001_retry01"
    assert repo_id == "fold_towel_task42_trial_0001_retry01"
    assert target_dir.name == "fold_towel_task42_trial_0001_retry01"
    assert cmd["task_data_id"] == "trial_0001"


def test_lite_root_and_timeout_env_parsing():
    assert resolve_lite_collection_root("/default", {}).as_posix() == "/default"
    assert resolve_lite_collection_root(
        "/default", {"DEEPCYBO_LITE_DATA_ROOT": "~/lite_data"}
    ).as_posix().endswith("/lite_data")

    assert get_lite_record_ready_timeout({}) == DEFAULT_RECORD_READY_TIMEOUT
    assert get_lite_record_ready_timeout(
        {"DEEPCYBO_LITE_RECORD_READY_TIMEOUT": "0.25"}
    ) == 0.25
    assert (
        get_lite_record_ready_timeout({"DEEPCYBO_LITE_RECORD_READY_TIMEOUT": "abc"})
        == DEFAULT_RECORD_READY_TIMEOUT
    )
    assert (
        get_lite_record_ready_timeout({"DEEPCYBO_LITE_RECORD_READY_TIMEOUT": "-1"})
        == DEFAULT_RECORD_READY_TIMEOUT
    )
