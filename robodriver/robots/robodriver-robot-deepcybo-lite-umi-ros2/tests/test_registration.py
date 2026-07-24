"""The adapter must self-register when only its top-level package is imported.

``robodriver.utils.import_utils.register_third_party_devices`` (used by
``robodriver.scripts.run``) discovers plugins by importing the top-level
package name and nothing deeper. So importing
``robodriver_robot_deepcybo_lite_umi_ros2`` alone must make
``deepcybo-lite-umi-ros2`` a valid ``--robot.type`` choice, exactly like the
aio adapter. A fresh interpreter keeps the global RobotConfig registry clean
so the assertion reflects this import and not some earlier one.
"""
import os
import subprocess
import sys
from pathlib import Path

PKG_PARENT = Path(__file__).resolve().parent.parent


def test_importing_top_level_package_registers_robot_type():
    code = (
        "import robodriver_robot_deepcybo_lite_umi_ros2\n"
        "from lerobot.robots import RobotConfig\n"
        "reg = RobotConfig._choice_registry\n"
        "assert 'deepcybo-lite-umi-ros2' in reg, sorted(reg)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(PKG_PARENT), os.environ.get("PYTHONPATH", "")]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
