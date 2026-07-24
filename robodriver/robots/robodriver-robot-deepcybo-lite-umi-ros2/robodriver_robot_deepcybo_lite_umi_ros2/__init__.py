__version__ = "0.1.0"

# Importing .config runs the @RobotConfig.register_subclass decorator, so
# importing just this top-level package registers "deepcybo-lite-umi-ros2" as a
# valid robot type. register_third_party_devices() (robodriver.scripts.run)
# imports plugins by top-level name only, so this import is what makes the
# adapter discoverable. .config is rclpy-free, keeping the pure numpy tests
# (compose/se3) importable without a sourced ROS 2 environment.
from .config import (  # noqa: F401
    DeepcyboLiteUmiRos2RobotConfig,
    DeepcyboLiteUmiRos2Topics,
)

__all__ = ["DeepcyboLiteUmiRos2RobotConfig", "DeepcyboLiteUmiRos2Topics"]


def __getattr__(name: str):
    # Expose the Robot class lazily so make_device_from_device_class() can find
    # it at runtime, without pulling rclpy in at package-import time.
    if name == "DeepcyboLiteUmiRos2Robot":
        from .robot import DeepcyboLiteUmiRos2Robot

        return DeepcyboLiteUmiRos2Robot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
