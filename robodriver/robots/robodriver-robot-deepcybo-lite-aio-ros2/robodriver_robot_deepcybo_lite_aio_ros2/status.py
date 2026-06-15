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


# 与 config.cameras / node.recv_images 的 key 一致
_CAMERA_HEAD = ("image_head", "头部相机", 640, 480)
_CAMERA_WRIST_L = ("image_wrist_left", "左腕相机", 640, 480)
_CAMERA_WRIST_R = ("image_wrist_right", "右腕相机", 640, 480)

# 与 node.recv_leader / recv_follower 的组件名一致（robot.update_status 用同名匹配）
_ARM_LEADER = "leader_arms"
_ARM_FOLLOWER = "follower_arms"


RobotStatus.register_subclass("deepcybo-lite-aio-ros2")


@dataclass
class DeepcyboLiteAioRos2RobotStatus(RobotStatus):
    device_name: str = "DeepCybo Lite"
    device_body: str = "DeepCybo"

    def __post_init__(self):
        self.specifications.end_type = "双臂 14 关节"
        self.specifications.fps = 30
        self.specifications.camera = CameraStatus(
            information=[
                CameraInfo(
                    name=_CAMERA_HEAD[0],
                    chinese_name=_CAMERA_HEAD[1],
                    type="RGB 相机",
                    width=_CAMERA_HEAD[2],
                    height=_CAMERA_HEAD[3],
                    is_connect=False,
                ),
                CameraInfo(
                    name=_CAMERA_WRIST_L[0],
                    chinese_name=_CAMERA_WRIST_L[1],
                    type="RGB 相机",
                    width=_CAMERA_WRIST_L[2],
                    height=_CAMERA_WRIST_L[3],
                    is_connect=False,
                ),
                CameraInfo(
                    name=_CAMERA_WRIST_R[0],
                    chinese_name=_CAMERA_WRIST_R[1],
                    type="RGB 相机",
                    width=_CAMERA_WRIST_R[2],
                    height=_CAMERA_WRIST_R[3],
                    is_connect=False,
                ),
            ]
        )

        self.specifications.arm = ArmStatus(
            information=[
                ArmInfo(
                    name=_ARM_LEADER,
                    type="双臂 14 关节（action / MITCommand 目标）",
                    start_pose=[],
                    joint_p_limit=[],
                    joint_n_limit=[],
                    is_connect=False,
                ),
                ArmInfo(
                    name=_ARM_FOLLOWER,
                    type="双臂 14 关节（observation / JointState 反馈）",
                    start_pose=[],
                    joint_p_limit=[],
                    joint_n_limit=[],
                    is_connect=False,
                ),
            ]
        )


# 兼容旧 import（Galaxea 拷贝遗留）
GALAXEALITEAIORos2RobotStatus = DeepcyboLiteAioRos2RobotStatus
