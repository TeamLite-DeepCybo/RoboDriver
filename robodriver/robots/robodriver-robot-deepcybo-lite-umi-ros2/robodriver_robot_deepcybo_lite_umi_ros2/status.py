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
