"""单一来源：从 lite_urdf URDF 读取夹爪 prismatic 限位（closed=0, positive=open）。

背景：夹爪开度上限（0.047）此前散落在多处硬编码。统一以 URDF
`<joint name=left_gripper|right_gripper type=prismatic><limit lower upper>` 为唯一来源。
本模块提供解析；调用方传入 URDF（文件路径或字符串），取不到时返回约定的
URDF 默认值 (0.0, 0.047)，并记录来源。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# 与 lite_urdf limit upper=0.047 一致的默认（仅当无法读取 URDF 时兜底）。
DEFAULT_GRIPPER_MIN_M = 0.0
DEFAULT_GRIPPER_MAX_M = 0.047

# 从 URDF 中识别的夹爪 joint 名集合。
GRIPPER_JOINT_NAMES = frozenset(("left_gripper", "right_gripper"))


def read_gripper_limits(urdf: str) -> tuple[float, float]:
    """解析 URDF 字符串，返回 (min, max) 夹爪 prismatic 限位。

    找不到/解析失败时返回 DEFAULT_GRIPPER_MIN_M/MAX_M（与 lite_urdf 一致）。
    """
    lo, hi = DEFAULT_GRIPPER_MIN_M, DEFAULT_GRIPPER_MAX_M
    try:
        root = ET.fromstring(urdf)
    except ET.ParseError:
        return lo, hi
    for joint in root.iter("joint"):
        if joint.get("name") not in GRIPPER_JOINT_NAMES:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower, upper = limit.get("lower"), limit.get("upper")
        if lower is not None and upper is not None:
            try:
                lo = float(lower)
                hi = float(upper)
            except (TypeError, ValueError):
                continue
    return lo, hi


def read_gripper_limits_from_path(path: str) -> tuple[float, float]:
    """从 URDF 文件路径读取夹爪限位。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return read_gripper_limits(fh.read())
    except OSError:
        return DEFAULT_GRIPPER_MIN_M, DEFAULT_GRIPPER_MAX_M
