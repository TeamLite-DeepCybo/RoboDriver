"""TEMPORARY units patch: normalized gripper opening (0..1) <-> SI meters.

背景：当前 PI05 模型（pi05_ft_deepcybo_lite_ep150_*）的训练数据是夹爪
归一化开度（0..1）录制的；机器人在 lite_urdf SI 修复后，夹爪以米制
[0.0, 0.047] 表示（URDF prismatic upper limit）。

本补丁在部署链路中做临时换算，让现有模型无需重训即可继续部署：
  - observation.state[14:16]（米）  -> 0..1 归一化（发送给推理服务端）
  - action[14:16]（0..1 归一化）    -> 米（下发机器人）

TODO（后续统一修改 / unified change later）：训练数据已统一转换为 SI
米制并重训后，移除本补丁，训练与部署全链路统一使用 SI 米制，不再需要
任何换算。届时本模块与 --norm2si 相关开关一并删除。
"""

from __future__ import annotations

import numpy as np

# 夹爪通道在 canonical 16 维向量中的下标（左夹爪、右夹爪）。
GRIPPER_INDICES = (14, 15)
# 归一化开度 1.0 对应的 SI 米制行程。单一来源 = lite_urdf URDF prismatic
# limit upper=0.047（闭合=0、正方向=打开）。调用方应传入机器人从 URDF 派生
# 的限位；此常量仅作为未拿到 URDF 时的默认值。
GRIPPER_MAX_POSITION_M = 0.047


def si_to_normalized(state: np.ndarray,
                     gripper_max_m: float = GRIPPER_MAX_POSITION_M) -> np.ndarray:
    """米制 -> 0..1 归一化（仅夹爪通道），用于按归一化数据训练的模型。

    gripper_max_m：夹爪全开对应的 SI 行程（URDF prismatic upper），默认 0.047。
    """
    vec = np.asarray(state, dtype=np.float32).copy()
    if gripper_max_m <= 0.0:
        return vec
    for idx in GRIPPER_INDICES:
        if idx < vec.shape[0]:
            vec[idx] = float(
                np.clip(vec[idx] / gripper_max_m, 0.0, 1.0)
            )
    return vec


def normalized_to_si(action: np.ndarray,
                     gripper_max_m: float = GRIPPER_MAX_POSITION_M) -> np.ndarray:
    """0..1 归一化 -> 米制（仅夹爪通道），用于把模型 action 下发到机器人。

    gripper_max_m：夹爪全开对应的 SI 行程（URDF prismatic upper），默认 0.047。
    """
    vec = np.asarray(action, dtype=np.float32).copy()
    if gripper_max_m <= 0.0:
        return vec
    for idx in GRIPPER_INDICES:
        if idx < vec.shape[0]:
            vec[idx] = float(
                np.clip(vec[idx], 0.0, 1.0) * gripper_max_m
            )
    return vec
