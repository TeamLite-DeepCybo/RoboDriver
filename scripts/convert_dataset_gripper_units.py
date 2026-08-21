#!/usr/bin/env python3
"""Convert a DoRobot dataset's gripper channels from normalized (0..1) to SI meters.

背景：lite_urdf SI 修复前录制的 DeepCybo Lite 数据集，夹爪通道为归一化
开度（0..1）；当前部署管线的机器人接口为米制（0..0.047）。本脚本把
action / observation.state 的夹爪通道乘以 0.047 另存为新数据集，并同步
更新 meta 中的统计文件，使新数据与当前部署/训练管线量纲完全一致。

用法:
    python scripts/convert_dataset_gripper_units.py \
      --input-dir  /path/to/deepcybo_lite_bilateral_gear \
      --output-dir /path/to/deepcybo_lite_bilateral_gear_si

只改写 data/ 下的 episode parquet（保留内嵌图片字节）与三个 meta 统计
文件（stats.json / episodes_stats.jsonl / relative_stats_dreamzero.json），
其余 meta 与 videos/ 原样复制。线性缩放对 min/max/mean/std/q01/q99 均
精确成立，无需重算。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GRIPPER_INDICES = (14, 15)
STAT_FIELDS = ("mean", "std", "min", "max", "q01", "q99")
EPISODE_STAT_FIELDS = ("min", "max", "mean", "std")


def _scale_list(values: Iterable[float], scale: float) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float32).copy()
    for idx in GRIPPER_INDICES:
        if idx < arr.shape[0]:
            arr[idx] = arr[idx] * scale
    return arr.tolist()


def _scale_stats_entry(entry: dict, scale: float) -> None:
    """就地缩放 stats.json / episodes_stats 中 16 维数组的夹爪通道。"""
    for stat in STAT_FIELDS:
        if stat in entry:
            entry[stat] = _scale_list(entry[stat], scale)


def _scale_episode_stats(stats: dict, scale: float) -> None:
    for key in ("action", "observation.state"):
        entry = stats.get(key)
        if not isinstance(entry, dict):
            continue
        for stat in EPISODE_STAT_FIELDS:
            if stat in entry:
                entry[stat] = _scale_list(entry[stat], scale)


def _scale_relative_stats(relative: dict, scale: float) -> None:
    """缩放 relative_stats_dreamzero.json 中的夹爪统计（1 维数组）。"""
    for key in ("left_gripper_pos", "right_gripper_pos"):
        entry = relative.get(key)
        if not isinstance(entry, dict):
            continue
        for stat, value in entry.items():
            if isinstance(value, list):
                entry[stat] = [float(v) * scale for v in value]
            elif isinstance(value, (int, float)):
                entry[stat] = float(value) * scale


def _convert_parquet(src: Path, dst: Path, scale: float) -> None:
    table = pq.read_table(src)
    for col_name in ("action", "observation.state"):
        field_index = table.schema.get_field_index(col_name)
        if field_index < 0:
            continue
        col = table.column(col_name)
        values = np.asarray(col.to_pylist(), dtype=np.float32)
        for idx in GRIPPER_INDICES:
            if idx < values.shape[1]:
                values[:, idx] = values[:, idx] * scale
        flat = pa.array(values.reshape(-1), type=col.type.value_type)
        new_col = pa.FixedSizeListArray.from_arrays(
            flat, list_size=col.type.list_size
        )
        table = table.set_column(field_index, col_name, new_col)
    table = table.replace_schema_metadata(table.schema.metadata)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)


def _verify_dataset(root: Path, scale: float) -> dict:
    """扫描转换后所有 episode，校验夹爪通道量纲并返回全局 min/max。"""
    gmin = {"action": np.inf, "state": np.inf}
    gmax = {"action": -np.inf, "state": -np.inf}
    count = 0
    for parquet in sorted((root / "data").rglob("episode_*.parquet")):
        table = pq.read_table(parquet, columns=["action", "observation.state"])
        for label, col_name in (("action", "action"), ("state", "observation.state")):
            arr = np.asarray(
                table.column(col_name).to_pylist(), dtype=np.float32
            )
            for idx in GRIPPER_INDICES:
                gmin[label] = min(gmin[label], float(arr[:, idx].min()))
                gmax[label] = max(gmax[label], float(arr[:, idx].max()))
        count += 1
    return {"episodes": count, "gripper_min": gmin, "gripper_max": gmax}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DoRobot dataset gripper channels normalized(0..1) -> SI "
            "meters (x0.047) and save as a new dataset."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--gripper-max-m",
        type=float,
        default=0.047,
        help="归一化 1.0 对应的米制行程（默认 0.047）",
    )
    parser.add_argument(
        "--force", action="store_true", help="输出目录已存在时覆盖"
    )
    args = parser.parse_args()

    src: Path = args.input_dir
    dst: Path = args.output_dir
    scale = args.gripper_max_m
    if scale <= 0:
        print(f"错误：--gripper-max-m 必须为正数，got {scale}", file=sys.stderr)
        return 2
    if not (src / "meta" / "info.json").exists():
        print(f"错误：{src} 不是 DoRobot 数据集（缺少 meta/info.json）", file=sys.stderr)
        return 2
    if dst.exists() and not args.force:
        print(f"错误：输出目录已存在：{dst}（加 --force 覆盖）", file=sys.stderr)
        return 2
    if dst == src:
        print("错误：--output-dir 不能与 --input-dir 相同", file=sys.stderr)
        return 2

    print(f"[1/5] 复制 videos/ 与 meta/ 骨架 -> {dst}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ("videos", "meta"):
        if (src / name).exists():
            shutil.copytree(src / name, dst / name)

    print("[2/5] 转换 episode parquet（保留内嵌图片字节）...")
    episodes = sorted((src / "data").rglob("episode_*.parquet"))
    for i, parquet in enumerate(episodes, 1):
        rel = parquet.relative_to(src / "data")
        _convert_parquet(parquet, dst / "data" / rel, scale)
        if i % 25 == 0 or i == len(episodes):
            print(f"      {i}/{len(episodes)}")

    print("[3/5] 更新 meta/stats.json")
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for key in ("observation.state", "action"):
        if key in stats:
            _scale_stats_entry(stats[key], scale)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    print("[4/5] 更新 meta/episodes_stats.jsonl 与 relative_stats_dreamzero.json")
    eps_path = dst / "meta" / "episodes_stats.jsonl"
    lines_out = []
    for line in eps_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record.get("stats"), dict):
            _scale_episode_stats(record["stats"], scale)
        lines_out.append(json.dumps(record, ensure_ascii=False))
    eps_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")

    rel_path = dst / "meta" / "relative_stats_dreamzero.json"
    if rel_path.exists():
        relative = json.loads(rel_path.read_text(encoding="utf-8"))
        _scale_relative_stats(relative, scale)
        rel_path.write_text(
            json.dumps(relative, ensure_ascii=False), encoding="utf-8"
        )

    note = dst / "CONVERSION_NOTES.md"
    note.write_text(
        "\n".join(
            [
                "# Dataset conversion notes / 数据转换说明",
                "",
                f"- Source: {src}",
                "- 转换：夹爪通道（idx 14/15）× 0.047，归一化开度 0..1 -> SI 米制 0..0.047",
                "- 同步更新：stats.json、episodes_stats.jsonl、relative_stats_dreamzero.json",
                "- 由 scripts/convert_dataset_gripper_units.py 生成",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("[5/5] 校验转换后夹爪量纲 ...")
    result = _verify_dataset(dst, scale)
    ok = True
    for label in ("action", "state"):
        lo, hi = result["gripper_min"][label], result["gripper_max"][label]
        valid = 0.0 <= lo and hi <= scale * 1.001 + 1e-6
        ok = ok and valid
        print(
            f"      {label} gripper range: [{lo:.6f}, {hi:.6f}] "
            f"{'OK' if valid else 'OUT OF RANGE'}"
        )
    print(f"      已转换 {result['episodes']} 个 episode")
    print(f"完成：{dst}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
