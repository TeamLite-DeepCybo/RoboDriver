import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

HAS_PARQUET_DEPS = (
    importlib.util.find_spec("pyarrow") is not None
    and importlib.util.find_spec("PIL") is not None
)

pytestmark = pytest.mark.skipif(
    not HAS_PARQUET_DEPS,
    reason="pyarrow and Pillow are required for dataset image reference tests",
)


def test_image_features_are_declared_but_not_written_to_episode_parquet():
    from robodriver.utils.dataset import get_hf_features_from_features

    features = {
        "observation.images.image_head": {
            "dtype": "image",
            "shape": (4, 5, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left", "right"],
        },
    }

    hf_features = get_hf_features_from_features(features)

    assert "observation.images.image_head" not in hf_features
    assert "action" in hf_features


def test_dorobot_dataset_keeps_images_external_and_parquet_light(tmp_path):
    import pyarrow.parquet as pq

    from robodriver.dataset.dorobot_dataset import DoRobotDataset

    image_key = "observation.images.image_head"
    features = {
        image_key: {
            "dtype": "image",
            "shape": (4, 5, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left", "right"],
        },
    }
    dataset = DoRobotDataset.create(
        "image_frame_reference_probe",
        fps=30,
        root=tmp_path,
        robot=SimpleNamespace(microphones={}),
        robot_type="probe",
        features=features,
        use_videos=False,
        use_audios=False,
    )

    for frame_index in range(3):
        image = np.full((4, 5, 3), frame_index * 40, dtype=np.uint8)
        dataset.add_frame(
            {
                image_key: image,
                "action": np.array([frame_index, frame_index + 1], dtype=np.float32),
                "task": "probe task",
            }
        )
    episode_index = dataset.save_episode()

    parquet_path = tmp_path / "data/chunk-000/episode_000000.parquet"
    image_dir = tmp_path / f"images/{image_key}/episode_000000"
    table = pq.read_table(parquet_path)
    image_paths = sorted(image_dir.glob("frame_*.jpg"))

    assert episode_index == 0
    assert table.num_rows == 3
    assert image_key not in table.column_names
    assert [path.name for path in image_paths] == [
        "frame_000000.jpg",
        "frame_000001.jpg",
        "frame_000002.jpg",
    ]
    assert dataset.meta.info["image_path"] == (
        "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg"
    )
