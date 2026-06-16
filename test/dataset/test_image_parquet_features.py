import json
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

HAS_PARQUET_DEPS = (
    importlib.util.find_spec("datasets") is not None
    and importlib.util.find_spec("pyarrow") is not None
    and importlib.util.find_spec("PIL") is not None
)

pytestmark = pytest.mark.skipif(
    not HAS_PARQUET_DEPS,
    reason="datasets, pyarrow, and Pillow are required for parquet image tests",
)


def test_image_features_are_included_in_hf_features():
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

    assert "observation.images.image_head" in hf_features
    assert hf_features["observation.images.image_head"]._type == "Image"
    assert "action" in hf_features


def test_dorobot_dataset_writes_image_columns_to_episode_parquet(tmp_path):
    import datasets
    import pyarrow.parquet as pq
    from PIL import Image

    from robodriver.dataset.dorobot_dataset import DoRobotDataset

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
    dataset = DoRobotDataset.create(
        "image_parquet_probe",
        fps=30,
        root=tmp_path,
        robot=SimpleNamespace(microphones={}),
        robot_type="probe",
        features=features,
        use_videos=False,
        use_audios=False,
    )

    image = np.full((4, 5, 3), 128, dtype=np.uint8)
    dataset.add_frame(
        {
            "observation.images.image_head": image,
            "action": np.array([1.0, 2.0], dtype=np.float32),
            "task": "probe task",
        }
    )
    episode_index = dataset.save_episode()

    parquet_path = tmp_path / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(parquet_path)
    row = table.to_pylist()[0]
    hf_meta = json.loads(table.schema.metadata[b"huggingface"])

    assert episode_index == 0
    assert "observation.images.image_head" in table.column_names
    assert (
        hf_meta["info"]["features"]["observation.images.image_head"]["_type"]
        == "Image"
    )
    assert row["observation.images.image_head"]["bytes"] is not None
    assert row["observation.images.image_head"]["path"] == "frame_000000.jpg"

    loaded = datasets.load_dataset(
        "parquet", data_files=str(parquet_path), split="train"
    )
    loaded_image = loaded[0]["observation.images.image_head"]
    assert isinstance(loaded_image, Image.Image)
    assert loaded_image.size == (5, 4)
