import pandas as pd

from app.services.ml_dataset_sampler import _reservoir_sample_frames, load_representative_sample


def test_reservoir_sample_frames_caps_size():
    frames = [pd.DataFrame({"x": range(1000)})]
    out = _reservoir_sample_frames(frames, 50, random_state=42)
    assert len(out) == 50


def test_load_representative_sample_small_csv(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"a": range(20), "b": range(20, 40)}).to_csv(path, index=False)
    df, meta = load_representative_sample(path, filename="tiny.csv", file_format="csv", sample_rows=100)
    assert len(df) == 20
    assert meta["sampling_method"] in {"full_load", "single_chunk"}