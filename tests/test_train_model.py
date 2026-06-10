import pandas as pd

from src.train_model import split_by_time


def test_split_by_time_uses_chronological_cutoff_and_metadata():
    dataset = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="min"),
            "target": [0, 1] * 5,
        }
    )

    train, test, metadata = split_by_time(dataset, train_fraction=0.7)

    assert len(train) == 7
    assert len(test) == 3
    assert train["timestamp"].max() < test["timestamp"].min()
    assert metadata["train_rows"] == 7
    assert metadata["test_rows"] == 3
    assert metadata["split_timestamp"] == str(train["timestamp"].max())
