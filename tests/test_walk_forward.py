import pandas as pd

from src.walk_forward import generate_walk_forward_folds


def test_generate_walk_forward_folds_rolls_chronologically():
    dataset = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="min"),
            "target": [0, 1] * 5,
        }
    )

    folds = generate_walk_forward_folds(dataset, train_size=4, test_size=2, step_size=2, max_folds=3)

    assert [fold.fold for fold in folds] == [1, 2, 3]
    assert list(folds[0].train["timestamp"]) == list(dataset.iloc[0:4]["timestamp"])
    assert list(folds[0].test["timestamp"]) == list(dataset.iloc[4:6]["timestamp"])
    assert list(folds[1].train["timestamp"]) == list(dataset.iloc[2:6]["timestamp"])
    assert list(folds[1].test["timestamp"]) == list(dataset.iloc[6:8]["timestamp"])
    assert folds[2].metadata["test_rows"] == 2


def test_generate_walk_forward_folds_returns_empty_when_not_enough_rows():
    dataset = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="min")})

    folds = generate_walk_forward_folds(dataset, train_size=4, test_size=2, step_size=1)

    assert folds == []
