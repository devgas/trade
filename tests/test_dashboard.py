import pandas as pd

from src.dashboard_helpers import split_candles_for_dashboard


def test_split_candles_for_dashboard_uses_model_split_timestamp():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="min"),
            "close": [100, 101, 102, 103, 104],
        }
    )
    metadata = {"split_timestamp": "2024-01-01 00:02:00"}

    train, test = split_candles_for_dashboard(candles, metadata)

    assert list(train["close"]) == [100, 101, 102]
    assert list(test["close"]) == [103, 104]
