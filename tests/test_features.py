import pandas as pd

from src.features import add_features, create_target


def sample_candles(rows: int = 260) -> pd.DataFrame:
    close = [100 + i * 0.1 for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "open": [price - 0.05 for price in close],
            "high": [price + 0.20 for price in close],
            "low": [price - 0.20 for price in close],
            "close": close,
            "volume": [100 + (i % 10) for i in range(rows)],
        }
    )


def test_add_features_creates_scalping_columns_without_all_nulls():
    featured = add_features(sample_candles())

    expected_columns = {
        "return_1",
        "return_5",
        "volatility_20",
        "rsi_14",
        "ema_20",
        "ema_50",
        "ema_200",
        "atr_14",
        "vwap_distance",
        "volume_spike_ratio",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
    }

    assert expected_columns.issubset(featured.columns)
    assert featured[list(expected_columns)].dropna().shape[0] > 0


def test_create_target_marks_take_profit_before_stop_loss():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="min"),
            "open": [100.0] * 20,
            "high": [100.0, 100.10, 100.25] + [100.0] * 17,
            "low": [100.0, 99.90, 99.90] + [100.0] * 17,
            "close": [100.0] * 20,
            "volume": [100.0] * 20,
        }
    )

    labeled = create_target(candles, horizon=15, take_profit=0.002, stop_loss=0.0015)

    assert labeled.loc[0, "target"] == 1


def test_create_target_marks_stop_loss_before_take_profit_as_zero():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="min"),
            "open": [100.0] * 20,
            "high": [100.0, 100.10, 100.25] + [100.0] * 17,
            "low": [100.0, 99.80, 99.90] + [100.0] * 17,
            "close": [100.0] * 20,
            "volume": [100.0] * 20,
        }
    )

    labeled = create_target(candles, horizon=15, take_profit=0.002, stop_loss=0.0015)

    assert labeled.loc[0, "target"] == 0
