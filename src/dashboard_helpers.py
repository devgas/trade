from __future__ import annotations

import pandas as pd


def split_candles_for_dashboard(candles: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_timestamp = metadata.get("split_timestamp")
    if not split_timestamp:
        return pd.DataFrame(), pd.DataFrame()
    split_at = pd.Timestamp(split_timestamp)
    train = candles[candles["timestamp"] <= split_at].copy()
    test = candles[candles["timestamp"] > split_at].copy()
    return train, test


def stats_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[
        [
            "sample",
            "period",
            "trades",
            "win_rate",
            "total_return_pct",
            "profit_factor",
            "max_drawdown_pct",
            "expectancy",
            "max_consecutive_losses",
        ]
    ]
