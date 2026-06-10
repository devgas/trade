from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
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
]


def add_features(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy().sort_values("timestamp").reset_index(drop=True)
    df["return_1"] = df["close"].pct_change()
    df["return_5"] = df["close"].pct_change(5)
    df["volatility_20"] = df["return_1"].rolling(20).std()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)

    for period in (20, 50, 200):
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    rolling_pv = (typical_price * df["volume"]).rolling(20).sum()
    rolling_volume = df["volume"].rolling(20).sum()
    vwap_20 = rolling_pv / rolling_volume.replace(0, np.nan)
    df["vwap_distance"] = (df["close"] - vwap_20) / vwap_20

    df["volume_spike_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    df["body_ratio"] = body / candle_range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range

    return df


def create_target(
    candles: pd.DataFrame,
    horizon: int = 15,
    take_profit: float = 0.002,
    stop_loss: float = 0.0015,
) -> pd.DataFrame:
    df = candles.copy().sort_values("timestamp").reset_index(drop=True)
    targets: list[float] = []

    for index, row in df.iterrows():
        entry = row["close"]
        tp_price = entry * (1 + take_profit)
        sl_price = entry * (1 - stop_loss)
        future = df.iloc[index + 1 : index + 1 + horizon]

        if len(future) < horizon:
            targets.append(np.nan)
            continue

        outcome = 0
        for _, candle in future.iterrows():
            hit_tp = candle["high"] >= tp_price
            hit_sl = candle["low"] <= sl_price
            if hit_tp and hit_sl:
                outcome = 0
                break
            if hit_tp:
                outcome = 1
                break
            if hit_sl:
                outcome = 0
                break
        targets.append(outcome)

    df["target"] = targets
    return df


def build_training_frame(candles: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    featured = add_features(candles)
    labeled = create_target(featured, **target_config)
    return labeled.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
