from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS, add_features
from src.utils import PROJECT_ROOT, load_config


def latest_feature_frame(latest: pd.Series) -> pd.DataFrame:
    """Build a one-row numeric frame for LightGBM prediction."""
    frame = pd.DataFrame([{column: latest[column] for column in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    return frame.apply(pd.to_numeric, errors="coerce")


def latest_signal(candles: pd.DataFrame, config: dict | None = None) -> dict:
    config = config or load_config()
    bundle = joblib.load(PROJECT_ROOT / Path(config["model"]["path"]))
    model = bundle["model"]
    featured = add_features(candles).dropna(subset=FEATURE_COLUMNS)
    if featured.empty:
        raise RuntimeError("Not enough candles to build latest feature row.")

    latest = featured.iloc[-1]
    feature_frame = latest_feature_frame(latest)
    probability = float(model.predict_proba(feature_frame)[0, 1])
    threshold = config["model"]["probability_threshold"]
    if probability >= threshold:
        signal = "bullish"
    elif probability <= 1 - threshold:
        signal = "bearish"
    else:
        signal = "no-trade"

    return {
        "timestamp": latest["timestamp"],
        "symbol": config["data"]["symbol"],
        "signal": signal,
        "probability": probability,
        "features": feature_frame.iloc[0].to_dict(),
    }
