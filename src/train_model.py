from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.database import load_ohlcv
from src.features import FEATURE_COLUMNS, build_training_frame
from src.utils import PROJECT_ROOT, ensure_dirs, load_config


def split_by_time(dataset: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    ordered = dataset.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(ordered) * train_fraction)
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Not enough rows for requested train/test split")

    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()
    metadata = {
        "train_fraction": train_fraction,
        "split_timestamp": str(train["timestamp"].max()),
        "train_start": str(train["timestamp"].min()),
        "train_end": str(train["timestamp"].max()),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    return train, test, metadata


def train() -> None:
    parser = argparse.ArgumentParser(description="Train a LightGBM scalping classifier.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    ensure_dirs()
    config = load_config()
    data_cfg = config["data"]
    symbol = args.symbol or data_cfg["symbol"]
    timeframe = args.timeframe or data_cfg["timeframe"]

    candles = load_ohlcv(data_cfg["database_path"], symbol, timeframe)
    dataset = build_training_frame(candles, config["target"])
    if dataset.empty:
        raise RuntimeError("No training rows available after feature/target generation.")

    train_frame, test_frame, split_metadata = split_by_time(dataset, args.train_fraction)
    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["target"].astype(int)
    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["target"].astype(int)

    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
        force_col_wise=True,
        verbose=-1,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= config["model"]["probability_threshold"]).astype(int)
    print("Train/test split:")
    print(split_metadata)
    print(classification_report(y_test, predictions, zero_division=0))
    if y_test.nunique() > 1:
        print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.4f}")

    model_path = PROJECT_ROOT / Path(config["model"]["path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "config": config,
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                **split_metadata,
            },
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train()
