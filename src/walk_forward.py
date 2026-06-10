from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from src.backtest import run_backtest
from src.database import load_ohlcv
from src.features import FEATURE_COLUMNS, build_training_frame
from src.utils import PROJECT_ROOT, load_config


@dataclass
class WalkForwardFold:
    fold: int
    train: pd.DataFrame
    test: pd.DataFrame
    metadata: dict


def generate_walk_forward_folds(
    dataset: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    max_folds: int | None = None,
) -> list[WalkForwardFold]:
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if step_size is None:
        step_size = test_size
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    ordered = dataset.sort_values("timestamp").reset_index(drop=True)
    folds: list[WalkForwardFold] = []
    start = 0
    fold_number = 1

    while start + train_size + test_size <= len(ordered):
        train = ordered.iloc[start : start + train_size].copy()
        test = ordered.iloc[start + train_size : start + train_size + test_size].copy()
        metadata = {
            "fold": fold_number,
            "train_start": str(train["timestamp"].min()),
            "train_end": str(train["timestamp"].max()),
            "test_start": str(test["timestamp"].min()),
            "test_end": str(test["timestamp"].max()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
        folds.append(WalkForwardFold(fold=fold_number, train=train, test=test, metadata=metadata))
        if max_folds is not None and len(folds) >= max_folds:
            break
        start += step_size
        fold_number += 1

    return folds


def train_fold_model(train_frame: pd.DataFrame) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=100,
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
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["target"].astype(int))
    return model


def run_walk_forward(
    candles: pd.DataFrame,
    config: dict,
    train_size: int = 1200,
    test_size: int = 400,
    step_size: int | None = None,
    max_folds: int | None = 5,
) -> pd.DataFrame:
    dataset = build_training_frame(candles, config["target"])
    folds = generate_walk_forward_folds(dataset, train_size, test_size, step_size, max_folds)
    rows = []

    for fold in folds:
        model = train_fold_model(fold.train)
        model_bundle = {"model": model, "features": FEATURE_COLUMNS, "config": config}
        test_start = pd.Timestamp(fold.metadata["test_start"])
        test_end = pd.Timestamp(fold.metadata["test_end"])
        test_candles = candles[(candles["timestamp"] >= test_start) & (candles["timestamp"] <= test_end)].copy()
        result = run_backtest(test_candles, model_bundle, config)
        rows.append({**fold.metadata, **result.stats})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward paper validation.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    data_cfg = config["data"]
    symbol = args.symbol or data_cfg["symbol"]
    timeframe = args.timeframe or data_cfg["timeframe"]
    candles = load_ohlcv(data_cfg["database_path"], symbol, timeframe)
    results = run_walk_forward(
        candles,
        config,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        max_folds=args.max_folds,
    )

    output = PROJECT_ROOT / "data" / "walk_forward_results.csv"
    results.to_csv(output, index=False)
    print(results.to_string(index=False) if not results.empty else "No walk-forward folds produced.")
    print(f"Saved walk-forward results to {output}")


if __name__ == "__main__":
    main()
