from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.database import load_ohlcv
from src.features import FEATURE_COLUMNS, build_training_frame
from src.utils import PROJECT_ROOT, ensure_dirs, load_config


def train() -> None:
    parser = argparse.ArgumentParser(description="Train a LightGBM scalping classifier.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
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

    x = dataset[FEATURE_COLUMNS]
    y = dataset["target"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= config["model"]["probability_threshold"]).astype(int)
    print(classification_report(y_test, predictions, zero_division=0))
    if y_test.nunique() > 1:
        print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.4f}")

    model_path = PROJECT_ROOT / Path(config["model"]["path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "config": config}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train()
