from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.database import load_ohlcv
from src.features import FEATURE_COLUMNS, build_training_frame
from src.utils import PROJECT_ROOT, load_config


def feature_importance_frame(model, feature_columns: list[str]) -> pd.DataFrame:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    frame = pd.DataFrame({"feature": feature_columns, "importance": importances})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def probability_bucket_performance(
    model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    bins: list[float] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "samples", "avg_probability", "target_rate"])
    bins = bins or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    scored = frame.copy()
    scored["probability"] = model.predict_proba(scored[feature_columns])[:, 1]
    scored["bucket"] = pd.cut(scored["probability"], bins=bins, include_lowest=True, right=True)

    rows = []
    for bucket, group in scored.groupby("bucket", observed=False):
        if group.empty:
            continue
        rows.append(
            {
                "bucket": f"{max(float(bucket.left), 0.0):.1f}-{float(bucket.right):.1f}",
                "samples": int(len(group)),
                "avg_probability": float(group["probability"].mean()),
                "target_rate": float(group["target"].mean()),
            }
        )
    return pd.DataFrame(rows)


def diagnostic_dataset(candles: pd.DataFrame, config: dict, metadata: dict) -> pd.DataFrame:
    dataset = build_training_frame(candles, config["target"])
    split_timestamp = metadata.get("split_timestamp")
    if split_timestamp:
        dataset = dataset[dataset["timestamp"] > pd.Timestamp(split_timestamp)]
    return dataset.reset_index(drop=True)


def build_model_diagnostics(candles: pd.DataFrame, model_bundle: dict, config: dict) -> dict[str, pd.DataFrame]:
    model = model_bundle["model"]
    feature_columns = model_bundle.get("features", FEATURE_COLUMNS)
    metadata = model_bundle.get("metadata", {})
    diagnostics_frame = diagnostic_dataset(candles, config, metadata)
    return {
        "feature_importance": feature_importance_frame(model, feature_columns),
        "probability_buckets": probability_bucket_performance(model, diagnostics_frame, feature_columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model diagnostic CSV files.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    args = parser.parse_args()

    config = load_config()
    data_cfg = config["data"]
    symbol = args.symbol or data_cfg["symbol"]
    timeframe = args.timeframe or data_cfg["timeframe"]
    candles = load_ohlcv(data_cfg["database_path"], symbol, timeframe)
    bundle = joblib.load(PROJECT_ROOT / Path(config["model"]["path"]))
    diagnostics = build_model_diagnostics(candles, bundle, config)

    importance_path = PROJECT_ROOT / "data" / "feature_importance.csv"
    buckets_path = PROJECT_ROOT / "data" / "probability_buckets.csv"
    diagnostics["feature_importance"].to_csv(importance_path, index=False)
    diagnostics["probability_buckets"].to_csv(buckets_path, index=False)
    print(diagnostics["feature_importance"].to_string(index=False))
    print(diagnostics["probability_buckets"].to_string(index=False))
    print(f"Saved diagnostics to {importance_path} and {buckets_path}")


if __name__ == "__main__":
    main()
