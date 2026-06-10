from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.database import load_ohlcv
from src.features import FEATURE_COLUMNS, add_features
from src.utils import PROJECT_ROOT, load_config


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    stats: dict[str, float]


def run_backtest(candles: pd.DataFrame, model_bundle: dict, config: dict) -> BacktestResult:
    df = add_features(candles).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    model = model_bundle["model"]
    threshold = config["model"]["probability_threshold"]
    df["probability"] = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df["signal"] = df["probability"] >= threshold

    bt_cfg = config["backtest"]
    target_cfg = config["target"]
    cash = float(bt_cfg["initial_cash"])
    equity_rows = []
    trades = []
    trades_by_day: dict[pd.Timestamp, int] = {}

    index = 0
    while index < len(df) - target_cfg["horizon"]:
        row = df.iloc[index]
        day = pd.Timestamp(row["timestamp"]).date()
        trades_today = trades_by_day.get(day, 0)

        if not row["signal"] or trades_today >= bt_cfg["max_trades_per_day"]:
            equity_rows.append({"timestamp": row["timestamp"], "equity": cash})
            index += 1
            continue

        entry_price = row["close"] * (1 + bt_cfg["slippage_rate"])
        tp_price = entry_price * (1 + target_cfg["take_profit"])
        sl_price = entry_price * (1 - target_cfg["stop_loss"])
        size = bt_cfg["position_size"] / entry_price
        exit_price = df.iloc[index + target_cfg["horizon"]]["close"]
        exit_time = df.iloc[index + target_cfg["horizon"]]["timestamp"]
        reason = "timeout"
        exit_index = index + target_cfg["horizon"]

        for future_index in range(index + 1, index + 1 + target_cfg["horizon"]):
            candle = df.iloc[future_index]
            if candle["low"] <= sl_price:
                exit_price = sl_price * (1 - bt_cfg["slippage_rate"])
                exit_time = candle["timestamp"]
                exit_index = future_index
                reason = "stop_loss"
                break
            if candle["high"] >= tp_price:
                exit_price = tp_price * (1 - bt_cfg["slippage_rate"])
                exit_time = candle["timestamp"]
                exit_index = future_index
                reason = "take_profit"
                break

        gross_pnl = (exit_price - entry_price) * size
        fees = (entry_price * size + exit_price * size) * bt_cfg["fee_rate"]
        net_pnl = gross_pnl - fees
        cash += net_pnl
        trades_by_day[day] = trades_today + 1
        trades.append(
            {
                "entry_time": row["timestamp"],
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "probability": row["probability"],
                "reason": reason,
                "net_pnl": net_pnl,
                "equity": cash,
            }
        )
        equity_rows.append({"timestamp": exit_time, "equity": cash})
        index = max(exit_index + 1, index + 1)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).drop_duplicates("timestamp")
    stats = summarize(trades_df, bt_cfg["initial_cash"], cash, equity_df)
    return BacktestResult(trades_df, equity_df, stats)


def calculate_max_drawdown_pct(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty or "equity" not in equity_curve:
        return 0.0
    equity = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
    if equity.empty:
        return 0.0
    drawdown = (equity / equity.cummax() - 1) * 100
    return float(drawdown.min())


def summarize(
    trades: pd.DataFrame,
    initial_cash: float,
    final_cash: float,
    equity_curve: pd.DataFrame | None = None,
) -> dict[str, float]:
    if equity_curve is None:
        equity_curve = trades[["equity"]] if "equity" in trades else pd.DataFrame()
    max_drawdown_pct = calculate_max_drawdown_pct(equity_curve)

    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": max_drawdown_pct,
        }
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    gross_profit = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())
    return {
        "trades": float(len(trades)),
        "win_rate": float(len(wins) / len(trades)),
        "total_return_pct": float((final_cash / initial_cash - 1) * 100),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else float("inf"),
        "max_drawdown_pct": max_drawdown_pct,
    }


def threshold_range(start: float, stop: float, step: float) -> list[float]:
    thresholds = []
    current = start
    while current <= stop + 1e-12:
        thresholds.append(round(current, 4))
        current += step
    return thresholds


def optimize_thresholds(
    candles: pd.DataFrame,
    model_bundle: dict,
    config: dict,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        threshold_config = {
            **config,
            "model": {**config["model"], "probability_threshold": threshold},
        }
        result = run_backtest(candles, model_bundle, threshold_config)
        rows.append({"threshold": threshold, **result.stats})
    return pd.DataFrame(rows).sort_values(
        ["profit_factor", "total_return_pct", "trades"], ascending=[False, False, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paper-only signal backtest.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--optimize-thresholds", action="store_true")
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    args = parser.parse_args()

    config = load_config()
    data_cfg = config["data"]
    symbol = args.symbol or data_cfg["symbol"]
    timeframe = args.timeframe or data_cfg["timeframe"]
    candles = load_ohlcv(data_cfg["database_path"], symbol, timeframe)
    bundle = joblib.load(PROJECT_ROOT / Path(config["model"]["path"]))

    if args.optimize_thresholds:
        thresholds = threshold_range(args.threshold_min, args.threshold_max, args.threshold_step)
        results = optimize_thresholds(candles, bundle, config, thresholds)
        output = PROJECT_ROOT / "data" / "threshold_optimization.csv"
        results.to_csv(output, index=False)
        print(results.to_string(index=False))
        print(f"Saved threshold optimization to {output}")
        return

    result = run_backtest(candles, bundle, config)
    print(result.stats)
    output = PROJECT_ROOT / "data" / "latest_backtest_trades.csv"
    result.trades.to_csv(output, index=False)
    print(f"Saved trades to {output}")


if __name__ == "__main__":
    main()
