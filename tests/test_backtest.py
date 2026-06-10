import numpy as np
import pandas as pd

from src.backtest import calculate_max_drawdown_pct, optimize_thresholds, summarize


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, frame):
        rows = len(frame)
        probs = self.probabilities[:rows]
        return np.array([[1 - probability, probability] for probability in probs])


def test_calculate_max_drawdown_pct_from_equity_curve():
    equity = pd.DataFrame({"equity": [100.0, 110.0, 104.5, 120.0, 90.0]})

    assert calculate_max_drawdown_pct(equity) == -25.0


def test_summarize_includes_max_drawdown_pct():
    trades = pd.DataFrame({"net_pnl": [10.0, -5.0], "equity": [110.0, 105.0]})

    stats = summarize(trades, initial_cash=100.0, final_cash=105.0)

    assert stats["max_drawdown_pct"] < 0


def test_optimize_thresholds_returns_one_row_per_threshold():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=260, freq="min"),
            "open": [100.0] * 260,
            "high": [100.3] * 260,
            "low": [99.9] * 260,
            "close": [100.0] * 260,
            "volume": [100.0] * 260,
        }
    )
    model_bundle = {"model": FixedProbabilityModel([0.7] * 260)}
    config = {
        "model": {"probability_threshold": 0.58},
        "target": {"horizon": 15, "take_profit": 0.002, "stop_loss": 0.0015},
        "backtest": {
            "initial_cash": 10000,
            "position_size": 1000,
            "fee_rate": 0.0004,
            "slippage_rate": 0.0002,
            "max_trades_per_day": 12,
        },
    }

    results = optimize_thresholds(candles, model_bundle, config, thresholds=[0.5, 0.8])

    assert list(results["threshold"]) == [0.5, 0.8]
    assert set(["trades", "win_rate", "total_return_pct", "profit_factor", "max_drawdown_pct"]).issubset(
        results.columns
    )
