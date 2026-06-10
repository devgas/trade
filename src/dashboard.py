from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import plotly.express as px
import streamlit as st

from src.backtest import optimize_thresholds, run_backtest, threshold_range
from src.database import load_ohlcv
from src.llm_explain import explain_signal
from src.model_diagnostics import build_model_diagnostics
from src.signal_engine import latest_signal
from src.utils import PROJECT_ROOT, load_config
from src.walk_forward import run_walk_forward

from src.dashboard_helpers import split_candles_for_dashboard, stats_table


st.set_page_config(page_title="AI Scalping Research", layout="wide")
st.title("AI Scalping Research Dashboard")
st.caption("Paper trading research only. No real order execution is implemented.")

config = load_config()
symbol = st.sidebar.text_input("Symbol", config["data"]["symbol"])
timeframe = st.sidebar.text_input("Timeframe", config["data"]["timeframe"])
run_llm = st.sidebar.checkbox("Ask Ollama for explanation", value=True)
show_thresholds = st.sidebar.checkbox("Run threshold sweep", value=True)
show_walk_forward = st.sidebar.checkbox("Run walk-forward validation", value=False)
show_diagnostics = st.sidebar.checkbox("Show model diagnostics", value=True)

try:
    candles = load_ohlcv(config["data"]["database_path"], symbol, timeframe)
except Exception as exc:
    st.error(f"No local candles found for {symbol} {timeframe}.")
    st.caption("Download that market/timeframe first, or switch back to the configured dataset.")
    st.code(f"python -m src.collect_data --symbol {symbol} --timeframe {timeframe}")
    st.exception(exc)
    st.stop()

bundle = joblib.load(PROJECT_ROOT / Path(config["model"]["path"]))
metadata = bundle.get("metadata", {})
signal = latest_signal(candles, config)

col1, col2, col3 = st.columns(3)
col1.metric("Latest signal", signal["signal"])
col2.metric("Model probability", f"{signal['probability']:.2%}")
col3.metric("Latest candle", str(signal["timestamp"]))

result = run_backtest(candles, bundle, config)

st.subheader("Backtest stats")
stats_cols = st.columns(5)
stats_cols[0].metric("Trades", int(result.stats["trades"]))
stats_cols[1].metric("Win rate", f"{result.stats['win_rate']:.2%}")
stats_cols[2].metric("Return", f"{result.stats['total_return_pct']:.2f}%")
stats_cols[3].metric("Profit factor", f"{result.stats['profit_factor']:.2f}")
stats_cols[4].metric("Max drawdown", f"{result.stats['max_drawdown_pct']:.2f}%")

train_candles, test_candles = split_candles_for_dashboard(candles, metadata)
if not train_candles.empty and not test_candles.empty:
    st.subheader("In-sample vs out-of-sample")
    train_result = run_backtest(train_candles, bundle, config)
    test_result = run_backtest(test_candles, bundle, config)
    comparison = stats_table(
        [
            {
                "sample": "In-sample",
                "period": f"{metadata.get('train_start')} to {metadata.get('train_end')}",
                **train_result.stats,
            },
            {
                "sample": "Out-of-sample",
                "period": f"{metadata.get('test_start')} to {metadata.get('test_end')}",
                **test_result.stats,
            },
        ]
    )
    st.dataframe(comparison, width="stretch", hide_index=True)
else:
    st.info("Retrain the model to add train/test split metadata for in-sample vs out-of-sample stats.")

st.subheader("Equity curve")
if not result.equity_curve.empty:
    st.plotly_chart(px.line(result.equity_curve, x="timestamp", y="equity"), width="stretch")
else:
    st.info("No trades produced by the current threshold and backtest rules.")

if show_thresholds:
    st.subheader("Threshold sweep")
    thresholds = threshold_range(0.50, 0.80, 0.05)
    optimization = optimize_thresholds(candles, bundle, config, thresholds)
    st.dataframe(optimization, width="stretch", hide_index=True)
    st.plotly_chart(
        px.line(
            optimization.sort_values("threshold"),
            x="threshold",
            y=["profit_factor", "total_return_pct", "max_drawdown_pct"],
            markers=True,
        ),
        width="stretch",
    )


if show_walk_forward:
    st.subheader("Walk-forward validation")
    st.caption("Trains a fresh model per fold and backtests only the next test window.")
    walk_forward = run_walk_forward(candles, config, train_size=1000, test_size=300, max_folds=3)
    if walk_forward.empty:
        st.info("Not enough data for walk-forward validation with the current settings.")
    else:
        st.dataframe(walk_forward, width="stretch", hide_index=True)
        st.plotly_chart(
            px.bar(
                walk_forward,
                x="fold",
                y=["total_return_pct", "max_drawdown_pct"],
                barmode="group",
            ),
            width="stretch",
        )


if show_diagnostics:
    st.subheader("Model diagnostics")
    diagnostics = build_model_diagnostics(candles, bundle, config)
    importance = diagnostics["feature_importance"]
    buckets = diagnostics["probability_buckets"]
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        st.caption("Feature importance")
        st.dataframe(importance, width="stretch", hide_index=True)
        if not importance.empty:
            st.plotly_chart(
                px.bar(importance.head(12), x="importance", y="feature", orientation="h"),
                width="stretch",
            )
    with diag_col2:
        st.caption("Out-of-sample probability buckets")
        st.dataframe(buckets, width="stretch", hide_index=True)
        if not buckets.empty:
            st.plotly_chart(
                px.bar(buckets, x="bucket", y=["avg_probability", "target_rate"], barmode="group"),
                width="stretch",
            )

st.subheader("Model split metadata")
if metadata:
    st.json(metadata)
else:
    st.write("No split metadata found in the saved model.")

st.subheader("LLM explanation")
if run_llm:
    st.write(explain_signal(signal, config))
else:
    st.write("LLM explanation disabled in the dashboard.")

with st.expander("Latest feature values"):
    st.json(signal["features"])
