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


@st.cache_data(show_spinner=False)
def cached_load_ohlcv(database_path: str, symbol: str, timeframe: str):
    return load_ohlcv(database_path, symbol, timeframe)


@st.cache_resource(show_spinner=False)
def cached_model_bundle(model_path: str):
    return joblib.load(PROJECT_ROOT / Path(model_path))


@st.cache_data(show_spinner=False)
def cached_backtest(candles, config, _bundle):
    return run_backtest(candles, _bundle, config)


@st.cache_data(show_spinner=False)
def cached_thresholds(candles, config, _bundle, start: float, stop: float, step: float):
    return optimize_thresholds(candles, _bundle, config, threshold_range(start, stop, step))


@st.cache_data(show_spinner=False)
def cached_diagnostics(candles, config, _bundle):
    return build_model_diagnostics(candles, _bundle, config)


@st.cache_data(show_spinner=False)
def cached_walk_forward(candles, config, train_size: int, test_size: int, max_folds: int):
    return run_walk_forward(candles, config, train_size=train_size, test_size=test_size, max_folds=max_folds)


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
threshold_min = st.sidebar.number_input("Threshold min", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
threshold_max = st.sidebar.number_input("Threshold max", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
threshold_step = st.sidebar.number_input("Threshold step", min_value=0.01, max_value=0.25, value=0.05, step=0.01)
walk_train_size = st.sidebar.number_input("WF train rows", min_value=100, max_value=100000, value=1000, step=100)
walk_test_size = st.sidebar.number_input("WF test rows", min_value=50, max_value=50000, value=300, step=50)
walk_max_folds = st.sidebar.number_input("WF max folds", min_value=1, max_value=50, value=3, step=1)

try:
    candles = cached_load_ohlcv(config["data"]["database_path"], symbol, timeframe)
except Exception as exc:
    st.error(f"No local candles found for {symbol} {timeframe}.")
    st.caption("Download that market/timeframe first, or switch back to the configured dataset.")
    st.code(f"python -m src.collect_data --symbol {symbol} --timeframe {timeframe}")
    st.exception(exc)
    st.stop()

bundle = cached_model_bundle(config["model"]["path"])
metadata = bundle.get("metadata", {})
signal = latest_signal(candles, config)

col1, col2, col3 = st.columns(3)
col1.metric("Latest signal", signal["signal"])
col2.metric("Model probability", f"{signal['probability']:.2%}")
col3.metric("Latest candle", str(signal["timestamp"]))

result = cached_backtest(candles, config, bundle)

st.subheader("Backtest stats")
stats_cols = st.columns(5)
stats_cols[0].metric("Trades", int(result.stats["trades"]))
stats_cols[1].metric("Win rate", f"{result.stats['win_rate']:.2%}")
stats_cols[2].metric("Return", f"{result.stats['total_return_pct']:.2f}%")
stats_cols[3].metric("Profit factor", f"{result.stats['profit_factor']:.2f}")
stats_cols[4].metric("Max drawdown", f"{result.stats['max_drawdown_pct']:.2f}%")
quality_cols = st.columns(4)
quality_cols[0].metric("Avg win", f"${result.stats['avg_win']:.2f}")
quality_cols[1].metric("Avg loss", f"${result.stats['avg_loss']:.2f}")
quality_cols[2].metric("Expectancy", f"${result.stats['expectancy']:.2f}")
quality_cols[3].metric("Max loss streak", int(result.stats["max_consecutive_losses"]))

train_candles, test_candles = split_candles_for_dashboard(candles, metadata)
if not train_candles.empty and not test_candles.empty:
    st.subheader("In-sample vs out-of-sample")
    train_result = cached_backtest(train_candles, config, bundle)
    test_result = cached_backtest(test_candles, config, bundle)
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
    thresholds = threshold_range(threshold_min, threshold_max, threshold_step)
    optimization = cached_thresholds(candles, config, bundle, threshold_min, threshold_max, threshold_step)
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
    walk_forward = cached_walk_forward(candles, config, int(walk_train_size), int(walk_test_size), int(walk_max_folds))
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
    diagnostics = cached_diagnostics(candles, config, bundle)
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
