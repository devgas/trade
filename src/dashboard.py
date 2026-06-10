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
from src.signal_engine import latest_signal
from src.utils import PROJECT_ROOT, load_config


st.set_page_config(page_title="AI Scalping Research", layout="wide")
st.title("AI Scalping Research Dashboard")
st.caption("Paper trading research only. No real order execution is implemented.")

config = load_config()
symbol = st.sidebar.text_input("Symbol", config["data"]["symbol"])
timeframe = st.sidebar.text_input("Timeframe", config["data"]["timeframe"])
run_llm = st.sidebar.checkbox("Ask Ollama for explanation", value=True)
show_thresholds = st.sidebar.checkbox("Run threshold sweep", value=True)

try:
    candles = load_ohlcv(config["data"]["database_path"], symbol, timeframe)
except Exception as exc:
    st.error(f"No local candles found for {symbol} {timeframe}.")
    st.caption("Download that market/timeframe first, or switch back to the configured dataset.")
    st.code(f"python -m src.collect_data --symbol {symbol} --timeframe {timeframe}")
    st.exception(exc)
    st.stop()

signal = latest_signal(candles, config)

col1, col2, col3 = st.columns(3)
col1.metric("Latest signal", signal["signal"])
col2.metric("Model probability", f"{signal["probability"]:.2%}")
col3.metric("Latest candle", str(signal["timestamp"]))

bundle = joblib.load(PROJECT_ROOT / Path(config["model"]["path"]))
result = run_backtest(candles, bundle, config)

st.subheader("Backtest stats")
stats_cols = st.columns(5)
stats_cols[0].metric("Trades", int(result.stats["trades"]))
stats_cols[1].metric("Win rate", f"{result.stats["win_rate"]:.2%}")
stats_cols[2].metric("Return", f"{result.stats["total_return_pct"]:.2f}%")
stats_cols[3].metric("Profit factor", f"{result.stats["profit_factor"]:.2f}")
stats_cols[4].metric("Max drawdown", f"{result.stats["max_drawdown_pct"]:.2f}%")

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

st.subheader("LLM explanation")
if run_llm:
    st.write(explain_signal(signal, config))
else:
    st.write("LLM explanation disabled in the dashboard.")

with st.expander("Latest feature values"):
    st.json(signal["features"])
