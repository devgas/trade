# Local AI Crypto/Forex Scalping Research System

Paper-trading research system for Binance market data. It downloads OHLCV candles, stores them in DuckDB, builds scalping features, trains a LightGBM classifier, backtests paper-only signals, and uses a local Ollama model to explain the latest signal.

This project does not place real orders. It has no real execution module and does not store exchange secret keys in code.

## Features

- Binance OHLCV downloader through CCXT
- DuckDB candle storage
- Scalping features: returns, volatility, RSI, EMA 20/50/200, ATR, VWAP distance, volume spike ratio, candle body and wick ratios
- Target: whether price hits `+0.2%` before `-0.15%` within the next 15 candles
- LightGBM classifier saved to `models/`
- Custom paper backtester with fees, slippage, stop loss, take profit, and max trades per day
- Streamlit dashboard with latest signal, probability, backtest stats, equity curve, and Ollama explanation
- Telegram fields are present in `.env.example`, but alerts are disabled by default and not wired to execution

## Setup

```bash
cd ai-scalping-system
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional Ollama setup:

```bash
ollama pull llama3.1
ollama serve
```

## Configure

Edit `config.yaml` for symbol, timeframe, model path, target, and backtest settings.

Default market:

```yaml
symbol: BTC/USDT
timeframe: 1m
```

Do not add exchange secret keys. This is market-data and paper-backtest only.

## Download Data

```bash
python -m src.collect_data --symbol BTC/USDT --timeframe 1m --since 2024-01-01T00:00:00Z
```

Candles are stored in `data/market_data.duckdb`.

## Train Model

```bash
python -m src.train_model
```

The trained model bundle is saved to `models/lightgbm_scalper.pkl`.

## Run Backtest

```bash
python -m src.backtest
```

Backtest trades are written to `data/latest_backtest_trades.csv`.

## Run Walk-Forward Validation

```bash
python -m src.walk_forward --train-size 1000 --test-size 300 --max-folds 3
```

Walk-forward results are written to `data/walk_forward_results.csv`. Each fold trains on an older window and backtests only the next future window.

## Generate Model Diagnostics

```bash
python -m src.model_diagnostics
```

Diagnostics are written to `data/feature_importance.csv` and `data/probability_buckets.csv`. The dashboard also shows these diagnostics when `Show model diagnostics` is enabled.

## Run Dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard shows the latest model signal, probability, backtest stats, equity curve, and a local Ollama explanation if Ollama is running.

## Run Tests

```bash
pytest
```

Tests cover feature generation and the target rule that take profit must be hit before stop loss within the configured horizon.

## Project Structure

```text
ai-scalping-system/
  README.md
  .env.example
  requirements.txt
  config.yaml
  data/
  models/
  notebooks/
  src/
    collect_data.py
    database.py
    features.py
    train_model.py
    backtest.py
    signal_engine.py
    llm_explain.py
    dashboard.py
    utils.py
  tests/
```

## Safety Notes

- Paper trading only.
- No real order execution code is included.
- Do not store exchange API secrets in this project.
- Model outputs are research signals, not financial advice.
