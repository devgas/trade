from __future__ import annotations

import argparse
import time

import ccxt
import pandas as pd

from src.database import save_ohlcv
from src.utils import ensure_dirs, load_config


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since: str,
    limit: int,
    market_type: str,
    max_batches: int | None = None,
) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": market_type}})
    since_ms = exchange.parse8601(since)
    rows: list[list[float]] = []

    batch_count = 0
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        batch_count += 1
        next_since = batch[-1][0] + 1
        if next_since <= since_ms:
            break
        since_ms = next_since
        if max_batches is not None and batch_count >= max_batches:
            break
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).assign(
        timestamp=lambda df: pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance OHLCV candles into DuckDB.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--since", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    config = load_config()
    data_cfg = config["data"]
    exchange_cfg = config["exchange"]
    symbol = args.symbol or data_cfg["symbol"]
    timeframe = args.timeframe or data_cfg["timeframe"]
    since = args.since or data_cfg["since"]

    candles = fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        limit=data_cfg["limit_per_request"],
        market_type=exchange_cfg.get("market_type", "spot"),
        max_batches=args.max_batches,
    )
    save_ohlcv(candles, data_cfg["database_path"], symbol, timeframe)
    print(f"Saved {len(candles)} candles for {symbol} {timeframe}")


if __name__ == "__main__":
    main()
