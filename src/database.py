from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.utils import PROJECT_ROOT, normalize_table_name


def db_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def connect(path: str | Path) -> duckdb.DuckDBPyConnection:
    resolved = db_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(resolved))


def save_ohlcv(
    candles: pd.DataFrame,
    database_path: str | Path,
    symbol: str,
    timeframe: str,
) -> None:
    table = normalize_table_name(symbol, timeframe)
    clean = candles.drop_duplicates("timestamp").sort_values("timestamp")
    with connect(database_path) as con:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                timestamp TIMESTAMP PRIMARY KEY,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            )
            """
        )
        con.register("new_candles", clean)
        con.execute(
            f"""
            DELETE FROM {table}
            WHERE timestamp IN (SELECT timestamp FROM new_candles)
            """
        )
        con.execute(
            f"""
            INSERT INTO {table}
            SELECT timestamp, open, high, low, close, volume
            FROM new_candles
            """
        )


def load_ohlcv(database_path: str | Path, symbol: str, timeframe: str) -> pd.DataFrame:
    table = normalize_table_name(symbol, timeframe)
    with connect(database_path) as con:
        return con.execute(f"SELECT * FROM {table} ORDER BY timestamp").df()
