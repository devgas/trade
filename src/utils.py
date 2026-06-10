from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs() -> None:
    for name in ("data", "models", "notebooks"):
        (PROJECT_ROOT / name).mkdir(exist_ok=True)


def normalize_table_name(symbol: str, timeframe: str) -> str:
    return f"ohlcv_{symbol.replace('/', '_').replace(':', '_').lower()}_{timeframe}"
