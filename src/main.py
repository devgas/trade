from __future__ import annotations

import argparse
import subprocess
import sys

COMMANDS = {
    "collect": [sys.executable, "-m", "src.collect_data"],
    "train": [sys.executable, "-m", "src.train_model"],
    "backtest": [sys.executable, "-m", "src.backtest"],
    "walk-forward": [sys.executable, "-m", "src.walk_forward"],
    "diagnostics": [sys.executable, "-m", "src.model_diagnostics"],
    "dashboard": ["streamlit", "run", "src/dashboard.py"],
}


def build_command(command: str, extra_args: list[str]) -> list[str]:
    if command not in COMMANDS:
        valid = ", ".join(sorted(COMMANDS))
        raise ValueError(f"Unknown command '{command}'. Valid commands: {valid}")
    return [*COMMANDS[command], *extra_args]


def main() -> None:
    parser = argparse.ArgumentParser(description="AI scalping research launcher.")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    raise SystemExit(subprocess.call(build_command(parsed.command, parsed.args)))


if __name__ == "__main__":
    main()
