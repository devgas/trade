# Usability Evaluation Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single CLI launcher, cache expensive dashboard computations, and expose configurable dashboard evaluation controls.

**Architecture:** Keep existing modules as the source of truth and add a thin `src/main.py` launcher that delegates to their module entrypoints. Dashboard caching will live inside `src/dashboard.py` using Streamlit cache decorators around data/model/backtest/diagnostic calls. Dashboard evaluation settings will be sidebar numeric inputs that feed threshold sweep and walk-forward functions without changing core strategy logic.

**Tech Stack:** Python 3.11+, argparse, Streamlit, pytest, existing LightGBM/Pandas/DuckDB modules.

---

## File Structure

- Create `src/main.py`: central CLI command dispatcher for collect/train/backtest/walk-forward/diagnostics/dashboard.
- Create `tests/test_main.py`: tests command construction without launching external processes.
- Modify `src/dashboard.py`: add cache wrappers and sidebar controls for threshold/walk-forward settings.
- Modify `README.md`: document `python -m src.main ...` commands.

---

### Task 1: CLI Launcher

**Files:**
- Create: `src/main.py`
- Test: `tests/test_main.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests**

Create `tests/test_main.py`:

```python
from src.main import build_command


def test_build_command_for_train_preserves_extra_args():
    command = build_command("train", ["--train-fraction", "0.7"])
    assert command == ["python", "-m", "src.train_model", "--train-fraction", "0.7"]


def test_build_command_for_dashboard_uses_streamlit():
    command = build_command("dashboard", [])
    assert command == ["streamlit", "run", "src/dashboard.py"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_main.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.main'`.

- [ ] **Step 3: Implement minimal launcher**

Create `src/main.py`:

```python
from __future__ import annotations

import argparse
import subprocess

COMMANDS = {
    "collect": ["python", "-m", "src.collect_data"],
    "train": ["python", "-m", "src.train_model"],
    "backtest": ["python", "-m", "src.backtest"],
    "walk-forward": ["python", "-m", "src.walk_forward"],
    "diagnostics": ["python", "-m", "src.model_diagnostics"],
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_main.py`
Expected: PASS.

- [ ] **Step 5: Update README**

Add a `Single CLI Launcher` section showing:

```bash
python -m src.main collect --symbol BTC/USDT --timeframe 1m
python -m src.main train --train-fraction 0.8
python -m src.main backtest --optimize-thresholds
python -m src.main walk-forward --train-size 1000 --test-size 300 --max-folds 3
python -m src.main diagnostics
python -m src.main dashboard
```

---

### Task 2: Dashboard Caching

**Files:**
- Modify: `src/dashboard.py`

- [ ] **Step 1: Add cached wrappers**

Add functions near imports:

```python
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
```

- [ ] **Step 2: Replace direct calls**

Use cached functions for loading candles, loading model bundle, backtesting, thresholds, diagnostics, and walk-forward.

- [ ] **Step 3: Verify dashboard compiles**

Run: `.venv/bin/python -m compileall src`
Expected: exit code 0.

---

### Task 3: Configurable Dashboard Evaluation Controls

**Files:**
- Modify: `src/dashboard.py`

- [ ] **Step 1: Add sidebar numeric inputs**

Add:

```python
threshold_min = st.sidebar.number_input("Threshold min", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
threshold_max = st.sidebar.number_input("Threshold max", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
threshold_step = st.sidebar.number_input("Threshold step", min_value=0.01, max_value=0.25, value=0.05, step=0.01)
walk_train_size = st.sidebar.number_input("WF train rows", min_value=100, max_value=100000, value=1000, step=100)
walk_test_size = st.sidebar.number_input("WF test rows", min_value=50, max_value=50000, value=300, step=50)
walk_max_folds = st.sidebar.number_input("WF max folds", min_value=1, max_value=50, value=3, step=1)
```

- [ ] **Step 2: Use input values**

Pass threshold values into cached threshold sweep. Pass walk-forward numeric values into cached walk-forward.

- [ ] **Step 3: Verify with Playwright**

Run Streamlit and open `http://127.0.0.1:8501` with Playwright.
Expected: dashboard renders with sidebar controls and browser console has 0 errors.

---

## Final Verification

- [ ] Run `.venv/bin/pytest` and confirm all tests pass.
- [ ] Run `.venv/bin/python -m compileall src` and confirm exit code 0.
- [ ] Run `python -m src.main diagnostics` from the venv and confirm it writes diagnostics.
- [ ] Verify dashboard with Playwright console check.
- [ ] Commit with message `Add launcher and dashboard controls`.
- [ ] Push to `origin/main`.
