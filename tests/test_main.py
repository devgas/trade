import sys

from src.main import build_command


def test_build_command_for_train_preserves_extra_args():
    command = build_command("train", ["--train-fraction", "0.7"])
    assert command == [sys.executable, "-m", "src.train_model", "--train-fraction", "0.7"]


def test_build_command_for_dashboard_uses_streamlit():
    command = build_command("dashboard", [])
    assert command == ["streamlit", "run", "src/dashboard.py"]
