from __future__ import annotations

import os

import ollama


def explain_signal(signal_payload: dict, config: dict) -> str:
    if not config.get("llm", {}).get("enabled", True):
        return "LLM explanations are disabled in config.yaml."

    host = os.getenv("OLLAMA_HOST", config["llm"].get("host", "http://localhost:11434"))
    model = os.getenv("OLLAMA_MODEL", config["llm"].get("model", "llama3.1"))
    client = ollama.Client(host=host)
    features = signal_payload["features"]
    compact_features = {key: round(float(value), 6) for key, value in features.items()}
    prompt = f"""
You are explaining a paper-trading scalping research signal.
This is not financial advice and no real order should be placed.

Signal: {signal_payload["signal"]}
Model probability: {signal_payload["probability"]:.4f}
Latest features: {compact_features}

Explain in 4 concise bullets why this is bullish, bearish, or no-trade.
Mention uncertainty and the role of fees/slippage.
"""
    try:
        response = client.generate(model=model, prompt=prompt)
        return response.get("response", "").strip()
    except Exception as exc:
        return f"Ollama explanation unavailable: {exc}"
