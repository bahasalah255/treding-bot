"""Generate and validate a single trading proposal with Groq."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd
from groq import Groq


@dataclass(frozen=True)
class Signal:
    action: str
    stop_loss: float | None
    take_profit: float | None
    rationale: str


def _content(response: object) -> str:
    return str(response.choices[0].message.content or "")


def request_signal(frame: pd.DataFrame) -> Signal:
    """Ask Groq for a JSON proposal based only on the latest closed bars."""
    snapshot = frame.tail(100).reset_index()
    snapshot["timestamp"] = snapshot["timestamp"].astype(str)
    prompt = (
        "You are a cautious XAUUSD M5 trading analyst. Use only the supplied closed bars. "
        "Return JSON only with keys action, stop_loss, take_profit, rationale. "
        "action must be BUY, SELL, or NO_TRADE. For NO_TRADE use null SL/TP. "
        "For BUY, stop_loss must be below the latest close and take_profit above it. "
        "For SELL, stop_loss must be above the latest close and take_profit below it. "
        "Never invent certainty; prefer NO_TRADE when the setup is unclear.\n\n"
        + snapshot.to_json(orient="records", date_format="iso")
    )
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=800,
        reasoning_effort="low",
    )
    content = _content(response)
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Groq returned no JSON object")
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Groq returned invalid JSON") from exc

    action = payload.get("action")
    if action not in {"BUY", "SELL", "NO_TRADE"}:
        raise ValueError("Groq returned an invalid action")
    if action == "NO_TRADE":
        return Signal(action, None, None, str(payload.get("rationale", "")))

    try:
        stop_loss = float(payload["stop_loss"])
        take_profit = float(payload["take_profit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BUY/SELL signal must include numeric stop_loss and take_profit") from exc
    latest_close = float(frame["close"].iloc[-1])
    if action == "BUY" and not (stop_loss < latest_close < take_profit):
        raise ValueError("BUY signal has invalid stop-loss or take-profit")
    if action == "SELL" and not (take_profit < latest_close < stop_loss):
        raise ValueError("SELL signal has invalid stop-loss or take-profit")
    return Signal(action, stop_loss, take_profit, str(payload.get("rationale", "")))
