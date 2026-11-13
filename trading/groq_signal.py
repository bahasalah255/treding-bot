"""Generate and validate a single trading proposal with Groq."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:  # pragma: no cover - optional at import time so tests stay hermetic
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore[assignment]

DEFAULT_MODEL = "openai/gpt-oss-20b"


@dataclass(frozen=True)
class Signal:
    action: str
    stop_loss: float | None
    take_profit: float | None
    rationale: str


def _content(response: Any) -> str:
    return str(response.choices[0].message.content or "")


def _context(frame: pd.DataFrame) -> str:
    """Summarise closed-bar context so the model is not working blind."""
    close = frame["close"]
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    true_range = (frame["high"] - frame["low"]).tail(14)
    atr = float(true_range.mean()) if len(true_range) else float("nan")
    return (
        f"last_close={float(close.iloc[-1]):.5f} "
        f"ema9={float(ema_fast.iloc[-1]):.5f} "
        f"ema21={float(ema_slow.iloc[-1]):.5f} "
        f"atr14={atr:.5f}"
    )


def _client(client: Any = None) -> Any:
    if client is not None:
        return client
    if Groq is None:
        raise RuntimeError("groq is not installed; run: pip install -r requirements.txt")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def build_prompt(frame: pd.DataFrame, context_bars: int = 100) -> str:
    """Build the prompt from closed bars only."""
    snapshot = frame.tail(context_bars).reset_index()
    snapshot["timestamp"] = snapshot["timestamp"].astype(str)
    return (
        "You are a cautious XAUUSD intraday analyst. Use only the supplied closed bars.\n"
        f"Indicator snapshot: {_context(frame)}\n"
        "Return JSON only with keys action, stop_loss, take_profit, rationale.\n"
        "action must be BUY, SELL, or NO_TRADE. For NO_TRADE use null SL/TP.\n"
        "For BUY, stop_loss must be below the last close and take_profit above it.\n"
        "For SELL, stop_loss must be above the last close and take_profit below it.\n"
        "Place the stop beyond recent structure, roughly 1 to 3 ATR away, and aim for\n"
        "at least 1.5x the stop distance. Never invent certainty; prefer NO_TRADE\n"
        "when the setup is unclear.\n\n"
        + snapshot.to_json(orient="records", date_format="iso")
    )


def request_signal(
    frame: pd.DataFrame,
    model: str | None = None,
    context_bars: int = 100,
    client: Any = None,
) -> Signal:
    """Ask Groq for a JSON proposal based only on the latest closed bars."""
    if frame.empty:
        raise ValueError("cannot request a signal from an empty frame")
    groq_client = _client(client)
    response = groq_client.chat.completions.create(
        model=model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        messages=[{"role": "user", "content": build_prompt(frame, context_bars)}],
        temperature=0,
        max_tokens=800,
        response_format={"type": "json_object"},
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
