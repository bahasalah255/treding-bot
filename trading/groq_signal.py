"""Validated XAUUSD structure analysis and Groq signal selection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:  # pragma: no cover - optional so tests stay hermetic
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore[assignment]

DEFAULT_MODEL = "openai/gpt-oss-20b"


@dataclass(frozen=True)
class Signal:
    action: str
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    risk_reward: float | None
    market_trend: str
    market_condition: str
    buy_setup_score: float
    sell_setup_score: float
    rationale: str
    rejection_reasons: tuple[str, ...]


def classify_structure(swing_highs: list[float], swing_lows: list[float]) -> tuple[str, ...]:
    """Classify confirmed swing points using standard HH/HL/LH/LL definitions."""
    labels: list[str] = []
    if len(swing_highs) >= 2:
        labels.append("HH" if swing_highs[-1] > swing_highs[-2] else "LH")
    if len(swing_lows) >= 2:
        labels.append("HL" if swing_lows[-1] > swing_lows[-2] else "LL")
    return tuple(labels or ["UNCONFIRMED"])


def _data_error(reason: str) -> Signal:
    return Signal(
        "DATA_ERROR", None, None, None, None, "UNKNOWN", "INVALID_DATA",
        0.0, 0.0, reason, (reason,),
    )


def _no_trade(reason: str, payload: dict[str, Any]) -> Signal:
    return Signal(
        "NO_TRADE", None, None, None, None,
        str(payload.get("market_trend", "UNKNOWN")),
        str(payload.get("market_condition", "REJECTED")),
        float(payload.get("buy_setup_score", 0)),
        float(payload.get("sell_setup_score", 0)),
        str(payload.get("rationale", reason)),
        (reason,),
    )


def _validate_frame(frame: pd.DataFrame) -> str | None:
    required = {"open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return "missing OHLC columns"
    if not isinstance(frame.index, pd.DatetimeIndex):
        return "timestamps must be a DatetimeIndex"
    if frame.index.tz is None:
        return "timestamps must include an explicit timezone"
    if str(frame.index.tz) != "UTC":
        return "timestamps must be normalized to UTC"
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        return "timestamps must be strictly increasing and unique"
    if frame[list(required)].isna().any().any():
        return "OHLC contains missing values"
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        return "high violates OHLC invariants"
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        return "low violates OHLC invariants"
    return None


def _market_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    """Build causal evidence using only candles at or before the latest candle."""
    data = frame.sort_index().copy()
    data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
    data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["range_high"] = data["high"].rolling(20).max().shift(1)
    data["range_low"] = data["low"].rolling(20).min().shift(1)
    data["atr14"] = (
        pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - data["close"].shift(1)).abs(),
                (data["low"] - data["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1).rolling(14).mean()
    )
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    recent = data.tail(50)
    swing_highs = recent["high"][
        (recent["high"] > recent["high"].shift(1))
        & (recent["high"] >= recent["high"].shift(-1))
    ].dropna()
    swing_lows = recent["low"][
        (recent["low"] < recent["low"].shift(1))
        & (recent["low"] <= recent["low"].shift(-1))
    ].dropna()
    last_highs = swing_highs.tail(2).tolist()
    last_lows = swing_lows.tail(2).tolist()
    structure = classify_structure(last_highs, last_lows)
    width = float(latest["range_high"] - latest["range_low"]) if pd.notna(latest["range_high"]) else 0
    in_range = width > 0 and width / float(latest["close"]) < 0.003
    near_range_middle = (
        in_range
        and abs(float(latest["close"]) - (float(latest["range_high"]) + float(latest["range_low"])) / 2)
        < width * 0.2
    )
    sweep_low = bool(
        pd.notna(latest["range_low"])
        and latest["low"] < latest["range_low"]
        and latest["close"] > latest["range_low"]
    )
    sweep_high = bool(
        pd.notna(latest["range_high"])
        and latest["high"] > latest["range_high"]
        and latest["close"] < latest["range_high"]
    )
    bos_up = bool(pd.notna(previous["range_high"]) and latest["close"] > previous["range_high"])
    bos_down = bool(pd.notna(previous["range_low"]) and latest["close"] < previous["range_low"])
    trend = "BULLISH" if latest["ema20"] > latest["ema50"] else "BEARISH"
    return {
        "as_of": data.index[-1].isoformat(),
        "latest_close": float(latest["close"]),
        "ema20": float(latest["ema20"]),
        "ema50": float(latest["ema50"]),
        "atr14": float(latest["atr14"]) if pd.notna(latest["atr14"]) else None,
        "structure": structure,
        "support": float(recent["low"].min()),
        "resistance": float(recent["high"].max()),
        "range": {
            "detected": in_range,
            "middle": near_range_middle,
            "high": float(latest["range_high"]) if pd.notna(latest["range_high"]) else None,
            "low": float(latest["range_low"]) if pd.notna(latest["range_low"]) else None,
        },
        "liquidity_sweep": {"buy_side": sweep_low, "sell_side": sweep_high},
        "bos": {"up": bos_up, "down": bos_down},
        "choch": {
            "bullish": sweep_low and trend == "BEARISH",
            "bearish": sweep_high and trend == "BULLISH",
        },
    }


def _content(response: object) -> str:
    return str(response.choices[0].message.content or "")


def _client(client: Any = None) -> Any:
    """Return an injected client, or build one from ``GROQ_API_KEY``."""
    if client is not None:
        return client
    if Groq is None:
        raise RuntimeError("groq is not installed; run: pip install -r requirements.txt")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def _validate_setup(action: str, entry: float, stop_loss: float, take_profit: float,
                    evidence: dict[str, Any]) -> str | None:
    """Reject targets beyond structure unless a confirmed break/retest exists."""
    atr = float(evidence["atr14"])
    support = float(evidence["support"])
    resistance = float(evidence["resistance"])
    if action == "SELL" and take_profit < support:
        if not evidence["bos"]["down"]:
            return "SELL take-profit is below support without confirmed bearish BOS"
        if entry > support + atr * 0.5:
            return "SELL entry is not a valid breakdown-retest near broken support"
    if action == "BUY" and take_profit > resistance:
        if not evidence["bos"]["up"]:
            return "BUY take-profit is above resistance without confirmed bullish BOS"
        if entry < resistance - atr * 0.5:
            return "BUY entry is not a valid breakout-retest near broken resistance"
    if evidence["range"]["middle"]:
        return "entry is in the middle of a consolidation range"
    return None


def build_prompt(evidence: dict[str, Any]) -> str:
    """Return the model prompt for one piece of causal market evidence."""
    return (
        "Analyze XAUUSD using ONLY this causal evidence from closed candles. "
        "Evaluate BUY and SELL independently. Detect HH, HL, LH, LL, consolidation, "
        "range middle, liquidity sweeps, BOS, CHoCH, support/resistance, pullback, "
        "and breakout-retest. Separate market direction from entry confirmation. "
        "Reject entries in the range middle. A trade requires a valid entry, SL, TP "
        "and risk_reward >= 2. Return JSON only with exactly these keys: "
        "action, entry_price, stop_loss, take_profit, risk_reward, market_trend, "
        "market_condition, buy_setup_score, sell_setup_score, rationale, rejection_reasons. "
        "Use action NO_TRADE if neither side is valid. Use null prices for NO_TRADE. "
        "Never invent missing data.\n\nEVIDENCE:\n"
        + json.dumps(evidence, separators=(",", ":"))
    )


def request_signal(
    frame: pd.DataFrame,
    model: str | None = None,
    context_bars: int = 100,
    client: Any = None,
) -> Signal:
    """Return a structured signal or DATA_ERROR; never return an incomplete trade."""
    error = _validate_frame(frame)
    if error:
        return _data_error(error)
    if context_bars and context_bars > 0:
        frame = frame.tail(max(context_bars, 60))
    evidence = _market_evidence(frame)
    if evidence["atr14"] is None:
        return _data_error("not enough candles to calculate ATR14")
    try:
        groq_client = _client(client)
        response = groq_client.chat.completions.create(
            model=model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
            messages=[{"role": "user", "content": build_prompt(evidence)}],
            temperature=0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 - never trade on a failed request
        return _data_error(f"Groq request failed: {type(exc).__name__}: {exc}")
    content = _content(response)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return _data_error("Groq returned no JSON object")
    try:
        payload = json.loads(content[start : end + 1])
        action = payload["action"]
        prices = [payload.get("entry_price"), payload.get("stop_loss"), payload.get("take_profit")]
        scores = [float(payload.get("buy_setup_score", 0)), float(payload.get("sell_setup_score", 0))]
        rejection = tuple(str(item) for item in payload.get("rejection_reasons", []))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _data_error(f"invalid structured Groq response: {exc}")
    if action not in {"BUY", "SELL", "NO_TRADE"}:
        return _data_error("Groq returned an invalid action")
    if action == "NO_TRADE":
        return Signal(
            action, None, None, None, None, str(payload.get("market_trend", "UNKNOWN")),
            str(payload.get("market_condition", "UNKNOWN")), *scores,
            str(payload.get("rationale", "")), rejection,
        )
    if any(value is None for value in prices):
        return _data_error("trade signal is missing entry, stop loss, or take profit")
    entry, stop_loss, take_profit = (float(value) for value in prices)
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    risk_reward = reward / risk if risk else 0.0
    valid_direction = (
        action == "BUY" and stop_loss < entry < take_profit
    ) or (
        action == "SELL" and take_profit < entry < stop_loss
    )
    if not valid_direction or risk_reward < 2:
        return _no_trade(
            "trade failed directional or minimum 1:2 risk-reward validation", payload
        )
    setup_error = _validate_setup(action, entry, stop_loss, take_profit, evidence)
    if setup_error:
        return _no_trade(setup_error, payload)
    return Signal(
        action, entry, stop_loss, take_profit, risk_reward,
        str(payload.get("market_trend", "UNKNOWN")),
        str(payload.get("market_condition", "UNKNOWN")),
        *scores, str(payload.get("rationale", "")), rejection,
    )
