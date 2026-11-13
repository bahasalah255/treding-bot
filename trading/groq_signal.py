"""Deterministic XAUUSD structure analysis with supplementary Groq review."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from groq import Groq

LOGGER = logging.getLogger(__name__)
MIN_ANALYSIS_CANDLES = 60
SWING_LEFT = 2
SWING_RIGHT = 2
MIN_RISK_REWARD = 2.0
RR_REL_TOLERANCE = 0.01
RR_ABS_TOLERANCE = 0.01


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
    local_buy_score: float = 0.0
    local_sell_score: float = 0.0
    local_score_breakdown: dict[str, Any] = field(default_factory=dict)
    diagnostic: str | None = None


def classify_structure(swing_highs: list[float], swing_lows: list[float]) -> tuple[str, ...]:
    """Classify the latest confirmed high and low against their predecessors."""
    labels: list[str] = []
    if len(swing_highs) >= 2:
        labels.append("HH" if swing_highs[-1] > swing_highs[-2] else "LH")
    if len(swing_lows) >= 2:
        labels.append("HL" if swing_lows[-1] > swing_lows[-2] else "LL")
    return tuple(labels or ["UNCONFIRMED"])


def _data_error(reason: str, diagnostic: str | None = None) -> Signal:
    return Signal(
        "DATA_ERROR", None, None, None, None, "UNKNOWN", "INVALID_DATA",
        0.0, 0.0, reason, (reason,),
        diagnostic=diagnostic,
    )


def _no_trade(
    reason: str,
    payload: dict[str, Any],
    evidence: dict[str, Any] | None = None,
) -> Signal:
    local = evidence or {}
    return Signal(
        "NO_TRADE", None, None, None, None,
        str(payload.get("market_trend", "UNKNOWN")),
        str(payload.get("market_condition", "REJECTED")),
        float(payload["buy_setup_score"]),
        float(payload["sell_setup_score"]),
        str(payload.get("rationale", reason)),
        tuple(payload["rejection_reasons"]) + (reason,),
        float(local.get("local_buy_score", 0.0)),
        float(local.get("local_sell_score", 0.0)),
        dict(local.get("local_score_breakdown", {})),
    )


def _validate_frame(frame: pd.DataFrame) -> str | None:
    required = {"open", "high", "low", "close"}
    if frame.empty:
        return "candle frame is empty"
    if len(frame) < MIN_ANALYSIS_CANDLES:
        return (
            f"not enough candles for EMA50 and confirmed structure: "
            f"required={MIN_ANALYSIS_CANDLES}, received={len(frame)}"
        )
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
    prices = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    if prices.isna().any().any():
        return "OHLC contains missing or non-numeric values"
    if not prices.map(math.isfinite).all().all():
        return "OHLC contains non-finite values"
    if (prices <= 0).any().any():
        return "OHLC prices must be positive"
    if (prices["high"] < prices[["open", "close"]].max(axis=1)).any():
        return "high violates OHLC invariants"
    if (prices["low"] > prices[["open", "close"]].min(axis=1)).any():
        return "low violates OHLC invariants"
    return None


def _confirmed_pivots(data: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Return pivots whose right confirmation candles have already closed."""
    result: dict[str, list[dict[str, Any]]] = {"high": [], "low": []}
    for kind, column, is_high in (
        ("high", "high", True),
        ("low", "low", False),
    ):
        values = data[column].to_numpy(dtype=float)
        for index in range(SWING_LEFT, len(data) - SWING_RIGHT):
            left = values[index - SWING_LEFT:index]
            right = values[index + 1:index + SWING_RIGHT + 1]
            pivot = values[index]
            if is_high:
                confirmed = pivot > left.max() and pivot >= right.max()
            else:
                confirmed = pivot < left.min() and pivot <= right.min()
            if confirmed:
                result[kind].append(
                    {
                        "timestamp": data.index[index].isoformat(),
                        "confirmation_timestamp": data.index[
                            index + SWING_RIGHT
                        ].isoformat(),
                        "price": pivot,
                    }
                )
    return result


def _label_pivots(pivots: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for current, previous in zip(pivots[1:], pivots):
        label = ("HH" if current["price"] > previous["price"] else "LH") \
            if kind == "high" else ("HL" if current["price"] > previous["price"] else "LL")
        labeled.append({**current, "label": label})
    return labeled


def _detect_breakout_retest(
    data: pd.DataFrame,
    range_high: pd.Series,
    range_low: pd.Series,
    atr: pd.Series,
) -> dict[str, Any]:
    """Detect only confirmed, causal breakout-retests ending at the latest candle."""
    result: dict[str, Any] = {
        "bullish": {
            "confirmed": False, "invalidated": False, "broken_level": None,
            "breakout_timestamp": None, "retest_timestamp": None,
            "retest_distance": None, "confirmation_candle": None,
        },
        "bearish": {
            "confirmed": False, "invalidated": False, "broken_level": None,
            "breakout_timestamp": None, "retest_timestamp": None,
            "retest_distance": None, "confirmation_candle": None,
        },
    }
    close = data["close"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)

    for direction, level_series, is_bullish in (
        ("bullish", range_high, True),
        ("bearish", range_low, False),
    ):
        levels = level_series.to_numpy(dtype=float)
        candidate: dict[str, Any] | None = None
        for index in range(1, len(data)):
            level = levels[index]
            if not math.isfinite(level):
                continue
            broke = (
                close[index] > level and close[index - 1] <= level
                if is_bullish else
                close[index] < level and close[index - 1] >= level
            )
            if broke:
                candidate = {
                    "confirmed": False,
                    "invalidated": False,
                    "broken_level": float(level),
                    "breakout_timestamp": data.index[index].isoformat(),
                    "retest_timestamp": None,
                    "retest_distance": None,
                    "confirmation_candle": None,
                }
                continue
            if candidate is None or index <= 1:
                continue
            if candidate["invalidated"]:
                continue
            tolerance = max(
                float(atr.iloc[index]) * 0.25
                if pd.notna(atr.iloc[index]) else 0.0,
                close[index] * 0.0001,
            )
            broken_level = float(candidate["broken_level"])
            invalidated = (
                close[index] < broken_level - tolerance
                if is_bullish else
                close[index] > broken_level + tolerance
            )
            if invalidated:
                candidate["invalidated"] = True
                candidate["confirmed"] = False
                continue
            retested = (
                low[index] <= broken_level + tolerance and close[index] > broken_level
                if is_bullish else
                high[index] >= broken_level - tolerance and close[index] < broken_level
            )
            if retested:
                candidate["confirmed"] = True
                candidate["retest_timestamp"] = data.index[index].isoformat()
                candidate["retest_distance"] = abs(
                    (low[index] if is_bullish else high[index]) - broken_level
                )
                candidate["confirmation_candle"] = data.index[index].isoformat()
        result[direction] = candidate or result[direction]
    return result


def _score_market(evidence: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
    """Score only positive local evidence on a documented 0-100 scale."""
    contributions = {
        "ema_alignment": {"buy": 20.0, "sell": 20.0},
        "structure": {"buy": 20.0, "sell": 20.0},
        "bos_or_choch": {"buy": 20.0, "sell": 20.0},
        "liquidity_sweep": {"buy": 10.0, "sell": 10.0},
        "breakout_retest": {"buy": 20.0, "sell": 20.0},
        "entry_quality": {"buy": 10.0, "sell": 10.0},
    }
    scores = {"buy": 0.0, "sell": 0.0}
    details: dict[str, Any] = {}
    bullish_trend = evidence["ema20"] > evidence["ema50"]
    details["ema_alignment"] = {"buy": 20.0 if bullish_trend else 0.0,
                                "sell": 20.0 if not bullish_trend else 0.0}
    scores["buy"] += details["ema_alignment"]["buy"]
    scores["sell"] += details["ema_alignment"]["sell"]
    labels = set(evidence["structure"])
    details["structure"] = {
        "buy": 20.0 if labels.intersection({"HH", "HL"}) == {"HH", "HL"} else 0.0,
        "sell": 20.0 if labels.intersection({"LH", "LL"}) == {"LH", "LL"} else 0.0,
    }
    scores["buy"] += details["structure"]["buy"]
    scores["sell"] += details["structure"]["sell"]
    details["bos_or_choch"] = {
        "buy": 20.0 if evidence["bos"]["up"] or evidence["choch"]["bullish"] else 0.0,
        "sell": 20.0 if evidence["bos"]["down"] or evidence["choch"]["bearish"] else 0.0,
    }
    scores["buy"] += details["bos_or_choch"]["buy"]
    scores["sell"] += details["bos_or_choch"]["sell"]
    details["liquidity_sweep"] = {
        "buy": 10.0 if evidence["liquidity_sweep"]["buy_side"] else 0.0,
        "sell": 10.0 if evidence["liquidity_sweep"]["sell_side"] else 0.0,
    }
    scores["buy"] += details["liquidity_sweep"]["buy"]
    scores["sell"] += details["liquidity_sweep"]["sell"]
    details["breakout_retest"] = {
        "buy": 20.0 if evidence["breakout_retest"]["bullish"]["confirmed"] else 0.0,
        "sell": 20.0 if evidence["breakout_retest"]["bearish"]["confirmed"] else 0.0,
    }
    scores["buy"] += details["breakout_retest"]["buy"]
    scores["sell"] += details["breakout_retest"]["sell"]
    atr = evidence["atr14"] or 0.0
    close = evidence["latest_close"]
    details["entry_quality"] = {
        "buy": 10.0 if atr and close - evidence["support"] <= atr else 0.0,
        "sell": 10.0 if atr and evidence["resistance"] - close <= atr else 0.0,
    }
    scores["buy"] += details["entry_quality"]["buy"]
    scores["sell"] += details["entry_quality"]["sell"]
    return scores["buy"], scores["sell"], details


def _market_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    """Build causal evidence using only candles at or before the latest candle."""
    data = frame.sort_index().copy()
    ohlc = ["open", "high", "low", "close"]
    data[ohlc] = data[ohlc].apply(pd.to_numeric)
    data["ema20"] = data["close"].ewm(
        span=20, adjust=False, min_periods=20
    ).mean()
    data["ema50"] = data["close"].ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    data["range_high"] = data["high"].rolling(20, min_periods=20).max().shift(1)
    data["range_low"] = data["low"].rolling(20, min_periods=20).min().shift(1)
    data["atr14"] = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1).rolling(14, min_periods=14).mean()
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    recent = data.tail(50)
    pivots = _confirmed_pivots(data)
    high_labels = _label_pivots(pivots["high"], "high")
    low_labels = _label_pivots(pivots["low"], "low")
    structure = classify_structure(
        [item["price"] for item in pivots["high"][-2:]],
        [item["price"] for item in pivots["low"][-2:]],
    )
    width = (
        float(latest["range_high"] - latest["range_low"])
        if pd.notna(latest["range_high"]) and pd.notna(latest["range_low"])
        else 0.0
    )
    in_range = width > 0 and width / float(latest["close"]) < 0.003
    near_range_middle = (
        in_range
        and abs(float(latest["close"]) - (
            float(latest["range_high"]) + float(latest["range_low"])
        ) / 2) < width * 0.2
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
    last_high = pivots["high"][-1]["price"] if pivots["high"] else None
    last_low = pivots["low"][-1]["price"] if pivots["low"] else None
    bos_up = bool(last_high is not None and latest["close"] > last_high)
    bos_down = bool(last_low is not None and latest["close"] < last_low)
    labels = set(structure)
    choch_bullish = bool(bos_up and "LH" in labels and "LL" in labels)
    choch_bearish = bool(bos_down and "HH" in labels and "HL" in labels)
    evidence = {
        "as_of": data.index[-1].isoformat(),
        "latest_close": float(latest["close"]),
        "ema20": float(latest["ema20"]),
        "ema50": float(latest["ema50"]),
        "atr14": float(latest["atr14"]),
        "structure": structure,
        "confirmed_swings": {"highs": high_labels[-6:], "lows": low_labels[-6:]},
        "support": float(recent["low"].min()),
        "resistance": float(recent["high"].max()),
        "range": {
            "detected": in_range,
            "middle": near_range_middle,
            "high": float(latest["range_high"]) if pd.notna(latest["range_high"]) else None,
            "low": float(latest["range_low"]) if pd.notna(latest["range_low"]) else None,
        },
        "liquidity_sweep": {
            "buy_side": sweep_low,
            "sell_side": sweep_high,
            "swept_level": float(latest["range_low"] if sweep_low else latest["range_high"])
            if sweep_low or sweep_high else None,
            "timestamp": data.index[-1].isoformat() if sweep_low or sweep_high else None,
            "candle_high": float(latest["high"]) if sweep_low or sweep_high else None,
            "candle_low": float(latest["low"]) if sweep_low or sweep_high else None,
            "close": float(latest["close"]) if sweep_low or sweep_high else None,
            "confirmed": bool(sweep_low or sweep_high),
        },
        "bos": {
            "up": bos_up, "down": bos_down,
            "level": last_high if bos_up else last_low if bos_down else None,
            "timestamp": data.index[-1].isoformat() if bos_up or bos_down else None,
        },
        "choch": {"bullish": choch_bullish, "bearish": choch_bearish},
    }
    evidence["breakout_retest"] = _detect_breakout_retest(
        data, data["range_high"], data["range_low"], data["atr14"]
    )
    buy_score, sell_score, breakdown = _score_market(evidence)
    evidence["local_buy_score"] = buy_score
    evidence["local_sell_score"] = sell_score
    evidence["local_score_breakdown"] = breakdown
    return evidence


def _content(response: object) -> str:
    return str(response.choices[0].message.content or "")


def _redact_diagnostic(value: str, limit: int = 2000) -> str:
    """Keep malformed-response diagnostics bounded and free of common secrets."""
    redacted = value[:limit]
    redacted = re.sub(
        r"(?i)(api[_-]?key\s*[:=]\s*[\"']?)[^,\"'}\s]+",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^,\"'}\s]+",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)\b(?:sk[-_]|gsk_|key-)[A-Za-z0-9_-]+",
        "[REDACTED]",
        redacted,
    )
    return redacted


def _rr_mismatch_diagnostic(
    action: str,
    entry: float | None,
    stop_loss: float | None,
    take_profit: float | None,
    supplied_rr: float | None,
    calculated_rr: float | None,
    timestamp: str | None,
) -> str:
    """Format numeric RR evidence without retaining the raw provider response."""
    absolute_difference = (
        abs(supplied_rr - calculated_rr)
        if supplied_rr is not None and calculated_rr is not None
        else None
    )
    relative_difference = (
        absolute_difference / abs(calculated_rr)
        if absolute_difference is not None and calculated_rr
        else None
    )
    values = {
        "symbol": "XAUUSD",
        "timestamp": timestamp or "unavailable",
        "action": action or "unavailable",
        "groq_entry": entry,
        "groq_stop_loss": stop_loss,
        "groq_take_profit": take_profit,
        "groq_risk_reward": supplied_rr,
        "calculated_risk_reward": calculated_rr,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "rel_tol": RR_REL_TOLERANCE,
        "abs_tol": RR_ABS_TOLERANCE,
    }
    return "Groq RR mismatch diagnostics: " + " ".join(
        f"{key}={value if value is not None else 'unavailable'}"
        for key, value in values.items()
    )


def _parser_error(reason: str, content: str) -> Signal:
    diagnostic = _redact_diagnostic(content)
    LOGGER.warning(
        "Malformed Groq structured response: reason=%s response=%s",
        reason,
        diagnostic,
    )
    return _data_error(reason, diagnostic)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_setup(
    action: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    evidence: dict[str, Any],
) -> str | None:
    """Reject invalid levels and targets beyond unbroken structure."""
    try:
        entry = _finite_number(entry, "entry_price")
        stop_loss = _finite_number(stop_loss, "stop_loss")
        take_profit = _finite_number(take_profit, "take_profit")
    except ValueError as exc:
        return str(exc)
    if min(entry, stop_loss, take_profit) <= 0:
        return "trade prices must be positive"
    if action == "BUY" and not stop_loss < entry < take_profit:
        return "BUY requires stop_loss < entry < take_profit"
    if action == "SELL" and not take_profit < entry < stop_loss:
        return "SELL requires take_profit < entry < stop_loss"
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk <= 0 or reward <= 0 or reward / risk < MIN_RISK_REWARD:
        return "trade failed directional or minimum 1:2 risk-reward validation"
    atr = float(evidence["atr14"])
    support = float(evidence["support"])
    resistance = float(evidence["resistance"])
    latest_close = evidence.get("latest_close")
    if latest_close is not None:
        latest_close = float(latest_close)
    if latest_close is not None and abs(entry - latest_close) > max(
        atr, latest_close * 0.001
    ):
        return "entry is too far from the latest closed market price"
    breakout_retest = evidence.get(
        "breakout_retest",
        {"bullish": {"confirmed": False}, "bearish": {"confirmed": False}},
    )
    if action == "SELL" and take_profit < support:
        if not evidence["bos"]["down"] or not breakout_retest["bearish"]["confirmed"]:
            return "SELL take-profit is below support without confirmed bearish BOS and retest"
        if entry > support + atr * 0.5:
            return "SELL entry is not a valid breakdown-retest near broken support"
    if action == "BUY" and take_profit > resistance:
        if not evidence["bos"]["up"] or not breakout_retest["bullish"]["confirmed"]:
            return "BUY take-profit is above resistance without confirmed bullish BOS and retest"
        if entry < resistance - atr * 0.5:
            return "BUY_RETEST_INVALID: entry is not near broken resistance"
    if evidence["range"]["middle"]:
        return "entry is in the middle of a consolidation range"
    return None


def request_signal(frame: pd.DataFrame) -> Signal:
    """Return a structured signal or DATA_ERROR; never return an incomplete trade."""
    error = _validate_frame(frame)
    if error:
        return _data_error(error)
    try:
        evidence = _market_evidence(frame)
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        return _data_error(f"market evidence calculation failed: {exc}")
    if not all(math.isfinite(float(evidence[key])) for key in ("ema20", "ema50", "atr14")):
        return _data_error("indicator warm-up is incomplete")
    prompt = (
        "Analyze XAUUSD M5 using ONLY this causal evidence from closed candles. "
        "Evaluate BUY and SELL independently. Groq is supplementary and cannot "
        "override local validation. Return JSON only with exactly this schema: "
        "action, entry_price, stop_loss, take_profit, risk_reward, market_trend, "
        "market_condition, buy_setup_score, sell_setup_score, rationale, "
        "rejection_reasons. Scores must be numbers from 0 to 100. "
        "rejection_reasons MUST ALWAYS be a JSON array of strings. Use [] when "
        "there are no rejection reasons. It must never be a string, null, object, "
        "number array, or omitted. Use NO_TRADE with null prices if neither side "
        "is valid. Never invent missing data.\n\n"
        "EVIDENCE:\n" + json.dumps(evidence, separators=(",", ":"))
    )
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1200,
            reasoning_effort="low",
        )
        content = _content(response)
    except Exception as exc:
        return _data_error(f"Groq analysis failed: {type(exc).__name__}: {exc}")
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return _parser_error("Groq returned no JSON object", content)
    try:
        payload = json.loads(content[start:end + 1])
        required = {
            "action", "entry_price", "stop_loss", "take_profit", "risk_reward",
            "market_trend", "market_condition", "buy_setup_score",
            "sell_setup_score", "rationale", "rejection_reasons",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Groq response missing fields: {sorted(missing)}")
        action = payload["action"]
        if action not in {"BUY", "SELL", "NO_TRADE"}:
            raise ValueError("Groq returned an invalid action")
        scores = (
            _finite_number(payload["buy_setup_score"], "buy_setup_score"),
            _finite_number(payload["sell_setup_score"], "sell_setup_score"),
        )
        if any(score < 0 or score > 100 for score in scores):
            raise ValueError("Groq scores must be between 0 and 100")
        rejection = payload["rejection_reasons"]
        if not isinstance(rejection, list) or not all(
            isinstance(item, str) for item in rejection
        ):
            raise ValueError("rejection_reasons must be a list of strings")
        if not isinstance(payload["rationale"], str):
            raise ValueError("rationale must be a string")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _parser_error(f"invalid structured Groq response: {exc}", content)
    if action == "NO_TRADE":
        if any(payload[field] is not None for field in ("entry_price", "stop_loss", "take_profit")):
            return _data_error("NO_TRADE must use null trade prices")
        return Signal(
            action, None, None, None, None, str(payload["market_trend"]),
            str(payload["market_condition"]), *scores, payload["rationale"],
            tuple(rejection), evidence["local_buy_score"],
            evidence["local_sell_score"], evidence["local_score_breakdown"],
        )
    try:
        entry = _finite_number(payload["entry_price"], "entry_price")
        stop_loss = _finite_number(payload["stop_loss"], "stop_loss")
        take_profit = _finite_number(payload["take_profit"], "take_profit")
        supplied_rr = _finite_number(payload["risk_reward"], "risk_reward")
    except ValueError as exc:
        return _data_error(str(exc))
    calculated_rr = abs(take_profit - entry) / abs(entry - stop_loss) \
        if entry != stop_loss else 0.0
    if not math.isclose(
        supplied_rr,
        calculated_rr,
        rel_tol=RR_REL_TOLERANCE,
        abs_tol=RR_ABS_TOLERANCE,
    ):
        diagnostic = _rr_mismatch_diagnostic(
            action,
            entry,
            stop_loss,
            take_profit,
            supplied_rr,
            calculated_rr,
            evidence.get("as_of"),
        )
        LOGGER.warning("%s", diagnostic)
        return _data_error("Groq risk_reward does not match calculated prices", diagnostic)
    setup_error = _validate_setup(action, entry, stop_loss, take_profit, evidence)
    if setup_error:
        return _no_trade(setup_error, payload, evidence)
    return Signal(
        action, entry, stop_loss, take_profit, calculated_rr,
        str(payload["market_trend"]), str(payload["market_condition"]),
        *scores, payload["rationale"], tuple(rejection),
        evidence["local_buy_score"], evidence["local_sell_score"],
        evidence["local_score_breakdown"],
    )
