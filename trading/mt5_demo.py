"""Guarded Demo-only MetaTrader 5 execution and position sizing."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import MetaTrader5 as mt5

from trading.groq_signal import Signal

LOGGER = logging.getLogger(__name__)
DEFAULT_RISK_PERCENT = 0.5
DEFAULT_MAX_VOLUME = 1.0


class PositionSizingError(ValueError):
    """Raised when broker specifications cannot produce a safe volume."""


def _positive_spec(symbol_info: Any, name: str) -> float:
    value = getattr(symbol_info, name, None)
    if value is None or float(value) <= 0:
        raise PositionSizingError(f"missing or invalid broker specification: {name}")
    return float(value)


def calculate_volume_for_risk(
    symbol_info: Any,
    entry: float,
    stop_loss: float,
    equity: float,
    risk_percent: float = DEFAULT_RISK_PERCENT,
    max_volume: float = DEFAULT_MAX_VOLUME,
) -> tuple[float, float]:
    """Return a step-aligned volume and its estimated monetary risk."""
    if equity <= 0 or not 0 < risk_percent <= 100:
        raise PositionSizingError("equity and risk_percent must be positive")
    if entry <= 0 or stop_loss <= 0 or entry == stop_loss:
        raise PositionSizingError("entry and stop-loss must be positive and different")
    if max_volume <= 0:
        raise PositionSizingError("max_volume must be positive")
    _positive_spec(symbol_info, "trade_contract_size")
    for currency_field in ("currency_profit", "currency_margin"):
        if not str(getattr(symbol_info, currency_field, "")).strip():
            raise PositionSizingError(f"missing broker specification: {currency_field}")
    tick_size = _positive_spec(symbol_info, "trade_tick_size")
    tick_value = getattr(symbol_info, "trade_tick_value_loss", None) or getattr(
        symbol_info, "trade_tick_value", None
    )
    if tick_value is None or float(tick_value) <= 0:
        raise PositionSizingError("missing or invalid broker specification: trade_tick_value")
    tick_value = float(tick_value)
    volume_min = _positive_spec(symbol_info, "volume_min")
    volume_max = _positive_spec(symbol_info, "volume_max")
    volume_step = _positive_spec(symbol_info, "volume_step")
    distance = abs(entry - stop_loss)
    loss_per_lot = distance / tick_size * tick_value
    risk_amount = equity * risk_percent / 100
    raw_volume = risk_amount / loss_per_lot
    allowed_max = min(volume_max, max_volume)
    volume = min(raw_volume, allowed_max)
    volume = math.floor(volume / volume_step + 1e-9) * volume_step
    volume = round(volume, 8)
    if volume < volume_min:
        raise PositionSizingError(
            "minimum broker volume would exceed the configured monetary risk"
        )
    estimated_risk = volume * loss_per_lot
    if estimated_risk > risk_amount + max(0.01, risk_amount * 1e-9):
        raise PositionSizingError("rounded volume exceeds configured monetary risk")
    return volume, estimated_risk


def _validate_market_levels(
    symbol_info: Any,
    action: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> str | None:
    if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        return "market prices must be positive"
    if action == "BUY" and not stop_loss < entry < take_profit:
        return "BUY requires stop_loss < entry < take_profit"
    if action == "SELL" and not take_profit < entry < stop_loss:
        return "SELL requires take_profit < entry < stop_loss"
    point = _positive_spec(symbol_info, "point")
    stops_level = float(getattr(symbol_info, "trade_stops_level", 0) or 0)
    freeze_level = float(getattr(symbol_info, "trade_freeze_level", 0) or 0)
    minimum_distance = max(stops_level, freeze_level) * point
    if action == "BUY":
        distances = (entry - stop_loss, take_profit - entry)
    else:
        distances = (stop_loss - entry, entry - take_profit)
    if any(distance < minimum_distance for distance in distances):
        return f"SL/TP violates minimum stop or freeze distance {minimum_distance}"
    return None


def execute_demo_signal(symbol: str, signal: Signal) -> str:
    """Execute one validated signal on a Demo account only."""
    if os.environ.get("ENABLE_DEMO_TRADING") != "1":
        return "DRY_RUN: set ENABLE_DEMO_TRADING=1 to allow demo execution"
    if signal.action in {"NO_TRADE", "DATA_ERROR"}:
        return "BLOCKED: " + signal.rationale if signal.action == "DATA_ERROR" else "NO_TRADE"
    if not mt5.initialize():
        return f"BLOCKED: MT5 initialization failed: {mt5.last_error()}"
    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None or account is None:
            return f"BLOCKED: unable to read MT5 permissions: {mt5.last_error()}"
        if not bool(getattr(terminal, "trade_allowed", False)):
            return "BLOCKED: AutoTrading disabled by MT5 terminal (retcode 10027)"
        if not bool(getattr(account, "trade_allowed", False)):
            return "BLOCKED: trading disabled for the MT5 account"
        if account.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL:
            return "BLOCKED: real accounts are prohibited"
        equity = getattr(account, "equity", None)
        if equity is None or float(equity) <= 0:
            return "BLOCKED: account equity is unavailable"
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return f"BLOCKED: unable to inspect positions: {mt5.last_error()}"
        if len(positions) >= 1:
            return "BLOCKED: a position is already open for this symbol"
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None or not mt5.symbol_select(symbol, True):
            return f"BLOCKED: unable to read {symbol}: {mt5.last_error()}"
        entry = float(tick.ask if signal.action == "BUY" else tick.bid)
        if signal.stop_loss is None or signal.take_profit is None:
            return "BLOCKED: stop-loss and take-profit are required"
        level_error = _validate_market_levels(
            info, signal.action, entry, signal.stop_loss, signal.take_profit
        )
        if level_error:
            return "BLOCKED: " + level_error
        reward = abs(signal.take_profit - entry)
        risk = abs(entry - signal.stop_loss)
        if reward / risk < 2:
            return "BLOCKED: current market entry no longer satisfies minimum 1:2"
        try:
            risk_percent = float(os.environ.get("DEMO_RISK_PERCENT", DEFAULT_RISK_PERCENT))
            max_volume = float(os.environ.get("MAX_DEMO_VOLUME", DEFAULT_MAX_VOLUME))
            volume, estimated_risk = calculate_volume_for_risk(
                info, entry, signal.stop_loss, float(equity), risk_percent, max_volume
            )
        except (TypeError, ValueError, PositionSizingError) as exc:
            return f"BLOCKED: position sizing: {exc}"
        order_type = mt5.ORDER_TYPE_BUY if signal.action == "BUY" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": entry,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "deviation": 20,
            "magic": 26001,
            "comment": "groq-demo",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        check = mt5.order_check(request)
        LOGGER.info("MT5 order_check request=%s result=%s", request, check)
        if check is None or check.retcode != 0:
            return f"BLOCKED: order_check failed: {check or mt5.last_error()}"
        LOGGER.info("Prepared Demo order volume=%s estimated_risk=%s", volume, estimated_risk)
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return f"BLOCKED: order_send failed: {result or mt5.last_error()}"
        return f"ORDER_SENT: {result.order}"
    finally:
        mt5.shutdown()
