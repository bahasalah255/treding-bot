"""Guarded Demo-only MetaTrader 5 execution and position sizing."""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5

from trading.groq_signal import Signal

LOGGER = logging.getLogger(__name__)
DEFAULT_RISK_PERCENT = 0.5
DEFAULT_MAX_VOLUME = 1.0
DEFAULT_MAX_SPREAD = 0.50
DEFAULT_MAX_DEVIATION_POINTS = 20
DEFAULT_VOLUME_MODE = "fixed"
DEFAULT_FIXED_VOLUME = 0.02
MAX_CONNECT_ATTEMPTS = 2
DEMO_MAGIC = 26001
DEFAULT_PENDING_TIMEOUT_SECONDS = 300
DEFAULT_PENDING_STATE_PATH = r"logs\pending_executions.sqlite3"

_EXECUTION_LOCK = threading.Lock()
_IN_FLIGHT_SYMBOLS: set[str] = set()


@dataclass(frozen=True)
class PendingExecution:
    symbol: str
    order_id: int | None
    deal_id: int | None
    action: str
    magic: int
    requested_volume: float
    created_at: float
    expires_at: float
    state: str = "PENDING"


_PENDING_EXECUTIONS: dict[str, PendingExecution] = {}


class PositionSizingError(ValueError):
    """Raised when broker specifications cannot produce a safe volume."""


def _configured_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _pending_state_path() -> Path:
    configured = os.environ.get("DEMO_PENDING_STATE_PATH", DEFAULT_PENDING_STATE_PATH)
    path = Path(configured)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _pending_db() -> sqlite3.Connection:
    connection = sqlite3.connect(_pending_state_path(), timeout=5)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_executions (
            symbol TEXT PRIMARY KEY,
            order_id INTEGER,
            deal_id INTEGER,
            action TEXT NOT NULL,
            magic INTEGER NOT NULL,
            requested_volume REAL NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            state TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    return connection


def _persist_pending(pending: PendingExecution) -> None:
    now = time.time()
    with _pending_db() as connection:
        connection.execute(
            """
            INSERT INTO pending_executions
                (symbol, order_id, deal_id, action, magic, requested_volume,
                 created_at, expires_at, state, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                order_id=excluded.order_id,
                deal_id=excluded.deal_id,
                action=excluded.action,
                magic=excluded.magic,
                requested_volume=excluded.requested_volume,
                created_at=excluded.created_at,
                expires_at=excluded.expires_at,
                state=excluded.state,
                updated_at=excluded.updated_at
            """,
            (
                pending.symbol,
                pending.order_id,
                pending.deal_id,
                pending.action,
                pending.magic,
                pending.requested_volume,
                pending.created_at,
                pending.expires_at,
                pending.state,
                now,
            ),
        )


def _load_pending(symbol: str) -> PendingExecution | None:
    with _pending_db() as connection:
        row = connection.execute(
            """
            SELECT symbol, order_id, deal_id, action, magic, requested_volume,
                   created_at, expires_at, state
            FROM pending_executions
            WHERE symbol = ? AND state IN ('PENDING', 'EXPIRED')
            """,
            (symbol,),
        ).fetchone()
    if row is None:
        return None
    pending = PendingExecution(*row)
    _PENDING_EXECUTIONS[symbol] = pending
    return pending


def _pending_for_symbol(symbol: str) -> PendingExecution | None:
    return _PENDING_EXECUTIONS.get(symbol) or _load_pending(symbol)


def _set_pending_state(pending: PendingExecution, state: str) -> None:
    updated = PendingExecution(
        pending.symbol,
        pending.order_id,
        pending.deal_id,
        pending.action,
        pending.magic,
        pending.requested_volume,
        pending.created_at,
        pending.expires_at,
        state,
    )
    _persist_pending(updated)
    if state == "FILLED":
        _PENDING_EXECUTIONS.pop(pending.symbol, None)
    else:
        _PENDING_EXECUTIONS[pending.symbol] = updated


def _connect_demo() -> tuple[Any, Any, str | None]:
    """Connect once or retry once, returning verified Demo account metadata."""
    last_error = "MT5 initialization failed"
    for attempt in range(MAX_CONNECT_ATTEMPTS):
        if attempt:
            mt5.shutdown()
        if not mt5.initialize():
            last_error = f"MT5 initialization failed: {mt5.last_error()}"
            LOGGER.warning("MT5 reconnect attempt %s failed: %s", attempt + 1, last_error)
            continue
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None or account is None:
            last_error = f"unable to read MT5 permissions: {mt5.last_error()}"
            LOGGER.warning("MT5 connection verification failed: %s", last_error)
            continue
        if not _is_verified_demo_account(account):
            return terminal, account, "account mode is not a verified Demo account"
        return terminal, account, None
    return None, None, last_error


def _select_filling_mode(symbol_info: Any) -> tuple[int | None, str | None]:
    """Select a broker-supported filling mode from the symbol bitmask."""
    supported = getattr(symbol_info, "filling_mode", None)
    if not isinstance(supported, int) or supported <= 0:
        return None, "broker did not provide supported filling modes"
    candidates = (
        (
            "IOC",
            getattr(mt5, "ORDER_FILLING_IOC", 1),
            getattr(mt5, "SYMBOL_FILLING_IOC", 2),
        ),
        (
            "FOK",
            getattr(mt5, "ORDER_FILLING_FOK", 0),
            getattr(mt5, "SYMBOL_FILLING_FOK", 1),
        ),
        (
            "RETURN",
            getattr(mt5, "ORDER_FILLING_RETURN", 2),
            getattr(mt5, "SYMBOL_FILLING_RETURN", 4),
        ),
    )
    for name, mode, flag in candidates:
        if supported & flag:
            LOGGER.info("Selected MT5 filling mode %s (%s)", name, mode)
            return mode, None
    return None, f"unsupported broker filling mode mask: {supported}"


def _retcode_explanation(retcode: int) -> str:
    return {
        10027: "AutoTrading disabled",
        10030: "unsupported or invalid filling mode",
        10016: "invalid stops",
        10014: "invalid volume",
    }.get(retcode, "see MT5 documentation and last_error")


def _execution_evidence(symbol: str, result: Any) -> tuple[bool, Any, Any]:
    """Return whether a fill is evidenced, plus position and history evidence."""
    positions = mt5.positions_get(symbol=symbol)
    position = None
    order_id = getattr(result, "order", None)
    deal_id = getattr(result, "deal", None)
    orders = mt5.history_orders_get(ticket=order_id) if order_id else None
    deals = mt5.history_deals_get(ticket=deal_id) if deal_id else None
    matching_order = next(
        (
            item for item in (orders or [])
            if _execution_record_matches(item, symbol, DEMO_MAGIC)
            and _record_id(item, ("ticket", "order")) == order_id
        ),
        None,
    )
    matching_deal = next(
        (
            item for item in (deals or [])
            if _execution_record_matches(item, symbol, DEMO_MAGIC)
            and _record_related_to_order(item, order_id, deal_id)
        ),
        None,
    )
    related_position_ids = {
        _record_id(item, ("position_id", "position"))
        for item in (deals or [])
        if item is not None
    }
    for candidate in positions or []:
        if not _execution_record_matches(candidate, symbol, DEMO_MAGIC):
            continue
        candidate_ids = {
            _record_id(candidate, ("ticket", "identifier", "position_id"))
        }
        if (
            matching_deal is not None
            and candidate_ids & related_position_ids
        ) or (
            order_id is not None
            and _record_id(candidate, ("order", "order_id")) == order_id
        ):
            position = candidate
            break
    evidenced = matching_deal is not None or position is not None
    return evidenced, position, matching_deal


def _record_id(record: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = getattr(record, name, None)
        if value is not None:
            return value
    return None


def _execution_record_matches(record: Any, symbol: str, magic: int) -> bool:
    return (
        getattr(record, "symbol", None) == symbol
        and getattr(record, "magic", None) == magic
    )


def _is_verified_demo_account(account: Any) -> bool:
    mode = getattr(account, "trade_mode", None)
    demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", None)
    return (
        account is not None
        and not isinstance(mode, bool)
        and demo_mode is not None
        and mode == demo_mode
    )


def _record_related_to_order(record: Any, order_id: int | None, deal_id: int | None) -> bool:
    if deal_id is not None and _record_id(record, ("ticket", "deal")) != deal_id:
        return False
    related_order = _record_id(record, ("order", "order_id"))
    return order_id is None or related_order in {None, order_id}


def _pending_status(pending: PendingExecution) -> str:
    """Reconcile one previously submitted order without resubmitting it."""
    evidenced, position, deal = _execution_evidence(
        pending.symbol,
        type(
            "PendingResult",
            (),
            {"order": pending.order_id, "deal": pending.deal_id},
        )(),
    )
    if evidenced:
        _set_pending_state(pending, "FILLED")
        partial = (
            deal is not None
            and getattr(deal, "volume", None) is not None
            and float(deal.volume) < pending.requested_volume
        )
        status = "ORDER_FILLED (partial)" if partial else "ORDER_FILLED"
        return (
            f"{status}: reconciled "
            f"order={pending.order_id} deal={pending.deal_id} "
            f"position={getattr(position, 'ticket', None)}"
        )
    if time.time() >= pending.expires_at:
        _set_pending_state(pending, "EXPIRED")
        LOGGER.error(
            "Execution status=EXECUTION_EXPIRED order=%s deal=%s",
            pending.order_id,
            pending.deal_id,
        )
        return (
            "EXECUTION_EXPIRED: confirmation timeout; "
            f"order={pending.order_id} requires manual reconciliation"
        )
    return (
        "EXECUTION_UNCONFIRMED: still awaiting evidence "
        f"order={pending.order_id} deal={pending.deal_id}"
    )


def reconcile_pending_executions(symbol: str) -> str | None:
    """Check a pending order once; never submit another order."""
    pending = _pending_for_symbol(symbol)
    if pending is None:
        return None
    if pending.state == "EXPIRED":
        return (
            "EXECUTION_EXPIRED: confirmation timeout; "
            f"order={pending.order_id} requires manual reconciliation"
        )
    terminal, account, connection_error = _connect_demo()
    if connection_error:
        return "EXECUTION_UNCONFIRMED: " + connection_error
    try:
        if not bool(getattr(terminal, "trade_allowed", False)):
            return "EXECUTION_UNCONFIRMED: AutoTrading unavailable"
        return _pending_status(pending)
    finally:
        mt5.shutdown()


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


def _validate_fixed_volume(symbol_info: Any, volume: float) -> str | None:
    """Validate an exact broker-compatible volume without rounding it."""
    from decimal import Decimal, InvalidOperation

    try:
        requested = Decimal(str(volume))
        minimum = Decimal(str(_positive_spec(symbol_info, "volume_min")))
        maximum = Decimal(str(_positive_spec(symbol_info, "volume_max")))
        step = Decimal(str(_positive_spec(symbol_info, "volume_step")))
    except (InvalidOperation, PositionSizingError, ValueError):
        return "fixed volume broker specification is invalid"
    if not requested.is_finite() or requested <= 0:
        return "fixed volume must be finite and positive"
    if requested < minimum:
        return f"fixed volume {volume} is below broker minimum {minimum}"
    if requested > maximum:
        return f"fixed volume {volume} exceeds broker maximum {maximum}"
    if (requested - minimum) % step != 0:
        return f"fixed volume {volume} is incompatible with broker step {step}"
    precision = max(0, -step.as_tuple().exponent)
    if requested != requested.quantize(Decimal(1).scaleb(-precision)):
        return f"fixed volume {volume} exceeds broker volume precision"
    return None


def _calculate_money_amounts(
    symbol: str,
    action: str,
    volume: float,
    entry: float,
    stop_loss: float,
    take_profit: float,
    symbol_info: Any,
) -> tuple[float, float] | None:
    """Calculate loss to SL and profit to TP using MT5 or broker tick specs."""
    if action not in {"BUY", "SELL"}:
        return None
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    try:
        loss_value = mt5.order_calc_profit(
            order_type, symbol, volume, entry, stop_loss
        )
        profit_value = mt5.order_calc_profit(
            order_type, symbol, volume, entry, take_profit
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        loss_value = None
        profit_value = None
    if loss_value is None or profit_value is None:
        tick_size = getattr(symbol_info, "trade_tick_size", None)
        tick_loss = getattr(symbol_info, "trade_tick_value_loss", None)
        tick_profit = getattr(symbol_info, "trade_tick_value_profit", None)
        if tick_profit is None:
            tick_profit = getattr(symbol_info, "trade_tick_value", None)
        if tick_size is None or tick_loss is None or tick_profit is None:
            return None
        try:
            tick_size = float(tick_size)
            tick_loss = float(tick_loss)
            tick_profit = float(tick_profit)
            loss_value = -abs(entry - stop_loss) / tick_size * tick_loss * volume
            profit_value = abs(take_profit - entry) / tick_size * tick_profit * volume
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    try:
        loss_value = abs(float(loss_value))
        profit_value = float(profit_value)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(loss_value)
        or not math.isfinite(profit_value)
        or loss_value <= 0
        or profit_value <= 0
    ):
        return None
    return loss_value, profit_value


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
    with _EXECUTION_LOCK:
        if symbol in _IN_FLIGHT_SYMBOLS:
            return "BLOCKED: execution already in progress for this symbol"
        pending = _pending_for_symbol(symbol)
        if pending is not None:
            if pending.state == "EXPIRED":
                return "BLOCKED: expired execution requires manual reconciliation"
            return "BLOCKED: execution confirmation is pending for this symbol"
        _IN_FLIGHT_SYMBOLS.add(symbol)
    try:
        terminal, account, connection_error = _connect_demo()
        if connection_error:
            return "BLOCKED: " + connection_error
        try:
            if not bool(getattr(terminal, "trade_allowed", False)):
                return "BLOCKED: AutoTrading disabled by MT5 terminal (retcode 10027)"
            if not _is_verified_demo_account(account):
                return "BLOCKED: account mode is not a verified Demo account"
            if not bool(getattr(account, "trade_allowed", False)):
                return "BLOCKED: trading disabled for the MT5 account"
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
            try:
                max_spread = _configured_float("MAX_DEMO_SPREAD", DEFAULT_MAX_SPREAD)
                max_deviation = int(
                    _configured_float(
                        "MAX_DEMO_DEVIATION_POINTS", DEFAULT_MAX_DEVIATION_POINTS
                    )
                )
            except ValueError as exc:
                return f"BLOCKED: execution configuration: {exc}"
            bid = float(getattr(tick, "bid", 0))
            ask = float(getattr(tick, "ask", 0))
            if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask < bid:
                return "BLOCKED: invalid market bid/ask"
            spread = ask - bid
            if spread > max_spread:
                reason = f"spread {spread:.8f} exceeds maximum {max_spread:.8f}"
                LOGGER.warning("Spread rejection symbol=%s reason=%s", symbol, reason)
                return "BLOCKED: " + reason
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
                volume_mode = os.environ.get(
                    "DEMO_VOLUME_MODE", DEFAULT_VOLUME_MODE
                ).strip().lower()
                if volume_mode != "fixed":
                    return (
                        "BLOCKED: unsupported volume mode; fixed mode is required "
                        "for approved Demo execution"
                    )
                fixed_volume = float(
                    os.environ.get("DEMO_FIXED_VOLUME", DEFAULT_FIXED_VOLUME)
                )
                volume_error = _validate_fixed_volume(info, fixed_volume)
                if volume_error:
                    return "BLOCKED: " + volume_error
                risk_percent = float(os.environ.get("DEMO_RISK_PERCENT", DEFAULT_RISK_PERCENT))
                if not 0 < risk_percent <= 100:
                    return "BLOCKED: DEMO_RISK_PERCENT must be between 0 and 100"
                volume = fixed_volume
                amounts = _calculate_money_amounts(
                    symbol,
                    signal.action,
                    volume,
                    entry,
                    signal.stop_loss,
                    signal.take_profit,
                    info,
                )
                if amounts is None:
                    return "BLOCKED: unable to calculate monetary SL/TP risk"
                estimated_risk, estimated_profit = amounts
                max_risk = float(
                    os.environ.get(
                        "DEMO_MAX_RISK_AMOUNT",
                        float(equity) * risk_percent / 100,
                    )
                )
                if not math.isfinite(max_risk) or max_risk < 0:
                    return "BLOCKED: DEMO_MAX_RISK_AMOUNT must be finite and non-negative"
                if estimated_risk > max_risk:
                    return (
                        "BLOCKED: fixed-volume potential loss exceeds risk limit "
                        f"({estimated_risk:.2f} > {max_risk:.2f})"
                    )
                LOGGER.info(
                    "Fixed Demo volume=%s potential_loss=%s potential_profit=%s",
                    volume,
                    estimated_risk,
                    estimated_profit,
                )
            except (TypeError, ValueError, PositionSizingError) as exc:
                return f"BLOCKED: fixed-volume risk configuration: {exc}"
            filling_mode, filling_error = _select_filling_mode(info)
            if filling_error:
                LOGGER.error("Filling mode rejection symbol=%s reason=%s", symbol, filling_error)
                return "BLOCKED: " + filling_error
            latest_account = mt5.account_info()
            if not _is_verified_demo_account(latest_account):
                return "BLOCKED: account mode changed or cannot be verified"
            latest_positions = mt5.positions_get(symbol=symbol)
            if latest_positions is None:
                return f"BLOCKED: unable to recheck positions: {mt5.last_error()}"
            if len(latest_positions) >= 1:
                return "BLOCKED: a position appeared before submission"
            order_type = mt5.ORDER_TYPE_BUY if signal.action == "BUY" else mt5.ORDER_TYPE_SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": entry,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": max_deviation,
                "magic": DEMO_MAGIC,
                "comment": (
                    "test-bot-demo"
                    if signal.rationale == "FORCED TEST TRADE"
                    else "groq-demo"
                ),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            check = mt5.order_check(request)
            LOGGER.info("MT5 order_check request=%s result=%s", request, check)
            if check is None or check.retcode != 0:
                return f"BLOCKED: order_check failed: {check or mt5.last_error()}"
            latest_positions = mt5.positions_get(symbol=symbol)
            if latest_positions is None:
                return f"BLOCKED: unable to recheck positions: {mt5.last_error()}"
            if len(latest_positions) >= 1:
                return "BLOCKED: a position appeared before submission"
            LOGGER.info("Prepared Demo order volume=%s estimated_risk=%s", volume, estimated_risk)
            result = mt5.order_send(request)
            if result is None:
                LOGGER.error("Execution status=ORDER_REJECTED reason=%s", mt5.last_error())
                return f"ORDER_REJECTED: {mt5.last_error()}"
            LOGGER.info(
                "MT5 order_send result retcode=%s comment=%s order=%s deal=%s "
                "volume=%s price=%s",
                getattr(result, "retcode", None),
                getattr(result, "comment", ""),
                getattr(result, "order", None),
                getattr(result, "deal", None),
                getattr(result, "volume", None),
                getattr(result, "price", None),
            )
            done = getattr(mt5, "TRADE_RETCODE_DONE", 10009)
            partial = getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)
            placed = getattr(mt5, "TRADE_RETCODE_PLACED", 10008)
            now = time.time()
            try:
                timeout = _configured_float(
                    "PENDING_EXECUTION_TIMEOUT_SECONDS",
                    DEFAULT_PENDING_TIMEOUT_SECONDS,
                )
            except ValueError as exc:
                return f"BLOCKED: execution configuration: {exc}"
            pending = PendingExecution(
                symbol, getattr(result, "order", None), getattr(result, "deal", None),
                signal.action, DEMO_MAGIC, volume, now, now + timeout
            )
            if result.retcode == placed:
                _persist_pending(pending)
                _PENDING_EXECUTIONS[symbol] = pending
                LOGGER.info(
                    "Execution status=ORDER_SENT order=%s comment=%s",
                    getattr(result, "order", None),
                    getattr(result, "comment", ""),
                )
                return f"ORDER_SENT: order={getattr(result, 'order', None)} pending execution"
            if result.retcode not in {done, partial}:
                LOGGER.error(
                    "MT5 order_send rejected retcode=%s (%s) last_error=%s",
                    result.retcode,
                    _retcode_explanation(result.retcode),
                    mt5.last_error(),
                )
                LOGGER.warning(
                    "Execution status=ORDER_REJECTED retcode=%s result=%s",
                    result.retcode,
                    result,
                )
                return f"ORDER_REJECTED: {result}"
            try:
                evidenced, position, deals = _execution_evidence(symbol, result)
            except Exception as exc:
                _persist_pending(pending)
                _PENDING_EXECUTIONS[symbol] = pending
                LOGGER.error("Execution status=EXECUTION_UNCONFIRMED evidence_error=%s", exc)
                return f"EXECUTION_UNCONFIRMED: unable to verify execution: {exc}"
            if not evidenced:
                _persist_pending(pending)
                _PENDING_EXECUTIONS[symbol] = pending
                LOGGER.error(
                    "Execution status=EXECUTION_UNCONFIRMED order=%s deal=%s",
                    getattr(result, "order", None),
                    getattr(result, "deal", None),
                )
                return "EXECUTION_UNCONFIRMED: no matching position or history evidence"
            execution_price = getattr(result, "price", None)
            if execution_price is None and position is not None:
                execution_price = getattr(position, "price_open", None)
            if execution_price is not None:
                price_delta = abs(float(execution_price) - entry)
                point = _positive_spec(info, "point")
                LOGGER.info(
                    "Execution price requested=%s actual=%s delta=%s points=%s",
                    entry,
                    execution_price,
                    price_delta,
                    price_delta / point,
                )
                if price_delta > max_deviation * point:
                    LOGGER.error(
                        "Execution slippage exceeded requested=%s actual=%s deviation_points=%s",
                        entry,
                        execution_price,
                        max_deviation,
                    )
            status = "ORDER_FILLED"
            if result.retcode == partial:
                status += " (partial)"
            LOGGER.info(
                "Execution status=%s order=%s deal=%s position=%s",
                status,
                getattr(result, "order", None),
                getattr(result, "deal", None),
                getattr(position, "ticket", None),
            )
            return f"{status}: order={getattr(result, 'order', None)} deal={getattr(result, 'deal', None)}"
        finally:
            mt5.shutdown()
    finally:
        with _EXECUTION_LOCK:
            _IN_FLIGHT_SYMBOLS.discard(symbol)
