"""Guarded MetaTrader 5 demo execution.

The only account this module will ever trade is an account whose
``trade_mode`` is DEMO. Every other account is refused before a request is
built. Ambiguous submissions are persisted in SQLite and reconciled later so a
restart can never send a duplicate order.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time as day_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

try:  # pragma: no cover - depends on the local terminal environment
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore[assignment]

from trading.config import DemoConfig
from trading.journal import Journal
from trading.risk import (
    Limits,
    MarketState,
    RiskError,
    SymbolSpec,
    entry_block_reason,
    market_level_error,
    volume_for_risk,
)
from trading.state import RuntimeState

LOGGER = logging.getLogger(__name__)
DEFAULT_PENDING_STATE_PATH = "logs/pending_executions.sqlite3"
ROOT = Path(__file__).resolve().parents[1]
PENDING_SCHEMA = """
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



class ExecutionError(RuntimeError):
    """Raised when a demo order cannot be placed safely."""


@dataclass(frozen=True)
class AccountSnapshot:
    login: int
    server: str
    currency: str
    balance: float
    equity: float
    trade_mode: int
    trade_expert: bool
    leverage: int


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    detail: str
    reason: str | None = None
    ticket: int | None = None
    deal: int | None = None
    volume: float | None = None


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


def get_terminal(terminal: Any = None) -> Any:
    """Return the MetaTrader5 module or raise a usable error."""
    module = terminal if terminal is not None else mt5
    if module is None:
        raise ExecutionError(
            "MetaTrader5 is not installed; run: pip install -r requirements-mt5.txt"
        )
    return module


@contextmanager
def session(terminal: Any = None, path: str | None = None) -> Iterator[Any]:
    """Initialize the terminal for a short-lived command."""
    module = get_terminal(terminal)
    if module.terminal_info() is not None:
        yield module
        return
    launched = module.initialize(path) if path else module.initialize()
    if not launched:
        raise ExecutionError(f"MT5 initialization failed: {module.last_error()}")
    try:
        yield module
    finally:
        module.shutdown()


def require_demo_account(
    terminal: Any = None, require_expert: bool = True
) -> AccountSnapshot:
    """Return the account only when it is a demo account.

    ``require_expert`` additionally refuses an account whose algo-trading
    permission is off. Preflight-style read-only commands pass ``False`` so they
    can still report the problem instead of dying on it.
    """
    module = get_terminal(terminal)
    account = module.account_info()
    if account is None:
        raise ExecutionError(
            f"unable to read account information: {module.last_error()}"
        )
    demo_mode = getattr(module, "ACCOUNT_TRADE_MODE_DEMO", 0)
    trade_mode = int(getattr(account, "trade_mode", -1))
    login = int(getattr(account, "login", 0))
    if trade_mode != int(demo_mode):
        # Real-money trading is intentionally not implemented. This is the
        # hard boundary of the project, not a configurable switch.
        raise ExecutionError(
            f"refusing account {login}: trade_mode {trade_mode} is not a demo "
            "account; this bot only ever trades demo accounts"
        )
    if require_expert and not bool(getattr(account, "trade_expert", True)):
        raise ExecutionError(
            "algo trading is disabled for this account; enable Algo Trading in the terminal"
        )
    return AccountSnapshot(
        login=int(getattr(account, "login", 0)),
        server=str(getattr(account, "server", "")),
        currency=str(getattr(account, "currency", "")),
        balance=float(getattr(account, "balance", 0.0)),
        equity=float(getattr(account, "equity", 0.0)),
        trade_mode=trade_mode,
        trade_expert=bool(getattr(account, "trade_expert", True)),
        leverage=int(getattr(account, "leverage", 0)),
    )


def require_symbol(terminal: Any, symbol: str) -> Any:
    """Return symbol info, selecting the symbol in Market Watch if needed."""
    module = get_terminal(terminal)
    info = module.symbol_info(symbol)
    if info is None:
        if not module.symbol_select(symbol, True):
            raise ExecutionError(
                f"symbol {symbol!r} is not available: {module.last_error()}"
            )
        info = module.symbol_info(symbol)
    if info is None:
        raise ExecutionError(
            f"unable to read symbol {symbol!r}: {module.last_error()}"
        )
    disabled = int(getattr(module, "SYMBOL_TRADE_MODE_DISABLED", 0))
    if int(getattr(info, "trade_mode", -1)) == disabled:
        raise ExecutionError(f"trading is disabled for {symbol!r}")
    return info


def spec_from_info(info: Any) -> SymbolSpec:
    """Map MT5 symbol fields onto the pure sizing model."""
    return SymbolSpec(
        point=float(info.point),
        volume_min=float(info.volume_min),
        volume_max=float(info.volume_max),
        volume_step=float(info.volume_step),
        stops_level_points=int(getattr(info, "trade_stops_level", 0) or 0),
        freeze_level_points=int(getattr(info, "trade_freeze_level", 0) or 0),
    )


def _validate_market_levels(
    symbol_info: Any,
    action: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> str | None:
    """Return a broker-level violation message, or ``None`` when the levels fit."""
    return market_level_error(
        action, entry, stop_loss, take_profit, spec_from_info(symbol_info)
    )


def validate_fixed_volume(symbol_info: Any, volume: float) -> str | None:
    """Validate an exact broker-compatible volume without rounding it."""
    from decimal import Decimal, InvalidOperation

    try:
        requested = Decimal(str(volume))
        minimum = Decimal(str(float(symbol_info.volume_min)))
        maximum = Decimal(str(float(symbol_info.volume_max)))
        step = Decimal(str(float(symbol_info.volume_step)))
    except (InvalidOperation, TypeError, ValueError, AttributeError):
        return "fixed volume broker specification is invalid"
    if not requested.is_finite() or requested <= 0:
        return "fixed volume must be finite and positive"
    if minimum <= 0 or maximum <= 0 or step <= 0:
        return "fixed volume broker specification is invalid"
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


def loss_per_lot(
    terminal: Any, symbol: str, entry: float, stop_loss: float, spec: SymbolSpec
) -> tuple[float, float]:
    """Return (money lost per lot at the stop, tick-field estimate).

    The first value comes from the terminal's own ``order_calc_profit``, so it
    already accounts for contract size, currency conversion, and any stale
    ``trade_tick_value`` reporting. The second value is what the tick fields
    imply, returned only so a mismatch can be surfaced in the journal.
    """
    module = get_terminal(terminal)
    distance = abs(entry - stop_loss)
    if distance <= 0:
        raise ExecutionError("entry and stop loss are identical")
    probe = 1.0 if spec.volume_max >= 1.0 else spec.volume_min
    if probe <= 0:
        raise ExecutionError("symbol has no usable volume minimum")
    profit = module.order_calc_profit(
        module.ORDER_TYPE_BUY, symbol, probe, entry, stop_loss
    )
    if profit is not None and abs(float(profit)) > 0:
        per_lot = abs(float(profit)) / probe
    else:
        info = require_symbol(module, symbol)
        contract = float(getattr(info, "trade_contract_size", 0) or 0)
        if contract <= 0:
            raise ExecutionError(
                "cannot size the trade: order_calc_profit and contract size are unavailable"
            )
        per_lot = distance * contract

    info = require_symbol(module, symbol)
    tick_size = float(getattr(info, "trade_tick_size", 0) or 0)
    tick_value = float(getattr(info, "trade_tick_value", 0) or 0)
    tick_estimate = (
        distance / tick_size * tick_value if tick_size > 0 and tick_value > 0 else 0.0
    )
    return per_lot, tick_estimate


def money_amounts(
    terminal: Any,
    symbol: str,
    action: str,
    volume: float,
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> tuple[float, float] | None:
    """Return (loss to SL, profit to TP) for a concrete volume.

    ``order_calc_profit`` is authoritative. If the terminal cannot answer, the
    contract size is used instead of the tick fields, because this broker
    reports ``trade_tick_value`` 10x too small.
    """
    module = get_terminal(terminal)
    if action not in {"BUY", "SELL"} or volume <= 0:
        return None
    order_type = (
        module.ORDER_TYPE_BUY if action == "BUY" else module.ORDER_TYPE_SELL
    )
    loss_value = profit_value = None
    try:
        loss_value = module.order_calc_profit(
            order_type, symbol, volume, entry, stop_loss
        )
        profit_value = module.order_calc_profit(
            order_type, symbol, volume, entry, take_profit
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        loss_value = profit_value = None
    if (
        loss_value is None
        or profit_value is None
        or not math.isfinite(float(loss_value))
        or not math.isfinite(float(profit_value))
    ):
        info = require_symbol(module, symbol)
        contract = float(getattr(info, "trade_contract_size", 0) or 0)
        if contract <= 0:
            return None
        loss_value = abs(entry - stop_loss) * contract * volume
        profit_value = abs(take_profit - entry) * contract * volume
    loss_value = abs(float(loss_value))
    profit_value = abs(float(profit_value))
    if not math.isfinite(loss_value) or not math.isfinite(profit_value):
        return None
    if loss_value <= 0 or profit_value <= 0:
        return None
    return loss_value, profit_value


def open_positions(terminal: Any, symbol: str, magic: int) -> list[Any]:
    """Return only this bot's open positions for the symbol."""
    module = get_terminal(terminal)
    positions = module.positions_get(symbol=symbol)
    if positions is None:
        return []
    return [
        position
        for position in positions
        if int(getattr(position, "magic", -1)) == magic
    ]


def daily_pnl(terminal: Any, magic: int, day: date) -> float:
    """Realized profit, commission, and swap for this bot on one UTC day."""
    module = get_terminal(terminal)
    start = datetime.combine(day, day_time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    # Query a wide window and filter on the deal's own UTC timestamp, so the
    # result does not depend on how the terminal interprets the range.
    deals = module.history_deals_get(
        start - timedelta(days=2), end + timedelta(days=2)
    )
    if not deals:
        return 0.0
    total = 0.0
    for deal in deals:
        if int(getattr(deal, "magic", 0)) != magic:
            continue
        when = datetime.fromtimestamp(int(deal.time), tz=timezone.utc)
        if not start <= when < end:
            continue
        total += float(deal.profit) + float(deal.commission) + float(deal.swap)
    return total


def new_deals(terminal: Any, magic: int, since_ticket: int) -> list[Any]:
    """Return this bot's deals newer than ``since_ticket``, oldest first."""
    module = get_terminal(terminal)
    now = datetime.now(timezone.utc)
    deals = module.history_deals_get(now - timedelta(days=7), now + timedelta(days=1))
    if not deals:
        return []
    fresh = [
        deal
        for deal in deals
        if int(getattr(deal, "magic", 0)) == magic
        and int(deal.ticket) > since_ticket
    ]
    return sorted(fresh, key=lambda deal: int(deal.ticket))


def resolve_filling(terminal: Any, info: Any, configured: str | None) -> int:
    """Pick a filling mode the symbol actually supports."""
    module = get_terminal(terminal)
    if configured:
        return int(getattr(module, f"ORDER_FILLING_{configured.upper()}"))
    flags = int(getattr(info, "filling_mode", 0) or 0)
    if flags & 2:
        return int(module.ORDER_FILLING_IOC)
    if flags & 1:
        return int(module.ORDER_FILLING_FOK)
    return int(module.ORDER_FILLING_RETURN)


def filling_candidates(terminal: Any, preferred: int) -> list[int]:
    module = get_terminal(terminal)
    order = [
        preferred,
        module.ORDER_FILLING_IOC,
        module.ORDER_FILLING_FOK,
        module.ORDER_FILLING_RETURN,
    ]
    unique: list[int] = []
    for mode in order:
        if mode not in unique:
            unique.append(mode)
    return unique


# ---------------------------------------------------------------------------
# Pending-execution state: survives a crash between order_send and evidence.
# ---------------------------------------------------------------------------


def _pending_path(config: DemoConfig, override: str | Path | None) -> Path:
    configured = (
        override if override is not None else config.execution.pending_state_path
    )
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def _pending_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=5)
    connection.execute(PENDING_SCHEMA)
    return connection


def _persist_pending(path: Path, pending: PendingExecution) -> None:
    with closing(_pending_db(path)) as connection:
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
                time.time(),
            ),
        )
        connection.commit()


def load_pending(path: Path, symbol: str) -> PendingExecution | None:
    with closing(_pending_db(path)) as connection:
        row = connection.execute(
            """
            SELECT symbol, order_id, deal_id, action, magic, requested_volume,
                   created_at, expires_at, state
            FROM pending_executions
            WHERE symbol = ? AND state IN ('PENDING', 'EXPIRED')
            """,
            (symbol,),
        ).fetchone()
    return PendingExecution(*row) if row else None


def _set_pending_state(
    path: Path, pending: PendingExecution, state: str
) -> PendingExecution:
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
    _persist_pending(path, updated)
    return updated


def _record_id(record: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = getattr(record, name, None)
        if value is not None:
            return int(value)
    return None


def _record_matches(record: Any, symbol: str, magic: int) -> bool:
    return (
        getattr(record, "symbol", None) == symbol
        and int(getattr(record, "magic", -1)) == magic
    )


def execution_evidence(
    terminal: Any,
    symbol: str,
    magic: int,
    order_id: int | None,
    deal_id: int | None,
) -> tuple[bool, Any, Any]:
    """Return (evidenced, position, deal) for one submitted order.

    A fill is only accepted when a position or deal exists that matches the
    symbol, this bot's magic number, and the order/deal ids returned by
    ``order_send``. Unrelated positions are never treated as fill evidence.
    """
    module = get_terminal(terminal)
    positions = module.positions_get(symbol=symbol) or []
    orders = module.history_orders_get(ticket=order_id) if order_id else None
    deals = module.history_deals_get(ticket=deal_id) if deal_id else None
    matching_order = next(
        (
            item
            for item in (orders or [])
            if _record_matches(item, symbol, magic)
            and _record_id(item, ("ticket", "order")) == order_id
        ),
        None,
    )
    matching_deal = next(
        (
            item
            for item in (deals or [])
            if _record_matches(item, symbol, magic)
            and _deal_related_to_order(item, order_id, deal_id)
        ),
        None,
    )
    related_position_ids = {
        _record_id(item, ("position_id", "position"))
        for item in (deals or [])
    }
    position = None
    for candidate in positions:
        if not _record_matches(candidate, symbol, magic):
            continue
        candidate_ids = {
            _record_id(candidate, ("ticket", "identifier", "position_id"))
        }
        if matching_deal is not None and candidate_ids & related_position_ids:
            position = candidate
            break
        if order_id is not None and _record_id(
            candidate, ("order", "order_id")
        ) == order_id:
            position = candidate
            break
    evidenced = matching_order is not None or matching_deal is not None
    return evidenced, position, matching_deal


def _deal_related_to_order(
    record: Any, order_id: int | None, deal_id: int | None
) -> bool:
    if deal_id is not None and _record_id(record, ("ticket", "deal")) != deal_id:
        return False
    related_order = _record_id(record, ("order", "order_id"))
    return order_id is None or related_order in {None, order_id}


def reconcile_pending(
    terminal: Any,
    config: DemoConfig,
    journal: Journal,
    pending_path: str | Path | None = None,
) -> str | None:
    """Reconcile one previously submitted order without ever resubmitting it."""
    path = _pending_path(config, pending_path)
    symbol = config.account.symbol
    pending = load_pending(path, symbol)
    if pending is None:
        return None
    if pending.state == "EXPIRED":
        journal.write(
            "reconciliation",
            symbol=symbol,
            status="expired",
            order_id=pending.order_id,
            deal_id=pending.deal_id,
            detail="expired execution requires manual reconciliation",
        )
        return "expired: confirmation timeout requires manual reconciliation"
    try:
        evidenced, position, deal = execution_evidence(
            terminal, symbol, pending.magic, pending.order_id, pending.deal_id
        )
    except Exception as exc:  # noqa: BLE001 - stay unconfirmed, never resubmit
        journal.write(
            "reconciliation",
            symbol=symbol,
            status="unconfirmed",
            order_id=pending.order_id,
            detail=f"evidence check failed: {type(exc).__name__}: {exc}",
        )
        return "unconfirmed: unable to verify execution evidence"
    if evidenced:
        _set_pending_state(path, pending, "FILLED")
        partial = (
            deal is not None
            and getattr(deal, "volume", None) is not None
            and float(deal.volume) < pending.requested_volume
        )
        journal.write(
            "reconciliation",
            symbol=symbol,
            status="filled",
            order_id=pending.order_id,
            deal_id=pending.deal_id,
            partial=partial,
            position=None if position is None else int(getattr(position, "ticket", 0)),
        )
        return "filled: order confirmed" + (" (partial)" if partial else "")
    if time.time() >= pending.expires_at:
        _set_pending_state(path, pending, "EXPIRED")
        journal.write(
            "reconciliation",
            symbol=symbol,
            status="expired",
            order_id=pending.order_id,
            deal_id=pending.deal_id,
            detail="confirmation timeout",
        )
        LOGGER.error(
            "Execution status=EXECUTION_EXPIRED order=%s deal=%s",
            pending.order_id,
            pending.deal_id,
        )
        return "expired: confirmation timeout requires manual reconciliation"
    journal.write(
        "reconciliation",
        symbol=symbol,
        status="unconfirmed",
        order_id=pending.order_id,
        deal_id=pending.deal_id,
    )
    LOGGER.warning(
        "Execution status=EXECUTION_UNCONFIRMED order=%s deal=%s",
        pending.order_id,
        pending.deal_id,
    )
    return "unconfirmed: still awaiting execution evidence"


# ---------------------------------------------------------------------------
# Entry validation, sizing, and submission.
# ---------------------------------------------------------------------------


def _blocked(
    journal: Journal,
    reason: str,
    signal: Any,
    entry: float,
    spread: float,
    market: MarketState,
) -> ExecutionResult:
    journal.write(
        "entry_blocked",
        reason=reason,
        action=getattr(signal, "action", None),
        entry=entry,
        spread_points=round(spread, 2),
        open_positions=market.open_positions,
        trades_today=market.trades_today,
        daily_pnl=market.daily_pnl,
    )
    return ExecutionResult(status="blocked", detail=reason, reason=reason)


def submit_signal(
    signal: Any,
    config: DemoConfig,
    journal: Journal,
    state: RuntimeState,
    terminal: Any = None,
    execute: bool = True,
    pending_path: str | Path | None = None,
) -> ExecutionResult:
    """Validate, size, and (optionally) send one demo order."""
    module = get_terminal(terminal)
    symbol = config.account.symbol
    execution = config.execution
    path = _pending_path(config, pending_path)
    if getattr(signal, "action", None) not in {"BUY", "SELL"}:
        journal.write(
            "signal_skipped",
            action=getattr(signal, "action", None),
            reason=getattr(signal, "rationale", ""),
        )
        return ExecutionResult(
            status="blocked",
            detail=f"action {getattr(signal, 'action', None)!r} is not executable",
            reason="not_a_trade",
        )

    pending = load_pending(path, symbol)
    if pending is not None:
        reason = (
            "expired_execution"
            if pending.state == "EXPIRED"
            else "pending_execution"
        )
        journal.write(
            "entry_blocked",
            reason=reason,
            order_id=pending.order_id,
            deal_id=pending.deal_id,
        )
        return ExecutionResult(status="blocked", detail=reason, reason=reason)

    account = require_demo_account(module)
    info = require_symbol(module, symbol)
    spec = spec_from_info(info)
    tick = module.symbol_info_tick(symbol)
    if tick is None:
        raise ExecutionError(
            f"unable to read a tick for {symbol!r}: {module.last_error()}"
        )

    entry = float(tick.ask if signal.action == "BUY" else tick.bid)
    limits = Limits(
        max_spread_points=execution.max_spread_points,
        max_trades_per_day=config.gate.max_trades_per_day,
        max_daily_loss_fraction=execution.max_daily_loss,
        max_open_positions=execution.max_open_positions,
        min_reward_risk=config.gate.min_rr,
    )
    positions = open_positions(module, symbol, execution.magic)
    spread = (float(tick.ask) - float(tick.bid)) / spec.point if spec.point > 0 else 0.0
    market = MarketState(
        balance=account.balance,
        spread_points=spread,
        open_positions=len(positions),
        trades_today=state.trades_today,
        daily_pnl=daily_pnl(module, execution.magic, datetime.now(timezone.utc).date()),
    )

    per_lot = 0.0
    volume = 0.0
    risk_amount = 0.0
    try:
        reason = entry_block_reason(
            signal.action,
            entry,
            signal.stop_loss,
            signal.take_profit,
            spec,
            limits,
            market,
        )
        if reason is None:
            if execution.volume_mode == "fixed":
                volume = config.gate.fixed_volume
                volume_error = validate_fixed_volume(info, volume)
                if volume_error:
                    reason = volume_error
                else:
                    amounts = money_amounts(
                        module,
                        symbol,
                        signal.action,
                        volume,
                        entry,
                        float(signal.stop_loss),
                        float(signal.take_profit),
                    )
                    if amounts is None:
                        reason = "unable_to_calculate_money_risk"
                    else:
                        risk_amount, _ = amounts
                        per_lot = risk_amount / volume
            else:
                per_lot, tick_estimate = loss_per_lot(
                    module, symbol, entry, float(signal.stop_loss), spec
                )
                if tick_estimate > 0 and abs(per_lot - tick_estimate) / per_lot > 0.2:
                    journal.write(
                        "sizing_warning",
                        order_calc_profit_per_lot=round(per_lot, 4),
                        tick_field_per_lot=round(tick_estimate, 4),
                        detail=(
                            "tick fields disagree with order_calc_profit; "
                            "using order_calc_profit"
                        ),
                    )
                volume = volume_for_risk(
                    account.balance,
                    execution.risk_fraction,
                    per_lot,
                    spec,
                    max_volume=execution.max_volume,
                )
                if volume <= 0:
                    reason = "risk_below_min_lot"
                else:
                    risk_amount = volume * per_lot
            if reason is None:
                max_risk = (
                    execution.max_risk_amount
                    if execution.max_risk_amount is not None
                    else account.balance * execution.risk_fraction
                )
                if risk_amount > max_risk + max(0.01, max_risk * 1e-9):
                    reason = (
                        "potential_loss_exceeds_risk_limit: "
                        f"{risk_amount:.2f} > {max_risk:.2f}"
                    )
    except RiskError as exc:
        journal.write(
            "signal_rejected",
            action=signal.action,
            reason=str(exc),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry=entry,
        )
        raise ExecutionError(f"signal rejected: {exc}") from exc

    if reason is not None:
        return _blocked(journal, reason, signal, entry, spread, market)

    reward_risk = abs(float(signal.take_profit) - entry) / abs(
        entry - float(signal.stop_loss)
    )
    request = {
        "action": module.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": module.ORDER_TYPE_BUY if signal.action == "BUY" else module.ORDER_TYPE_SELL,
        "price": entry,
        "sl": float(signal.stop_loss),
        "tp": float(signal.take_profit),
        "deviation": execution.deviation_points,
        "magic": execution.magic,
        "comment": (
            "test-bot-demo"
            if getattr(signal, "rationale", "") == "FORCED TEST TRADE"
            else "groq-demo"
        ),
        "type_time": module.ORDER_TIME_GTC,
    }

    if not execute:
        journal.write(
            "dry_run",
            action=signal.action,
            entry=entry,
            volume=volume,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_amount=round(risk_amount, 2),
            reward_risk=reward_risk,
            volume_mode=execution.volume_mode,
        )
        return ExecutionResult(
            status="dry_run", detail="execution disabled", volume=volume
        )

    if not bool(getattr(module.terminal_info(), "trade_allowed", True)):
        raise ExecutionError(
            "Algo Trading is off in the terminal; enable it before running"
        )

    result = None
    preferred = resolve_filling(module, info, execution.filling_mode)
    for filling in filling_candidates(module, preferred):
        request["type_filling"] = filling
        check = module.order_check(request)
        if check is None:
            raise ExecutionError(f"order_check failed: {module.last_error()}")
        if check.retcode != 0:
            if int(check.retcode) == int(
                getattr(module, "TRADE_RETCODE_INVALID_FILL", 10030)
            ):
                continue
            journal.write(
                "order_check_failed",
                retcode=check.retcode,
                comment=getattr(check, "comment", ""),
                request=request,
            )
            raise ExecutionError(
                f"order_check rejected the order: {getattr(check, 'comment', '')}"
            )
        # Another process may have opened a position between the checks.
        if open_positions(module, symbol, execution.magic):
            return _blocked(
                journal,
                "position_appeared_before_submission",
                signal,
                entry,
                spread,
                market,
            )
        result = module.order_send(request)
        LOGGER.info(
            "MT5 order_send result retcode=%s comment=%s order=%s deal=%s "
            "volume=%s price=%s",
            getattr(result, "retcode", None) if result is not None else None,
            getattr(result, "comment", "") if result is not None else "",
            getattr(result, "order", None) if result is not None else None,
            getattr(result, "deal", None) if result is not None else None,
            getattr(result, "volume", None) if result is not None else None,
            getattr(result, "price", None) if result is not None else None,
        )
        if result is not None and int(result.retcode) == int(
            getattr(module, "TRADE_RETCODE_INVALID_FILL", 10030)
        ):
            result = None
            continue
        break

    if result is None:
        raise ExecutionError("no supported filling mode was accepted by the broker")
    done = int(getattr(module, "TRADE_RETCODE_DONE", 10009))
    partial = int(getattr(module, "TRADE_RETCODE_DONE_PARTIAL", 10010))
    placed = int(getattr(module, "TRADE_RETCODE_PLACED", 10008))
    now = time.time()
    pending = PendingExecution(
        symbol=symbol,
        order_id=getattr(result, "order", None),
        deal_id=getattr(result, "deal", None),
        action=signal.action,
        magic=execution.magic,
        requested_volume=volume,
        created_at=now,
        expires_at=now + execution.pending_timeout_seconds,
    )

    def record_pending(status: str, detail: str) -> ExecutionResult:
        _persist_pending(path, pending)
        state.record_trade()
        journal.write(
            "order_sent",
            status=status,
            order_id=pending.order_id,
            deal_id=pending.deal_id,
            action=signal.action,
            symbol=symbol,
            volume=volume,
            entry=entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            magic=execution.magic,
            risk_amount=round(risk_amount, 2),
            rationale=getattr(signal, "rationale", ""),
        )
        return ExecutionResult(
            status=status,
            detail=detail,
            ticket=pending.order_id,
            deal=pending.deal_id,
            volume=volume,
        )

    if int(result.retcode) == placed:
        return record_pending(
            "placed", f"order {pending.order_id} accepted, awaiting fill evidence"
        )
    if int(result.retcode) not in {done, partial}:
        explanations = {
            10027: "AutoTrading disabled",
            10030: "unsupported or invalid filling mode",
            10016: "invalid stops",
            10014: "invalid volume",
        }
        LOGGER.error(
            "MT5 order_send rejected retcode=%s (%s) last_error=%s",
            result.retcode,
            explanations.get(int(result.retcode), "see MT5 documentation"),
            module.last_error(),
        )
        journal.write(
            "order_failed",
            retcode=result.retcode,
            comment=getattr(result, "comment", ""),
            request=request,
        )
        raise ExecutionError(
            f"order_send failed: retcode={result.retcode} "
            f"{getattr(result, 'comment', '')}"
        )

    try:
        evidenced, position, deal = execution_evidence(
            module, symbol, execution.magic, pending.order_id, pending.deal_id
        )
    except Exception as exc:  # noqa: BLE001 - never claim a fill we cannot prove
        LOGGER.error(
            "Execution status=EXECUTION_UNCONFIRMED evidence_error=%s", exc
        )
        return record_pending(
            "unconfirmed", f"unable to verify execution: {exc}"
        )
    if not evidenced:
        LOGGER.error(
            "Execution status=EXECUTION_UNCONFIRMED order=%s deal=%s",
            pending.order_id,
            pending.deal_id,
        )
        return record_pending(
            "unconfirmed", "no matching position or history evidence"
        )

    state.record_trade()
    execution_price = getattr(result, "price", None)
    if execution_price is None and position is not None:
        execution_price = getattr(position, "price_open", None)
    if execution_price is not None and spec.point > 0:
        slippage_points = abs(float(execution_price) - entry) / spec.point
        journal.write(
            "slippage",
            requested=entry,
            actual=float(execution_price),
            points=round(slippage_points, 2),
            limit_points=execution.deviation_points,
            exceeded=slippage_points > execution.deviation_points,
        )
    status = "filled(partial)" if int(result.retcode) == partial else "filled"
    journal.write(
        "order_sent",
        status=status,
        order_id=pending.order_id,
        deal_id=pending.deal_id,
        position=None
        if position is None
        else int(getattr(position, "ticket", 0)),
        action=signal.action,
        symbol=symbol,
        volume=volume,
        entry=float(execution_price) if execution_price is not None else entry,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        magic=execution.magic,
        risk_amount=round(risk_amount, 2),
        rationale=getattr(signal, "rationale", ""),
    )
    return ExecutionResult(
        status=status,
        detail=f"order {pending.order_id} deal {pending.deal_id}",
        ticket=pending.order_id,
        deal=pending.deal_id,
        volume=volume,
    )


def is_live_trading_enabled() -> bool:
    """The second safety switch, on top of the demo-account check.

    Only the demo switch exists: there is no environment variable that turns a
    real account into a tradeable one.
    """
    return os.environ.get("ENABLE_DEMO_TRADING") == "1"
