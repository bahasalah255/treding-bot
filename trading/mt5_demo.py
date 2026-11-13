"""Guarded MetaTrader 5 demo execution.

The only account this module will ever trade is an account whose
``trade_mode`` is DEMO. Every other account is refused before any request is
built.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
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
    volume_for_risk,
)
from trading.state import RuntimeState


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
    volume: float | None = None


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
    """Initialize the terminal for a short-lived read-only command."""
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


def require_demo_account(terminal: Any = None) -> AccountSnapshot:
    """Return the account only when it is a demo account with algo trading on."""
    module = get_terminal(terminal)
    account = module.account_info()
    if account is None:
        raise ExecutionError(f"unable to read account information: {module.last_error()}")
    demo_mode = getattr(module, "ACCOUNT_TRADE_MODE_DEMO", 0)
    if int(getattr(account, "trade_mode", -1)) != int(demo_mode):
        raise ExecutionError(
            f"refusing to trade account {getattr(account, 'login', '?')}: "
            f"trade_mode {getattr(account, 'trade_mode', '?')} is not a demo account"
        )
    if not bool(getattr(account, "trade_expert", True)):
        raise ExecutionError(
            "algo trading is disabled for this account; enable Algo Trading in the terminal"
        )
    return AccountSnapshot(
        login=int(getattr(account, "login", 0)),
        server=str(getattr(account, "server", "")),
        currency=str(getattr(account, "currency", "")),
        balance=float(getattr(account, "balance", 0.0)),
        equity=float(getattr(account, "equity", 0.0)),
        trade_mode=int(getattr(account, "trade_mode", -1)),
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
        raise ExecutionError(f"unable to read symbol {symbol!r}: {module.last_error()}")
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
    )


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


def open_positions(terminal: Any, symbol: str, magic: int) -> list[Any]:
    """Return only this bot's open positions for the symbol."""
    module = get_terminal(terminal)
    positions = module.positions_get(symbol=symbol)
    if positions is None:
        return []
    return [position for position in positions if int(position.magic) == magic]


def daily_pnl(terminal: Any, magic: int, day: date) -> float:
    """Realized profit, commission, and swap for this bot on one UTC day."""
    module = get_terminal(terminal)
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    # Query a wide window and filter on the deal's own UTC timestamp, so the
    # result does not depend on how the terminal interprets the range.
    deals = module.history_deals_get(start - timedelta(days=2), end + timedelta(days=2))
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
        if int(getattr(deal, "magic", 0)) == magic and int(deal.ticket) > since_ticket
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
    order = [preferred, module.ORDER_FILLING_IOC, module.ORDER_FILLING_FOK, module.ORDER_FILLING_RETURN]
    unique: list[int] = []
    for mode in order:
        if mode not in unique:
            unique.append(mode)
    return unique


def submit_signal(
    signal: Any,
    config: DemoConfig,
    journal: Journal,
    state: RuntimeState,
    terminal: Any = None,
    execute: bool = True,
) -> ExecutionResult:
    """Validate, size, and (optionally) send one demo order."""
    module = get_terminal(terminal)
    symbol = config.account.symbol
    account = require_demo_account(module)
    info = require_symbol(module, symbol)
    spec = spec_from_info(info)
    tick = module.symbol_info_tick(symbol)
    if tick is None:
        raise ExecutionError(f"unable to read a tick for {symbol!r}: {module.last_error()}")

    entry = float(tick.ask if signal.action == "BUY" else tick.bid)
    execution = config.execution
    limits = Limits(
        max_spread_points=execution.max_spread_points,
        max_trades_per_day=execution.max_trades_per_day,
        max_daily_loss_fraction=execution.max_daily_loss,
        max_open_positions=execution.max_open_positions,
        min_reward_risk=execution.min_reward_risk,
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

    try:
        reason = entry_block_reason(
            signal.action, entry, signal.stop_loss, signal.take_profit, spec, limits, market
        )
        volume = 0.0
        per_lot = 0.0
        if reason is None:
            per_lot, tick_estimate = loss_per_lot(
                module, symbol, entry, float(signal.stop_loss), spec
            )
            if tick_estimate > 0 and abs(per_lot - tick_estimate) / per_lot > 0.2:
                journal.write(
                    "sizing_warning",
                    order_calc_profit_per_lot=round(per_lot, 4),
                    tick_field_per_lot=round(tick_estimate, 4),
                    detail="tick fields disagree with order_calc_profit; using order_calc_profit",
                )
            volume = volume_for_risk(
                account.balance, execution.risk_fraction, per_lot, spec
            )
            if volume <= 0:
                reason = "risk_below_min_lot"
    except RiskError as exc:
        journal.write(
            "signal_rejected", action=signal.action, reason=str(exc),
            stop_loss=signal.stop_loss, take_profit=signal.take_profit, entry=entry,
        )
        raise ExecutionError(f"signal rejected: {exc}") from exc

    if reason is not None:
        journal.write(
            "entry_blocked", reason=reason, action=signal.action, entry=entry,
            spread_points=round(spread, 2), open_positions=market.open_positions,
            trades_today=market.trades_today, daily_pnl=market.daily_pnl,
        )
        return ExecutionResult(status="blocked", detail=reason, reason=reason)

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
        "comment": "groq-demo",
        "type_time": module.ORDER_TIME_GTC,
    }

    if not execute:
        journal.write(
            "dry_run", action=signal.action, entry=entry, volume=volume,
            stop_loss=signal.stop_loss, take_profit=signal.take_profit,
            reward_risk=abs(float(signal.take_profit) - entry)
            / abs(entry - float(signal.stop_loss)),
        )
        return ExecutionResult(status="dry_run", detail="execution disabled", volume=volume)

    if not bool(getattr(module.terminal_info(), "trade_allowed", True)):
        raise ExecutionError("Algo Trading is off in the terminal; enable it before running")

    result = None
    for filling in filling_candidates(module, resolve_filling(module, info, execution.filling_mode)):
        request["type_filling"] = filling
        check = module.order_check(request)
        if check is None:
            raise ExecutionError(f"order_check failed: {module.last_error()}")
        if check.retcode != 0:
            if int(check.retcode) == int(getattr(module, "TRADE_RETCODE_INVALID_FILL", 10030)):
                continue
            journal.write(
                "order_check_failed", retcode=check.retcode,
                comment=getattr(check, "comment", ""), request=request,
            )
            raise ExecutionError(f"order_check rejected the order: {check.comment}")
        result = module.order_send(request)
        if result is not None and int(result.retcode) == int(
            getattr(module, "TRADE_RETCODE_INVALID_FILL", 10030)
        ):
            result = None
            continue
        break

    if result is None:
        raise ExecutionError("no supported filling mode was accepted by the broker")
    if int(result.retcode) != int(module.TRADE_RETCODE_DONE):
        journal.write(
            "order_failed", retcode=result.retcode, comment=getattr(result, "comment", ""),
            request=request,
        )
        raise ExecutionError(
            f"order_send failed: retcode={result.retcode} {getattr(result, 'comment', '')}"
        )

    state.record_trade()
    journal.write(
        "order_sent", ticket=int(result.order), deal=int(getattr(result, "deal", 0)),
        action=signal.action, symbol=symbol, volume=volume, entry=float(result.price),
        stop_loss=float(signal.stop_loss), take_profit=float(signal.take_profit),
        magic=execution.magic, rationale=getattr(signal, "rationale", ""),
        balance=account.balance, risk_amount=round(volume * per_lot, 2),
    )
    return ExecutionResult(
        status="filled", detail=f"order {result.order}", ticket=int(result.order), volume=volume
    )


def is_live_trading_enabled() -> bool:
    """The second safety switch, on top of the demo-account check."""
    return os.environ.get("ENABLE_DEMO_TRADING") == "1"
