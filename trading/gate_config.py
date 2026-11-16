"""Pure entry-gate configuration and evaluation.

The gate is deliberately independent from MetaTrader 5 and Groq.  It is the
last local decision before execution and is therefore the single owner of the
minimum reward/risk requirement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateConfig:
    min_score: float = 70.0
    min_score_gap: float = 30.0
    sl_atr_mult: float = 1.5
    min_rr: float = 2.0
    fixed_volume: float = 0.01
    max_daily_loss_usd: float = 100.0
    max_consecutive_losses: int = 2
    max_trades_per_day: int = 3
    max_total_loss_usd: float = 500.0
    session_enabled: bool = False
    session_utc_start: str = "00:00"
    session_utc_end: str = "23:59"
    max_spread: float = 0.50

    def session_bounds(self) -> tuple[time, time]:
        try:
            return (
                time.fromisoformat(self.session_utc_start),
                time.fromisoformat(self.session_utc_end),
            )
        except ValueError as exc:
            raise ValueError("session UTC bounds must use HH:MM") from exc


@dataclass(frozen=True)
class GateMarketState:
    spread: float
    utc_now: datetime
    candle_id: str
    local_direction: str | None = None


@dataclass(frozen=True)
class GateAccountState:
    daily_loss_usd: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    total_loss_usd: float = 0.0
    open_positions: int = 0
    entered_candle: str | None = None
    kill_switch: bool = False


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failed_rules: tuple[str, ...] = ()


def _score(signal: Any, action: str) -> float:
    return float(
        getattr(signal, "local_buy_score" if action == "BUY" else "local_sell_score", 0.0)
    )


def _in_session(config: GateConfig, value: datetime) -> bool:
    start, end = config.session_bounds()
    current = value.time()
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def evaluate_entry_gate(
    signal: Any,
    market_state: GateMarketState,
    account_state: GateAccountState,
    config: GateConfig | None = None,
) -> GateResult:
    """Evaluate every local entry rule without side effects."""
    settings = config or GateConfig()
    failed: list[str] = []
    action = getattr(signal, "action", "")
    opposite = "SELL" if action == "BUY" else "BUY"
    chosen = _score(signal, action)
    other = _score(signal, opposite)

    if action not in {"BUY", "SELL"}:
        failed.append(f"action {action!r} is not BUY or SELL")
    if chosen < settings.min_score:
        failed.append(f"local_score {chosen:g} < {settings.min_score:g}")
    if chosen - other < settings.min_score_gap:
        failed.append(
            f"score_gap {chosen - other:g} < {settings.min_score_gap:g}"
        )
    if market_state.local_direction and action != market_state.local_direction:
        failed.append(
            f"direction {action} != local_direction {market_state.local_direction}"
        )

    rr = getattr(signal, "risk_reward", None)
    if rr is None or float(rr) < settings.min_rr:
        failed.append(f"risk_reward {rr!r} < {settings.min_rr:g}")
    if settings.session_enabled and not _in_session(settings, market_state.utc_now):
        failed.append("outside_enabled_session")
    if market_state.spread > settings.max_spread:
        failed.append(f"spread {market_state.spread:g} > {settings.max_spread:g}")
    if account_state.daily_loss_usd >= settings.max_daily_loss_usd:
        failed.append("max_daily_loss_usd")
    if account_state.consecutive_losses >= settings.max_consecutive_losses:
        failed.append("max_consecutive_losses")
    if account_state.trades_today >= settings.max_trades_per_day:
        failed.append("max_trades_per_day")
    if account_state.total_loss_usd >= settings.max_total_loss_usd:
        failed.append("max_total_loss_usd")
    if account_state.kill_switch:
        failed.append("kill_switch_active")
    if account_state.open_positions > 0:
        failed.append("position_already_open")
    if account_state.entered_candle == market_state.candle_id:
        failed.append("candle_already_entered")
    return GateResult(not failed, tuple(failed))


def log_gate_result(result: GateResult) -> None:
    """Log every failed rule for the current monitoring cycle."""
    if result.failed_rules:
        LOGGER.info("REJECTED: %s", "; ".join(result.failed_rules))
