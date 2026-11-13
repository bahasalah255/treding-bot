"""Pure risk, sizing, and entry-gating logic for demo execution.

Everything in this module is free of MetaTrader 5 imports so it can be tested
without a terminal attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class RiskError(ValueError):
    """Raised when a trade proposal cannot be sized or validated safely."""


@dataclass(frozen=True)
class SymbolSpec:
    """The contract fields needed to size and validate a trade.

    Tick value is deliberately absent: some terminals report a stale or
    converted ``trade_tick_value``. The executor derives the money loss per lot
    from the terminal's own ``order_calc_profit`` instead, which is
    authoritative for the account's currency and contract size.
    """

    point: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: int = 0
    freeze_level_points: int = 0

    def min_stop_distance(self) -> float:
        """Smallest distance the broker allows between price and SL/TP."""
        levels = max(self.stops_level_points, self.freeze_level_points)
        return levels * self.point


@dataclass(frozen=True)
class Limits:
    """Hard caps applied before any order is considered."""

    max_spread_points: float
    max_trades_per_day: int
    max_daily_loss_fraction: float
    max_open_positions: int = 1
    min_reward_risk: float = 0.0


@dataclass(frozen=True)
class MarketState:
    """Values read from the terminal at the moment of the decision."""

    balance: float
    spread_points: float
    open_positions: int
    trades_today: int
    daily_pnl: float


def round_volume(volume: float, spec: SymbolSpec) -> float:
    """Snap a lot size down to a valid step, or return 0.0 when it is too small."""
    if spec.volume_step <= 0:
        raise RiskError("volume_step must be positive")
    if not math.isfinite(volume) or volume < spec.volume_min:
        return 0.0
    capped = min(volume, spec.volume_max)
    steps = math.floor(capped / spec.volume_step + 1e-9)
    rounded = round(steps * spec.volume_step, 8)
    if rounded < spec.volume_min:
        return 0.0
    return rounded


def volume_for_risk(
    balance: float,
    risk_fraction: float,
    loss_per_lot: float,
    spec: SymbolSpec,
    max_volume: float | None = None,
) -> float:
    """Return the lot size whose loss at the stop is ``risk_fraction`` of balance."""
    if not 0 < risk_fraction <= 1:
        raise RiskError("risk_fraction must be between 0 and 1")
    if balance <= 0:
        raise RiskError("balance must be positive")
    if loss_per_lot <= 0:
        raise RiskError("loss per lot must be positive")
    volume = balance * risk_fraction / loss_per_lot
    if max_volume is not None:
        if max_volume <= 0:
            raise RiskError("max_volume must be positive")
        volume = min(volume, max_volume)
    return round_volume(volume, spec)


def market_level_error(
    action: str,
    entry: float,
    stop_loss: float | None,
    take_profit: float | None,
    spec: SymbolSpec,
) -> str | None:
    """Return a human-readable broker-level violation, or ``None`` when valid.

    This mirrors :func:`validate_proposal` but reports instead of raising, which
    keeps a plain-text execution path (and its tests) easy to reason about.
    """
    if action not in {"BUY", "SELL"}:
        return f"unsupported action: {action!r}"
    if stop_loss is None or take_profit is None:
        return "stop_loss and take_profit are required"
    for name, value in (
        ("entry", entry),
        ("stop_loss", stop_loss),
        ("take_profit", take_profit),
    ):
        if value is None or not math.isfinite(value) or value <= 0:
            return f"{name} must be finite and positive"
    if action == "BUY" and not stop_loss < entry < take_profit:
        return "BUY requires stop_loss < entry < take_profit"
    if action == "SELL" and not take_profit < entry < stop_loss:
        return "SELL requires take_profit < entry < stop_loss"
    minimum = spec.min_stop_distance()
    if action == "BUY":
        distances = (entry - stop_loss, take_profit - entry)
    else:
        distances = (stop_loss - entry, entry - take_profit)
    if minimum > 0 and any(distance < minimum for distance in distances):
        return f"SL/TP violates minimum stop or freeze distance {minimum:.5f}"
    return None


def validate_proposal(
    action: str,
    entry: float,
    stop_loss: float | None,
    take_profit: float | None,
    spec: SymbolSpec,
    min_reward_risk: float = 0.0,
) -> tuple[float, float]:
    """Check a signal against broker constraints and return (risk, reward).

    A malformed or unsafe signal raises instead of being silently adjusted, so
    the journal keeps a truthful record of what was refused and why.
    """
    if action not in {"BUY", "SELL"}:
        raise RiskError(f"unsupported action: {action!r}")
    if stop_loss is None or take_profit is None:
        raise RiskError("stop_loss and take_profit are required")
    prices = (entry, stop_loss, take_profit)
    if not all(math.isfinite(value) and value > 0 for value in prices):
        raise RiskError("prices must be finite and positive")

    if action == "BUY":
        risk = entry - stop_loss
        reward = take_profit - entry
    else:
        risk = stop_loss - entry
        reward = entry - take_profit
    if risk <= 0:
        raise RiskError(f"{action} stop loss is on the wrong side of the entry")
    if reward <= 0:
        raise RiskError(f"{action} take profit is on the wrong side of the entry")

    minimum = spec.min_stop_distance()
    if minimum > 0:
        distances = (risk, reward)
        if any(distance < minimum for distance in distances):
            raise RiskError(
                f"SL/TP distance is below the broker minimum {minimum:.5f}"
            )
    if min_reward_risk > 0 and reward / risk < min_reward_risk:
        raise RiskError(
            f"reward:risk {reward / risk:.2f} is below the required {min_reward_risk:.2f}"
        )
    return risk, reward


def entry_block_reason(
    action: str,
    entry: float,
    stop_loss: float | None,
    take_profit: float | None,
    spec: SymbolSpec,
    limits: Limits,
    state: MarketState,
) -> str | None:
    """Return why a valid signal must not be traded, or ``None`` to go ahead.

    Invalid signals raise :class:`RiskError`; this function only reports
    market and account conditions, which are normal and expected.
    """
    validate_proposal(
        action, entry, stop_loss, take_profit, spec, limits.min_reward_risk
    )
    if state.spread_points > limits.max_spread_points:
        return "spread_too_wide"
    if state.open_positions >= limits.max_open_positions:
        return "position_already_open"
    if state.trades_today >= limits.max_trades_per_day:
        return "max_trades_per_day"
    if state.balance > 0 and state.daily_pnl <= -abs(
        state.balance * limits.max_daily_loss_fraction
    ):
        return "daily_loss_limit"
    return None
