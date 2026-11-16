"""Typed loading for the demo runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "demo.yaml"
FILLING_MODES = {"ioc", "fok", "return"}
VOLUME_MODES = {"risk", "fixed"}
# Configured timeframe name -> candle size in minutes.
TIMEFRAMES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}


class ConfigError(ValueError):
    """Raised when the demo configuration cannot be used safely."""


@dataclass(frozen=True)
class AccountConfig:
    symbol: str
    timeframe: str


@dataclass(frozen=True)
class ExecutionConfig:
    magic: int
    risk_fraction: float
    max_daily_loss: float
    max_trades_per_day: int
    max_open_positions: int
    max_spread_points: float
    min_reward_risk: float
    deviation_points: int
    filling_mode: str | None
    max_volume: float
    volume_mode: str = "risk"
    fixed_volume: float = 0.02
    max_risk_amount: float | None = None
    pending_state_path: str = "logs/pending_executions.sqlite3"
    pending_timeout_seconds: int = 300


@dataclass(frozen=True)
class MonitorConfig:
    bars: int
    poll_seconds: int
    max_signal_attempts: int
    max_backoff_seconds: int
    csv_log: str = "logs/monitoring.csv"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    context_bars: int


@dataclass(frozen=True)
class DemoConfig:
    account: AccountConfig
    execution: ExecutionConfig
    monitor: MonitorConfig
    model: ModelConfig


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"config section '{name}' is missing or not a mapping")
    return section


def _number(section: dict[str, Any], name: str, where: str) -> float:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{name} must be a number")
    return float(value)


def _integer(section: dict[str, Any], name: str, where: str) -> int:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{name} must be an integer")
    return value


def _fraction(section: dict[str, Any], name: str, where: str, upper: float) -> float:
    value = _number(section, name, where)
    if not 0 < value <= upper:
        raise ConfigError(f"{where}.{name} must be between 0 and {upper}")
    return value


def parse_config(config: dict[str, Any]) -> DemoConfig:
    """Validate one configuration mapping and return typed settings."""
    if not isinstance(config, dict):
        raise ConfigError("configuration must be a mapping")

    account_section = _section(config, "account")
    symbol = account_section.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ConfigError("account.symbol must be a non-empty string")
    timeframe = account_section.get("timeframe")
    if timeframe not in TIMEFRAMES:
        raise ConfigError(f"account.timeframe must be one of {sorted(TIMEFRAMES)}")

    execution = _section(config, "execution")
    filling_mode = execution.get("filling_mode")
    if filling_mode is not None and filling_mode not in FILLING_MODES:
        raise ConfigError(
            f"execution.filling_mode must be one of {sorted(FILLING_MODES)} or null"
        )
    magic = _integer(execution, "magic", "execution")
    if not 0 < magic < 2**31:
        raise ConfigError("execution.magic must be a positive 32-bit integer")

    monitor = _section(config, "monitor")
    poll_seconds = _integer(monitor, "poll_seconds", "monitor")
    if poll_seconds < 5:
        raise ConfigError("monitor.poll_seconds must be at least 5")

    max_volume = _number(execution, "max_volume", "execution")
    if max_volume <= 0:
        raise ConfigError("execution.max_volume must be positive")

    volume_mode = execution.get("volume_mode", "risk")
    if not isinstance(volume_mode, str) or volume_mode.strip().lower() not in VOLUME_MODES:
        raise ConfigError(
            f"execution.volume_mode must be one of {sorted(VOLUME_MODES)}"
        )
    volume_mode = volume_mode.strip().lower()

    fixed_volume = execution.get("fixed_volume", 0.02)
    if isinstance(fixed_volume, bool) or not isinstance(fixed_volume, (int, float)):
        raise ConfigError("execution.fixed_volume must be a number")
    fixed_volume = float(fixed_volume)
    if fixed_volume <= 0:
        raise ConfigError("execution.fixed_volume must be positive")

    max_risk_amount = execution.get("max_risk_amount")
    if max_risk_amount is not None:
        if isinstance(max_risk_amount, bool) or not isinstance(
            max_risk_amount, (int, float)
        ):
            raise ConfigError("execution.max_risk_amount must be a number or null")
        max_risk_amount = float(max_risk_amount)
        if max_risk_amount <= 0:
            raise ConfigError("execution.max_risk_amount must be positive or null")

    pending_state_path = execution.get(
        "pending_state_path", "logs/pending_executions.sqlite3"
    )
    if not isinstance(pending_state_path, str) or not pending_state_path.strip():
        raise ConfigError("execution.pending_state_path must be a non-empty string")

    pending_timeout_seconds = execution.get("pending_timeout_seconds", 300)
    if isinstance(pending_timeout_seconds, bool) or not isinstance(
        pending_timeout_seconds, int
    ):
        raise ConfigError("execution.pending_timeout_seconds must be an integer")
    if pending_timeout_seconds <= 0:
        raise ConfigError("execution.pending_timeout_seconds must be positive")

    csv_log = monitor.get("csv_log", "logs/monitoring.csv")
    if not isinstance(csv_log, str) or not csv_log.strip():
        raise ConfigError("monitor.csv_log must be a non-empty string")

    model = _section(config, "model")
    name = model.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("model.name must be a non-empty string")

    return DemoConfig(
        account=AccountConfig(symbol=symbol.strip(), timeframe=timeframe),
        execution=ExecutionConfig(
            magic=magic,
            risk_fraction=_fraction(execution, "risk_per_trade", "execution", 0.05),
            max_daily_loss=_fraction(execution, "max_daily_loss", "execution", 1.0),
            max_trades_per_day=_integer(execution, "max_trades_per_day", "execution"),
            max_open_positions=_integer(execution, "max_open_positions", "execution"),
            max_spread_points=_number(execution, "max_spread_points", "execution"),
            min_reward_risk=_number(execution, "min_reward_risk", "execution"),
            deviation_points=_integer(execution, "deviation_points", "execution"),
            filling_mode=filling_mode,
            max_volume=max_volume,
            volume_mode=volume_mode,
            fixed_volume=fixed_volume,
            max_risk_amount=max_risk_amount,
            pending_state_path=pending_state_path.strip(),
            pending_timeout_seconds=pending_timeout_seconds,
        ),
        monitor=MonitorConfig(
            bars=_integer(monitor, "bars", "monitor"),
            poll_seconds=poll_seconds,
            max_signal_attempts=_integer(monitor, "max_signal_attempts", "monitor"),
            max_backoff_seconds=_integer(monitor, "max_backoff_seconds", "monitor"),
            csv_log=csv_log.strip(),
        ),
        model=ModelConfig(
            name=name.strip(),
            context_bars=_integer(model, "context_bars", "model"),
        ),
    )


def load_config(path: str | Path = DEFAULT_PATH) -> DemoConfig:
    """Load and validate the demo configuration from YAML."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ConfigError(f"{path} is empty")
    return parse_config(raw)
