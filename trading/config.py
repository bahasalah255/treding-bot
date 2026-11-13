"""Typed loading for the demo runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "demo.yaml"
FILLING_MODES = {"ioc", "fok", "return"}
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


@dataclass(frozen=True)
class MonitorConfig:
    bars: int
    poll_seconds: int
    max_signal_attempts: int
    max_backoff_seconds: int


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
        ),
        monitor=MonitorConfig(
            bars=_integer(monitor, "bars", "monitor"),
            poll_seconds=poll_seconds,
            max_signal_attempts=_integer(monitor, "max_signal_attempts", "monitor"),
            max_backoff_seconds=_integer(monitor, "max_backoff_seconds", "monitor"),
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
