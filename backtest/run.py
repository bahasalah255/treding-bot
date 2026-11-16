"""Run the initial XAUUSD MTF EMA backtest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from backtest.ema_mtf import build_signals
from backtest.engine import run_backtest
from data.validate import load_and_validate_csv


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML backtest configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} did not contain a configuration mapping")
    return config


def price_costs(costs: dict[str, Any]) -> dict[str, float]:
    """Convert point- and lot-denominated cost settings into price units.

    The engine charges costs in price, so anything that cannot be converted
    without broker contract specifications is rejected instead of being
    silently ignored.
    """
    price_per_point = costs.get("price_per_point")
    spread_points = float(costs.get("spread_points") or 0.0)
    slippage_points = float(costs.get("slippage_points") or 0.0)
    commission = float(costs.get("commission_per_lot_mad") or 0.0)

    if commission:
        raise ValueError(
            "commission_per_lot_mad cannot be converted to price units without "
            "the contract size and lot value; set it to 0 until that is recorded"
        )
    if (spread_points or slippage_points) and not price_per_point:
        raise ValueError(
            "costs.price_per_point must be set before point-based spread or "
            "slippage can be modelled (record it in data/metadata/ first)"
        )

    per_point = float(price_per_point or 0.0)
    return {
        "spread_price": spread_points * per_point,
        "slippage_price": slippage_points * per_point,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "default.yaml")
    capital = config["capital"]
    strategy = config["strategy"]

    m5 = load_and_validate_csv(root / "data" / "raw" / "XAUUSD_M5.csv")
    m15 = load_and_validate_csv(root / "data" / "raw" / "XAUUSD_M15.csv")
    signals = build_signals(
        m5,
        m15,
        fast_period=int(strategy["fast_ema"]),
        slow_period=int(strategy["slow_ema"]),
        trend_period=int(strategy["trend_ema"]),
    )
    report = run_backtest(
        signals,
        risk_percent=float(capital["risk_per_trade"]) * 100,
        reward_risk=float(strategy["reward_risk"]),
        stop_distance_atr=float(strategy["stop_distance_atr"]),
        atr_period=int(strategy["atr_period"]),
        initial_balance=float(capital["initial_mad"]),
        max_daily_loss=float(capital["max_daily_loss"]),
        max_total_drawdown=float(capital["max_total_drawdown"]),
        **price_costs(config["costs"]),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
