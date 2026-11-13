"""Read-only checks to run before letting the demo bot trade.

This command never sends an order. It launches the MT5 terminal if it is not
already running, reads the account, symbol, and contract details, and prints a
checklist.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from trading.config import DEFAULT_PATH, ConfigError, load_config


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  PASS  {label}" + (f"  [{detail}]" if detail else ""))

    def warn(self, label: str, detail: str = "") -> None:
        self.warnings += 1
        print(f"  WARN  {label}" + (f"  [{detail}]" if detail else ""))

    def fail(self, label: str, detail: str = "") -> None:
        self.failures += 1
        print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        if condition:
            self.ok(label, detail)
        else:
            self.fail(label, detail)
        return condition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only preflight for the demo bot")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    args = parser.parse_args(argv)
    report = Report()

    print("Configuration")
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        report.fail("config loads", str(exc))
        return 1
    report.ok("config loads", str(Path(args.config)))
    report.ok("symbol", config.account.symbol)
    report.ok("timeframe", config.account.timeframe)
    report.ok(
        "risk per trade", f"{config.execution.risk_fraction * 100:.2f}% of balance"
    )
    report.ok("magic", str(config.execution.magic))

    print("Terminal")
    try:
        from trading.mt5_demo import (
            ExecutionError,
            daily_pnl,
            open_positions,
            spec_from_info,
            session,
        )
    except ImportError as exc:
        report.fail("MetaTrader5 package", str(exc))
        return 1

    try:
        with session() as terminal:
            info = terminal.terminal_info()
            report.ok(
                "terminal connected",
                f"{getattr(info, 'company', '')} build {getattr(info, 'build', 0)}",
            )
            if not report.check(
                bool(getattr(info, "trade_allowed", False)),
                "Algo Trading enabled in terminal",
            ):
                print("        enable the Algo Trading button in the MT5 toolbar")
            report.check(
                bool(getattr(info, "connected", True)), "terminal is connected"
            )

            account = terminal.account_info()
            if account is None:
                report.fail("account readable", str(terminal.last_error()))
                return 1
            mode = int(getattr(account, "trade_mode", -1))
            demo_mode = int(getattr(terminal, "ACCOUNT_TRADE_MODE_DEMO", 0))
            if not report.check(
                mode == demo_mode,
                "account is DEMO",
                f"login {account.login} @ {account.server}, mode={mode}",
            ):
                print("        the bot refuses to trade a non-demo account")

            print(
                f"        balance {account.balance:.2f} {account.currency}, "
                f"equity {account.equity:.2f}, leverage 1:{account.leverage}"
            )
            report.check(
                bool(getattr(account, "trade_expert", False)),
                "expert/algo trading allowed for the account",
            )

            symbol = config.account.symbol
            from trading.mt5_demo import require_symbol

            try:
                symbol_info = require_symbol(terminal, symbol)
            except ExecutionError as exc:
                report.fail("symbol available", str(exc))
                return 1
            report.ok(
                "symbol available",
                f"{symbol_info.name} path={symbol_info.path}",
            )

            spec = spec_from_info(symbol_info)
            report.check(spec.point > 0, "point size", f"{spec.point}")
            report.check(spec.volume_step > 0, "volume step", f"{spec.volume_step}")
            from trading.mt5_demo import resolve_filling

            flags = int(getattr(symbol_info, "filling_mode", 0) or 0)
            chosen = resolve_filling(terminal, symbol_info, config.execution.filling_mode)
            names = {
                getattr(terminal, "ORDER_FILLING_IOC", -1): "IOC",
                getattr(terminal, "ORDER_FILLING_FOK", -2): "FOK",
                getattr(terminal, "ORDER_FILLING_RETURN", -3): "RETURN",
            }
            report.ok(
                "filling mode",
                f"symbol flags {flags}, using {names.get(chosen, chosen)}",
            )
            print(
                f"        contract size {getattr(symbol_info, 'trade_contract_size', 0)}, "
                f"tick size {getattr(symbol_info, 'trade_tick_size', 0)}, "
                f"reported tick value {getattr(symbol_info, 'trade_tick_value', 0)}"
            )
            print(
                f"        volume min/max {spec.volume_min}/{spec.volume_max}, "
                f"stops level {spec.stops_level_points} points, "
                f"freeze level {spec.freeze_level_points} points "
                f"(minimum distance {spec.min_stop_distance():.5f} price)"
            )

            tick = terminal.symbol_info_tick(symbol)
            if tick is None:
                report.fail("tick readable", str(terminal.last_error()))
                return 1
            spread = (float(tick.ask) - float(tick.bid)) / spec.point
            if spread <= config.execution.max_spread_points:
                report.ok(
                    "spread within limit",
                    f"{spread:.1f} <= {config.execution.max_spread_points:.0f} points",
                )
            else:
                report.warn(
                    "spread above limit",
                    f"{spread:.1f} > {config.execution.max_spread_points:.0f} points "
                    "(entries will be skipped)",
                )

            positions = open_positions(terminal, symbol, config.execution.magic)
            report.ok("open bot positions", str(len(positions)))

            from trading.mt5_demo import loss_per_lot
            from trading.risk import volume_for_risk

            probe_distance = max(spec.min_stop_distance(), 1.0)
            per_lot, tick_estimate = loss_per_lot(
                terminal, symbol, float(tick.ask), float(tick.ask) - probe_distance, spec
            )
            if tick_estimate > 0 and abs(per_lot - tick_estimate) / per_lot > 0.05:
                report.warn(
                    "sizing source disagrees with tick fields",
                    f"order_calc_profit {per_lot:.2f}/lot vs tick fields "
                    f"{tick_estimate:.2f}/lot; sizing uses order_calc_profit",
                )
            else:
                report.ok("sizing source", f"{per_lot:.2f} per lot for a {probe_distance:.2f} move")
            example_volume = volume_for_risk(
                account.balance,
                config.execution.risk_fraction,
                per_lot,
                spec,
                max_volume=config.execution.max_volume,
            )
            print(
                f"        a {probe_distance:.2f} price move risks "
                f"{per_lot:.2f} {account.currency} per lot, so "
                f"{config.execution.risk_fraction * 100:.2f}% of balance is "
                f"{example_volume} lots (capped at {config.execution.max_volume})"
            )
            if example_volume <= 0:
                report.warn(
                    "risk too small for the minimum lot",
                    "increase account size or risk_per_trade",
                )

            today = datetime.now(timezone.utc).date()
            pnl = daily_pnl(terminal, config.execution.magic, today)
            print(
                f"        today's realized P&L for magic "
                f"{config.execution.magic}: {pnl:.2f} {account.currency}"
            )
    except ExecutionError as exc:
        report.fail("MT5 session", str(exc))
        return 1

    print("Model")
    if os.environ.get("GROQ_API_KEY"):
        report.ok("GROQ_API_KEY is set")
    else:
        report.warn("GROQ_API_KEY is missing", "the daemon cannot request signals")

    print()
    if report.failures:
        print(f"NOT READY: {report.failures} failure(s), {report.warnings} warning(s)")
        return 1
    print(f"READY: 0 failures, {report.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
