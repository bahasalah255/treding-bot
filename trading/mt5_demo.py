"""Guarded one-shot MetaTrader 5 demo execution."""

from __future__ import annotations

import os
from typing import Any

import MetaTrader5 as mt5

from trading.groq_signal import Signal


def _volume_for_risk(symbol_info: Any, entry: float, stop_loss: float, balance: float) -> float:
    risk_amount = balance * 0.01
    tick_size = float(symbol_info.trade_tick_size)
    tick_value = float(symbol_info.trade_tick_value)
    if tick_size <= 0 or tick_value <= 0:
        raise RuntimeError("MT5 symbol has no usable tick value or tick size")
    loss_per_lot = abs(entry - stop_loss) / tick_size * tick_value
    volume = risk_amount / loss_per_lot
    step = float(symbol_info.volume_step)
    volume = max(float(symbol_info.volume_min), min(float(symbol_info.volume_max), volume))
    return round((volume // step) * step, 8)


def execute_demo_signal(symbol: str, signal: Signal) -> str:
    """Execute one validated signal on a demo account only."""
    if os.environ.get("ENABLE_DEMO_TRADING") != "1":
        return "DRY_RUN: set ENABLE_DEMO_TRADING=1 to allow demo execution"
    if signal.action == "NO_TRADE":
        return "NO_TRADE"
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        account = mt5.account_info()
        if account is None or account.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL:
            raise RuntimeError("real accounts are prohibited")
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            raise RuntimeError(f"unable to inspect positions: {mt5.last_error()}")
        if len(positions) >= 1:
            return "BLOCKED: a position is already open for this symbol"
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None or not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"unable to read {symbol} from MT5: {mt5.last_error()}")
        entry = float(tick.ask if signal.action == "BUY" else tick.bid)
        volume = _volume_for_risk(info, entry, signal.stop_loss, float(account.balance))
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
        if check is None or check.retcode != 0:
            raise RuntimeError(f"MT5 order check failed: {check or mt5.last_error()}")
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"MT5 order send failed: {result or mt5.last_error()}")
        return f"ORDER_SENT: {result.order}"
    finally:
        mt5.shutdown()
