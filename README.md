# XAUUSD Automated Trading Bot

This project is being developed incrementally for historical XAUUSD backtesting.
The initial stage is strictly backtest-only. It does not connect to MetaTrader 5
accounts and does not submit orders.

## Phase 1 status

Phase 1 provides:

- A modular project layout.
- A UTC-normalized historical CSV schema.
- Basic OHLCV validation.
- A read-only MetaTrader 5 historical importer with a real-account guard.
- A backtest-only configuration template.
- Focused tests for the data validation rules.

## Installation on Debian Linux

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip build-essential libsqlite3-dev git
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Run the Phase 1 tests:

```bash
pytest -q
```

## Historical data format

Place broker-exported files under `data/raw/`. The normalized CSV must contain:

```text
timestamp,open,high,low,close,tick_volume,spread
```

Timestamps are interpreted as UTC. The validator rejects duplicate timestamps,
invalid OHLC relationships, missing values, non-positive prices, and negative
volume or spread values.

Before importing data, record its broker, symbol name, date range, timezone,
export method, and cost assumptions in `data/metadata/`. Broker-specific symbol
names and contract specifications must not be guessed.

## MT5 historical import

The importer only reads account metadata, Market Watch symbols, and historical
rates. It never submits orders. It refuses to run when the connected account is
identified as a real account.

Install the optional MT5 dependency in the environment that can access the
terminal:

```bash
python -m pip install -r requirements-mt5.txt
```

With the MT5 terminal already connected to the intended demo account, run:

```bash
python -m data.mt5_importer \
  --start 2024-01-01T00:00:00Z \
  --end 2025-01-01T00:00:00Z \
  --output-dir data/raw
```

If more than one gold symbol is visible, provide the exact Market Watch name:

```bash
python -m data.mt5_importer --symbol XAUUSDm \
  --start 2024-01-01T00:00:00Z --end 2025-01-01T00:00:00Z
```

The command writes separate M5 and M15 CSV files containing timestamp, OHLC,
tick volume, spread, and real volume when supplied by MT5. On Debian, the
official MT5 Python package may require a Windows terminal or a verified Wine
configuration; this repository does not assume that setup is available.

## Debian/Wine alternative: MT5-native exporter

If the Linux Python package cannot connect to the Wine terminal, use the
read-only [MT5 exporter](tools/mt5_exporter/ExportXAUUSD.mq5) instead:

1. Open MT5's MetaEditor under Wine.
2. Create a new script named `ExportXAUUSD.mq5`.
3. Paste the exporter source and compile it.
4. Attach it to an `XAUUSD` chart, or set `SymbolName` to the exact Market
   Watch symbol.
5. Set `StartDate` and `EndDate`, then run the script.

The script calls only `CopyRates` and writes M5/M15 CSV files. It does not
contain trade or order functions. Files are written to MT5's Common Files
folder; copy the generated `xauusd_M5.csv` and `xauusd_M15.csv` files into
`data/raw/` on Debian for validation and backtesting.

## Safety boundary

Do not add account credentials or enable execution until historical validation,
backtesting, out-of-sample testing, and explicit manual authorization have been
completed.
