from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from trading.groq_signal import (
    MIN_ANALYSIS_CANDLES,
    _detect_breakout_retest,
    _confirmed_pivots,
    _market_evidence,
    _redact_diagnostic,
    _rr_mismatch_diagnostic,
    _validate_frame,
    _validate_setup,
    request_signal,
)


def _frame(closes=None):
    if closes is None:
        closes = np.linspace(100, 120, MIN_ANALYSIS_CANDLES + 10)
    closes = np.asarray(closes, dtype=float)
    index = pd.date_range("2026-01-01", periods=len(closes), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes - 0.2,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
        },
        index=index,
    )


def _groq_response(payload):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )


def _valid_payload(action="NO_TRADE"):
    return {
        "action": action,
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward": None,
        "market_trend": "BULLISH",
        "market_condition": "TREND",
        "buy_setup_score": 40,
        "sell_setup_score": 20,
        "rationale": "insufficient confirmation",
        "rejection_reasons": ["no confirmed retest"],
    }


def test_empty_and_insufficient_frames_are_data_errors():
    assert _validate_frame(pd.DataFrame()) == "candle frame is empty"
    assert "not enough candles" in _validate_frame(_frame(np.arange(20)))


def test_invalid_ohlc_and_non_utc_frames_are_rejected():
    frame = _frame()
    frame.loc[frame.index[0], "high"] = np.inf
    assert "non-finite" in _validate_frame(frame)
    naive = _frame().tz_localize(None)
    assert "explicit timezone" in _validate_frame(naive)


def test_market_evidence_has_warm_indicators_and_local_scores():
    evidence = _market_evidence(_frame())
    assert np.isfinite(evidence["ema20"])
    assert np.isfinite(evidence["ema50"])
    assert np.isfinite(evidence["atr14"])
    assert 0 <= evidence["local_buy_score"] <= 100
    assert 0 <= evidence["local_sell_score"] <= 100
    assert "breakout_retest" in evidence


def test_breakout_retest_requires_retest_confirmation():
    closes = np.array([100.0] * 25 + [99.0, 101.5, 101.2, 101.4, 101.6])
    frame = _frame(np.pad(closes, (0, MIN_ANALYSIS_CANDLES + 10 - len(closes)), mode="edge"))
    evidence = _market_evidence(frame)
    assert "bullish" in evidence["breakout_retest"]
    assert set(evidence["breakout_retest"]["bullish"]) >= {
        "confirmed", "broken_level", "breakout_timestamp"
    }


def test_breakout_retest_invalidation_cannot_become_confirmation():
    closes = np.array([100.0] * 20 + [101.0, 99.0] + [99.0] * 48)
    frame = _frame(closes)
    evidence = _market_evidence(frame)
    bullish = evidence["breakout_retest"]["bullish"]
    assert bullish["invalidated"] is True
    assert bullish["confirmed"] is False


def test_swings_are_only_exposed_after_right_confirmation_candles():
    frame = _frame()
    pivots = _confirmed_pivots(frame)
    latest = frame.index[-1]
    for pivot in pivots["high"] + pivots["low"]:
        assert pd.Timestamp(pivot["confirmation_timestamp"]) < latest


def test_validate_setup_leaves_minimum_rr_to_entry_gate():
    evidence = {
        "atr14": 1.0, "support": 90.0, "resistance": 120.0,
        "latest_close": 100.0, "range": {"middle": False},
        "bos": {"up": False, "down": False},
        "breakout_retest": {
            "bullish": {"confirmed": False}, "bearish": {"confirmed": False}
        },
    }
    assert _validate_setup("SELL", 100, 110, 97, evidence) is None


def test_request_signal_rejects_missing_groq_scores(monkeypatch):
    payload = _valid_payload()
    del payload["buy_setup_score"]
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **options: _groq_response(json.dumps(payload))
                )
            )
        ),
    )
    signal = request_signal(_frame())
    assert signal.action == "DATA_ERROR"
    assert "missing fields" in signal.rationale


def test_malformed_response_diagnostic_redacts_sensitive_values():
    diagnostic = _redact_diagnostic(
        '{"api_key":"gsk_secret-value","Authorization":"Bearer sk_token-value"}'
    )
    assert "secret-value" not in diagnostic
    assert "token-value" not in diagnostic
    assert diagnostic.count("[REDACTED]") >= 2


def test_request_signal_accepts_empty_rejection_array(monkeypatch):
    payload = _valid_payload()
    payload["rejection_reasons"] = []
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )
    signal = request_signal(_frame())
    assert signal.action == "NO_TRADE"
    assert signal.rejection_reasons == ()


def test_request_signal_accepts_string_rejection_array(monkeypatch):
    payload = _valid_payload()
    payload["rejection_reasons"] = ["no confirmed retest", "range middle"]
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )
    signal = request_signal(_frame())
    assert signal.action == "NO_TRADE"
    assert signal.rejection_reasons == tuple(payload["rejection_reasons"])


@pytest.mark.parametrize(
    "rejection_reasons",
    ["one reason", None, [1], [{"reason": "bad"}]],
)
def test_request_signal_rejects_malformed_rejection_reasons(
    monkeypatch, rejection_reasons
):
    payload = _valid_payload()
    payload["rejection_reasons"] = rejection_reasons
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )
    signal = request_signal(_frame())
    assert signal.action == "DATA_ERROR"
    assert "rejection_reasons" in signal.rationale


def test_request_signal_rejects_missing_rejection_reasons(monkeypatch):
    payload = _valid_payload()
    del payload["rejection_reasons"]
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )
    signal = request_signal(_frame())
    assert signal.action == "DATA_ERROR"
    assert "missing fields" in signal.rationale


def test_request_signal_rejects_out_of_range_scores(monkeypatch):
    payload = _valid_payload()
    payload["buy_setup_score"] = 101
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )
    signal = request_signal(_frame())
    assert signal.action == "DATA_ERROR"
    assert "between 0 and 100" in signal.rationale


def test_request_signal_handles_invalid_json_and_api_failure(monkeypatch):
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **options: _groq_response("not json")
                )
            )
        ),
    )
    invalid_json_signal = request_signal(_frame())
    assert invalid_json_signal.action == "DATA_ERROR"
    assert invalid_json_signal.diagnostic == "not json"

    def fail(**options):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
        ),
    )
    signal = request_signal(_frame())
    assert signal.action == "DATA_ERROR"
    assert "Groq analysis failed" in signal.rationale


def test_local_rejection_preserves_groq_rejection_reasons(monkeypatch):
    payload = _valid_payload("SELL")
    payload.update({
        "entry_price": 100.0,
        "stop_loss": 101.0,
        "take_profit": 99.0,
        "risk_reward": 1.0,
        "rejection_reasons": ["target lacks confirmation"],
    })
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )
    signal = request_signal(_frame())
    assert signal.action == "NO_TRADE"
    assert signal.rejection_reasons[0] == "target lacks confirmation"
    assert "entry is too far" in signal.rejection_reasons[-1]


@pytest.mark.parametrize(
    ("action", "entry", "stop_loss", "take_profit"),
    [
        ("BUY", 100.0, 90.0, 120.0),
        ("SELL", 100.0, 110.0, 80.0),
    ],
)
def test_rr_mismatch_returns_data_error_with_diagnostics(
    monkeypatch, action, entry, stop_loss, take_profit
):
    payload = _valid_payload(action)
    payload.update({
        "entry_price": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": 1.0,
    })
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )

    signal = request_signal(_frame())

    assert signal.action == "DATA_ERROR"
    assert signal.rationale == "Groq risk_reward does not match calculated prices"
    assert "symbol=XAUUSD" in signal.diagnostic
    assert f"action={action}" in signal.diagnostic
    assert "groq_entry=100.0" in signal.diagnostic
    assert "groq_stop_loss=" in signal.diagnostic
    assert "groq_take_profit=" in signal.diagnostic
    assert "groq_risk_reward=1.0" in signal.diagnostic
    assert "calculated_risk_reward=2.0" in signal.diagnostic
    assert "absolute_difference=1.0" in signal.diagnostic
    assert "relative_difference=0.5" in signal.diagnostic
    assert "rel_tol=0.01" in signal.diagnostic
    assert "abs_tol=0.01" in signal.diagnostic


def test_rr_within_tolerance_is_accepted(monkeypatch):
    payload = _valid_payload("BUY")
    payload.update({
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "take_profit": 120.0,
        "risk_reward": 2.01,
    })
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )

    signal = request_signal(_frame())

    assert signal.action != "DATA_ERROR"


def test_rr_outside_tolerance_remains_data_error(monkeypatch):
    payload = _valid_payload("SELL")
    payload.update({
        "entry_price": 100.0,
        "stop_loss": 110.0,
        "take_profit": 80.0,
        "risk_reward": 2.03,
    })
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )

    signal = request_signal(_frame())

    assert signal.action == "DATA_ERROR"
    assert "absolute_difference=0.0299999999999998" in signal.diagnostic


def test_rr_mismatch_diagnostic_marks_missing_values():
    diagnostic = _rr_mismatch_diagnostic(
        "BUY", None, None, None, None, None, None
    )

    assert "symbol=XAUUSD" in diagnostic
    assert "timestamp=unavailable" in diagnostic
    assert "action=BUY" in diagnostic
    assert "groq_entry=unavailable" in diagnostic
    assert "calculated_risk_reward=unavailable" in diagnostic
    assert "absolute_difference=unavailable" in diagnostic


def test_no_trade_preserves_local_scores_and_does_not_call_orders(monkeypatch):
    payload = _valid_payload()
    text = json.dumps(payload)
    monkeypatch.setattr(
        "trading.groq_signal.Groq",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **options: _groq_response(text)))
        ),
    )
    signal = request_signal(_frame())
    assert signal.action == "NO_TRADE"
    assert signal.local_score_breakdown
    assert signal.entry_price is None
