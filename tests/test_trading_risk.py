import pytest

from trading.risk import (
    Limits,
    MarketState,
    RiskError,
    SymbolSpec,
    entry_block_reason,
    round_volume,
    validate_proposal,
    volume_for_risk,
)

SPEC = SymbolSpec(
    point=0.01,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    stops_level_points=10,
)

LIMITS = Limits(
    max_spread_points=50,
    max_trades_per_day=3,
    max_daily_loss_fraction=0.02,
    max_open_positions=1,
    min_reward_risk=1.0,
)


def _state(**overrides: float) -> MarketState:
    values = {
        "balance": 1000.0,
        "spread_points": 20.0,
        "open_positions": 0,
        "trades_today": 0,
        "daily_pnl": 0.0,
    }
    values.update(overrides)
    return MarketState(**values)  # type: ignore[arg-type]


def test_round_volume_snaps_down_to_step():
    assert round_volume(0.0149, SPEC) == pytest.approx(0.01)
    assert round_volume(0.1, SPEC) == pytest.approx(0.1)


def test_round_volume_returns_zero_below_minimum_lot():
    assert round_volume(0.004, SPEC) == 0.0


def test_round_volume_caps_at_symbol_maximum():
    assert round_volume(500.0, SPEC) == pytest.approx(100.0)


def test_volume_for_risk_targets_the_requested_balance_fraction():
    # A 0.50 price stop on a 100 oz contract loses 50 per lot.
    # 0.5% of 1000 = 5.0, so 0.1 lots.
    volume = volume_for_risk(1000.0, 0.005, 50.0, SPEC)

    assert volume == pytest.approx(0.1)


def test_volume_for_risk_returns_zero_when_risk_cannot_buy_min_lot():
    volume = volume_for_risk(10.0, 0.005, 50.0, SPEC)

    assert volume == 0.0


def test_volume_for_risk_rejects_zero_loss_per_lot():
    with pytest.raises(RiskError):
        volume_for_risk(1000.0, 0.005, 0.0, SPEC)


def test_validate_proposal_rejects_stop_on_the_wrong_side():
    with pytest.raises(RiskError, match="wrong side"):
        validate_proposal("BUY", 2000.0, 2001.0, 2002.0, SPEC)


def test_validate_proposal_rejects_stop_inside_broker_minimum():
    with pytest.raises(RiskError, match="minimum"):
        validate_proposal("BUY", 2000.0, 1999.95, 2005.0, SPEC)


def test_validate_proposal_rejects_poor_reward_risk():
    with pytest.raises(RiskError, match="reward:risk"):
        validate_proposal("BUY", 2000.0, 1999.0, 2000.5, SPEC, min_reward_risk=1.0)


def test_validate_proposal_accepts_a_sound_sell():
    risk, reward = validate_proposal("SELL", 2000.0, 2001.0, 1997.5, SPEC, 1.0)

    assert risk == pytest.approx(1.0)
    assert reward == pytest.approx(2.5)


def test_entry_allowed_when_inside_every_limit():
    assert entry_block_reason("BUY", 2000.0, 1999.0, 2002.0, SPEC, LIMITS, _state()) is None


def test_entry_blocked_when_spread_is_too_wide():
    reason = entry_block_reason(
        "BUY", 2000.0, 1999.0, 2002.0, SPEC, LIMITS, _state(spread_points=75.0)
    )

    assert reason == "spread_too_wide"


def test_entry_blocked_when_a_position_is_already_open():
    reason = entry_block_reason(
        "BUY", 2000.0, 1999.0, 2002.0, SPEC, LIMITS, _state(open_positions=1)
    )

    assert reason == "position_already_open"


def test_entry_blocked_after_daily_trade_cap():
    reason = entry_block_reason(
        "BUY", 2000.0, 1999.0, 2002.0, SPEC, LIMITS, _state(trades_today=3)
    )

    assert reason == "max_trades_per_day"


def test_entry_blocked_when_daily_loss_limit_is_reached():
    reason = entry_block_reason(
        "BUY", 2000.0, 1999.0, 2002.0, SPEC, LIMITS, _state(daily_pnl=-20.0)
    )

    assert reason == "daily_loss_limit"


def test_entry_allowed_just_before_the_daily_loss_limit():
    reason = entry_block_reason(
        "BUY", 2000.0, 1999.0, 2002.0, SPEC, LIMITS, _state(daily_pnl=-19.99)
    )

    assert reason is None


def test_entry_block_reason_still_rejects_malformed_signals():
    with pytest.raises(RiskError):
        entry_block_reason("BUY", 2000.0, None, 2002.0, SPEC, LIMITS, _state())
