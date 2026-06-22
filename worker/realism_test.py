from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

try:
    from market_math import BAZAAR_TAX, net_exit_return, spread_pct
    from portfolio import simulate_daily_rebalance
except ModuleNotFoundError:
    from .market_math import BAZAAR_TAX, net_exit_return, spread_pct
    from .portfolio import simulate_daily_rebalance


def test_bazaar_spread_direction() -> None:
    bid = 100.0
    ask = 110.0
    assert spread_pct(bid, ask) > 0
    assert net_exit_return(entry_ask=ask, exit_bid=bid, tax=BAZAAR_TAX) < 0


def test_daily_portfolio_buys_ask_and_marks_bid() -> None:
    safe_universe = pd.DataFrame(
        [
            {
                "item_id": "TEST_ITEM",
                "score": 1.0,
                "rank_score": 1.0,
                "expected_return_7d": 0.10,
                "confidence_7d": 0.90,
                "return_3d": 0.02,
                "volume_collapse_frac": 1.0,
                "spread_pct": spread_pct(100, 110),
                "max_alloc_pct_feasible": 0.30,
            }
        ]
    )
    buy_items = [
        {
            "item_id": "TEST_ITEM",
            "score": 1.0,
            "spread_pct": spread_pct(100, 110),
            "max_alloc_pct_feasible": 0.30,
        }
    ]

    equity, holdings, *_ = simulate_daily_rebalance(
        day=date(2026, 1, 1),
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        buy_items=buy_items,
        safe_universe=safe_universe,
        current_prices={"TEST_ITEM": {"buy_price": 100.0, "sell_price": 110.0}},
        previous_holdings=[],
        previous_equity=None,
        start_equity=1_000.0,
        historical_peak_equity=1_000.0,
        previous_max_drawdown_pct=0.0,
        position_state={},
        market_regime="MIXED",
        rebalance_band=0.0,
    )

    assert holdings, "expected the portfolio to open a small paper position"
    assert equity["cash_value"] < 1_000.0
    assert equity["equity_value"] < 1_000.0, "crossing ask→bid plus tax should not create instant profit"


if __name__ == "__main__":
    test_bazaar_spread_direction()
    test_daily_portfolio_buys_ask_and_marks_bid()
    print("Realism tests passed.")
