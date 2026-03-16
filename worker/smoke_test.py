from __future__ import annotations

import pandas as pd

from worker.allocator import BasketConfig, build_basket


def run_allocator_smoke_test() -> None:
    df = pd.DataFrame(
        [
            {
                "item_id": "GOOD_A",
                "expected_return_3d": 0.025,
                "expected_return_7d": 0.042,
                "expected_return_14d": 0.070,
                "confidence_7d": 0.72,
                "spread_pct": 0.012,
                "liquidity_score": 4.2,
                "volatility_7d": 0.05,
                "volatility_30d": 0.07,
                "imbalance": 0.20,
                "volume_zscore_30d": 1.4,
                "volume_slope_30d": 0.7,
                "volume_collapse_frac": 1.1,
                "turnover_min_pass": 1,
                "max_alloc_pct_feasible": 0.30,
                "turnover_coins": 25_000_000,
                "return_3d": 0.015,
            },
            {
                "item_id": "GOOD_B",
                "expected_return_3d": 0.021,
                "expected_return_7d": 0.034,
                "expected_return_14d": 0.056,
                "confidence_7d": 0.69,
                "spread_pct": 0.011,
                "liquidity_score": 3.8,
                "volatility_7d": 0.06,
                "volatility_30d": 0.08,
                "imbalance": 0.16,
                "volume_zscore_30d": 1.1,
                "volume_slope_30d": 0.5,
                "volume_collapse_frac": 1.0,
                "turnover_min_pass": 1,
                "max_alloc_pct_feasible": 0.25,
                "turnover_coins": 18_000_000,
                "return_3d": 0.011,
            },
            {
                "item_id": "GOOD_C",
                "expected_return_3d": 0.018,
                "expected_return_7d": 0.029,
                "expected_return_14d": 0.045,
                "confidence_7d": 0.65,
                "spread_pct": 0.010,
                "liquidity_score": 3.5,
                "volatility_7d": 0.05,
                "volatility_30d": 0.07,
                "imbalance": 0.14,
                "volume_zscore_30d": 0.9,
                "volume_slope_30d": 0.4,
                "volume_collapse_frac": 0.95,
                "turnover_min_pass": 1,
                "max_alloc_pct_feasible": 0.23,
                "turnover_coins": 14_000_000,
                "return_3d": 0.009,
            },
            {
                "item_id": "DOWN_A",
                "expected_return_3d": -0.020,
                "expected_return_7d": -0.028,
                "expected_return_14d": -0.037,
                "confidence_7d": 0.30,
                "spread_pct": 0.018,
                "liquidity_score": 3.1,
                "volatility_7d": 0.09,
                "volatility_30d": 0.12,
                "imbalance": -0.25,
                "volume_zscore_30d": -1.3,
                "volume_slope_30d": -0.8,
                "volume_collapse_frac": 0.62,
                "turnover_min_pass": 1,
                "max_alloc_pct_feasible": 0.18,
                "turnover_coins": 11_000_000,
                "return_3d": -0.014,
            },
        ]
    )

    cfg = BasketConfig(
        mode="BOOTSTRAP_15",
        market_regime="MIXED",
        spread_max=0.05,
        liquidity_min=1.5,
        vol_max=0.35,
        min_weight_pct=0.05,
        max_weight_pct=0.30,
        sell_neg_threshold=-0.005,
        sell_neg_threshold_14d=-0.01,
        max_holdings=5,
    )

    buy_items, sell_items, *_ = build_basket(df, blacklist=set(), cfg=cfg)
    assert buy_items, "Smoke test failed: BUY basket is empty."
    assert sell_items, "Smoke test failed: SELL list is empty."
    print("Smoke test passed.")
    print(f"BUY={len(buy_items)} SELL={len(sell_items)}")


if __name__ == "__main__":
    run_allocator_smoke_test()
