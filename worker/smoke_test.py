from __future__ import annotations

import pandas as pd

from worker.allocator import BasketConfig, build_basket


def run_allocator_smoke_test() -> None:
    df = pd.DataFrame(
        [
            {
                "item_id": "GOOD_A",
                "expected_return_7d": 0.028,
                "expected_return_14d": 0.042,
                "expected_return_30d": 0.070,
                "confidence_7d": 0.74,
                "spread_pct": 0.011,
                "liquidity_score": 4.1,
                "volatility_30d": 0.06,
                "imbalance": 0.22,
                "max_alloc_pct_feasible": 0.30,
                "volume_daily": 2_500_000,
                "turnover_coins": 22_000_000,
                "volume_median_30d": 2_100_000,
                "return_3d": 0.015,
                "return_7d": 0.022,
                "turnover_min_pass": 1,
            },
            {
                "item_id": "GOOD_B",
                "expected_return_7d": 0.019,
                "expected_return_14d": 0.025,
                "expected_return_30d": 0.050,
                "confidence_7d": 0.66,
                "spread_pct": 0.014,
                "liquidity_score": 3.8,
                "volatility_30d": 0.07,
                "imbalance": 0.16,
                "max_alloc_pct_feasible": 0.22,
                "volume_daily": 1_900_000,
                "turnover_coins": 16_500_000,
                "volume_median_30d": 1_700_000,
                "return_3d": 0.011,
                "return_7d": 0.018,
                "turnover_min_pass": 1,
            },
            {
                "item_id": "DOWN_A",
                "expected_return_7d": -0.020,
                "expected_return_14d": -0.028,
                "expected_return_30d": -0.032,
                "confidence_7d": 0.28,
                "spread_pct": 0.018,
                "liquidity_score": 3.0,
                "volatility_30d": 0.09,
                "imbalance": -0.21,
                "max_alloc_pct_feasible": 0.20,
                "volume_daily": 1_300_000,
                "turnover_coins": 11_200_000,
                "volume_median_30d": 1_600_000,
                "return_3d": -0.010,
                "return_7d": -0.018,
                "turnover_min_pass": 1,
            },
        ]
    )

    cfg = BasketConfig(
        mode="BOOTSTRAP_15",
        spread_max=0.05,
        liquidity_min=1.5,
        vol_max=0.35,
        volume_drop_frac=0.10,
        min_expected_return_buy=0.003,
        conf_min_buy=0.45,
        min_weight_pct=0.05,
        max_weight_pct=0.30,
        sell_neg_threshold=-0.005,
        sell_neg_threshold_14d=-0.01,
        min_basket_size=2,
    )

    buy_items, sell_items, *_ = build_basket(df, blacklist=set(), cfg=cfg)
    assert buy_items, "Smoke test failed: BUY basket is empty."
    assert sell_items, "Smoke test failed: SELL list is empty."
    print("Smoke test passed.")
    print(f"BUY={len(buy_items)} SELL={len(sell_items)}")


if __name__ == "__main__":
    run_allocator_smoke_test()
