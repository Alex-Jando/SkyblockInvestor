from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _rolling_slope(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    y = values[mask]
    x = np.arange(len(values), dtype=float)[mask]
    x_centered = x - x.mean()
    denom = np.sum(x_centered**2)
    if denom <= EPS:
        return 0.0
    return float(np.sum(x_centered * (y - y.mean())) / denom)


def build_features(
    snapshots: pd.DataFrame,
    spread_max: float,
    paper_equity_coins: float,
    feasibility_factor: float,
) -> pd.DataFrame:
    if snapshots.empty:
        return snapshots

    df = snapshots.copy()
    df["day"] = pd.to_datetime(df["day"]).dt.date
    df = df.sort_values(["item_id", "day"]).reset_index(drop=True)

    for col in ["buy_price", "sell_price", "mid_price", "buy_volume", "sell_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby("item_id", group_keys=False)

    df["buy_volume"] = grouped["buy_volume"].ffill()
    df["sell_volume"] = grouped["sell_volume"].ffill()
    df["buy_volume"] = df["buy_volume"].fillna(0.0)
    df["sell_volume"] = df["sell_volume"].fillna(0.0)

    df["volume_daily"] = df["buy_volume"] + df["sell_volume"]
    df["log_mid_price"] = np.log(df["mid_price"].clip(lower=EPS))
    df["daily_log_return"] = grouped["log_mid_price"].diff()

    for horizon in [1, 3, 7, 14, 30]:
        df[f"return_{horizon}d"] = grouped["log_mid_price"].diff(horizon)

    df["volatility_7d"] = grouped["daily_log_return"].transform(
        lambda s: s.rolling(window=7, min_periods=3).std()
    )
    df["volatility_30d"] = grouped["daily_log_return"].transform(
        lambda s: s.rolling(window=30, min_periods=7).std()
    )

    df["volume_mean_30d"] = grouped["volume_daily"].transform(
        lambda s: s.rolling(window=30, min_periods=7).mean()
    )
    df["volume_std_30d"] = grouped["volume_daily"].transform(
        lambda s: s.rolling(window=30, min_periods=7).std()
    )
    df["volume_median_30d"] = grouped["volume_daily"].transform(
        lambda s: s.rolling(window=30, min_periods=7).median()
    )
    df["volume_zscore_30d"] = (
        df["volume_daily"] - df["volume_mean_30d"]
    ) / (df["volume_std_30d"] + EPS)
    df["volume_slope_30d"] = grouped["volume_daily"].transform(
        lambda s: s.rolling(window=30, min_periods=7).apply(_rolling_slope, raw=True)
    )

    demand_proxy = df["buy_volume"]
    supply_proxy = df["sell_volume"]
    df["imbalance"] = (demand_proxy - supply_proxy) / (demand_proxy + supply_proxy + EPS)

    df["spread_pct"] = (df["buy_price"] - df["sell_price"]) / df["mid_price"].clip(lower=EPS)

    spread_component = 1.0 - np.minimum(df["spread_pct"] / max(spread_max, EPS), 1.0)
    df["liquidity_score"] = np.log1p(df["volume_daily"].clip(lower=0.0)) * spread_component

    df["estimated_daily_turnover_coins"] = df["volume_daily"] * df["mid_price"]
    df["max_alloc_pct_feasible"] = np.minimum(
        0.30,
        (
            df["estimated_daily_turnover_coins"] / max(paper_equity_coins, EPS)
        )
        * feasibility_factor,
    )

    df["volume_collapse_frac"] = df["volume_daily"] / (df["volume_median_30d"] + EPS)

    return df
