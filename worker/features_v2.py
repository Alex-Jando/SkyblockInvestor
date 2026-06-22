"""
Feature engineering from intraday price history.

Primary data source: Coflnet /history/day (5-min intervals, ~288 points = 24h).
Supplementary: /history/week (2h intervals, ~84 points = 7d) for longer-term context.
"""
from __future__ import annotations

import math
import sqlite3
from typing import Any

try:
    from market_math import spread_pct as bazaar_spread_pct
except ModuleNotFoundError:
    from .market_math import spread_pct as bazaar_spread_pct

EPS = 1e-9

# Lookback windows in multiples of 5-min intervals (day-history)
# 30m=6, 1h=12, 4h=48, 12h=144, 24h=288
# 2h-interval (week-history): 4h=2pts, 24h=12pts, 7d=84pts


def compute_features(
    conn: sqlite3.Connection,
    item_id: str,
) -> dict[str, Any] | None:
    """
    Compute trading features for a single item.
    Returns None if insufficient history.
    """
    # ── 5-min data (last 24h) ────────────────────────────────────────────────
    rows_5m = conn.execute(
        """SELECT ts, buy_price, sell_price, buy_vol_q, sell_vol_q, buy_wk, sell_wk
           FROM price_history
           WHERE item_id=? AND interval_m=5
             AND ts >= datetime('now','-25 hours')
           ORDER BY ts ASC""",
        (item_id,),
    ).fetchall()

    # Need at least 12 points (1h) for meaningful features
    if len(rows_5m) < 12:
        return None

    # ── 2h data (last 7d) ────────────────────────────────────────────────────
    rows_2h = conn.execute(
        """SELECT ts, buy_price, sell_price, buy_vol_q, sell_vol_q, buy_wk, sell_wk
           FROM price_history
           WHERE item_id=? AND interval_m=120
             AND ts >= datetime('now','-8 days')
           ORDER BY ts ASC""",
        (item_id,),
    ).fetchall()

    latest = rows_5m[-1]
    buy_price = latest["buy_price"]
    sell_price = latest["sell_price"]

    if not buy_price or not sell_price or buy_price <= 0 or sell_price <= 0:
        return None

    mid_price = (buy_price + sell_price) / 2.0
    spread_pct = bazaar_spread_pct(buy_price, sell_price)

    # Weekly traded volumes (most recent point)
    sell_wk = latest["sell_wk"] or 0
    buy_wk = latest["buy_wk"] or 0
    # Items per hour flowing through the market
    hourly_sell_vol = sell_wk / (7 * 24) if sell_wk else 0
    hourly_buy_vol = buy_wk / (7 * 24) if buy_wk else 0

    # ── Mid-price series ─────────────────────────────────────────────────────
    def _mid(row: sqlite3.Row) -> float | None:
        b, s = row["buy_price"], row["sell_price"]
        return (b + s) / 2.0 if b and s and b > 0 and s > 0 else None

    mids_5m = [m for r in rows_5m if (m := _mid(r)) is not None]
    n = len(mids_5m)
    if n < 12:
        return None

    p_now = mids_5m[-1]

    # Momentum: returns over key windows (indices from end)
    def _ret(lookback_pts: int) -> float:
        idx = max(0, n - lookback_pts - 1)
        p_prev = mids_5m[idx]
        return (p_now - p_prev) / (p_prev + EPS)

    return_30m = _ret(6)    # 6×5min = 30 min
    return_1h  = _ret(12)   # 12×5min = 1h
    return_4h  = _ret(48)   # 48×5min = 4h
    return_12h = _ret(144) if n >= 100 else _ret(n // 2)
    return_24h = _ret(min(287, n - 1))

    # ── 7-day context from 2h data ───────────────────────────────────────────
    return_7d = 0.0
    vwap_7d = p_now
    if len(rows_2h) >= 4:
        mids_2h = [m for r in rows_2h if (m := _mid(r)) is not None]
        if mids_2h:
            return_7d = (p_now - mids_2h[0]) / (mids_2h[0] + EPS)
            vwap_7d = sum(mids_2h) / len(mids_2h)
    vwap_deviation = (p_now - vwap_7d) / (vwap_7d + EPS)

    # ── Volatility (std dev of 5-min log returns) ────────────────────────────
    log_rets = []
    for i in range(1, len(mids_5m)):
        if mids_5m[i - 1] > 0:
            log_rets.append(math.log(mids_5m[i] / mids_5m[i - 1]))

    volatility_24h = 0.02  # default
    if len(log_rets) >= 3:
        mean_lr = sum(log_rets) / len(log_rets)
        variance = sum((r - mean_lr) ** 2 for r in log_rets) / len(log_rets)
        volatility_24h = math.sqrt(variance)  # std dev of 5-min returns

    # Annualised vol (for reference): volatility_24h * sqrt(288*365)
    # We keep raw 5-min std dev; thresholds are calibrated to this scale.

    # ── Volume trend ─────────────────────────────────────────────────────────
    # Compare current weekly moving vol to 6h ago
    sell_wk_series = [r["sell_wk"] for r in rows_5m if r["sell_wk"]]
    vol_zscore = 0.0
    vol_trend = 0.0
    if len(sell_wk_series) >= 6:
        recent_avg = sum(sell_wk_series[-6:]) / 6
        full_avg = sum(sell_wk_series) / len(sell_wk_series)
        if full_avg > 0:
            vol_zscore = (recent_avg - full_avg) / (full_avg * 0.15 + EPS)
        # Linear trend (positive = volume growing)
        if len(sell_wk_series) >= 12:
            half = len(sell_wk_series) // 2
            first_half = sum(sell_wk_series[:half]) / half
            second_half = sum(sell_wk_series[half:]) / max(1, len(sell_wk_series) - half)
            vol_trend = (second_half - first_half) / (first_half + EPS)

    # ── Order-book queue imbalance ───────────────────────────────────────────
    bq = latest["buy_vol_q"] or 0
    sq = latest["sell_vol_q"] or 0
    queue_imbalance = (bq - sq) / (bq + sq + EPS)  # >0 = more buy demand

    # ── Momentum acceleration (2nd derivative) ───────────────────────────────
    # Is the momentum speeding up or slowing down?
    if n >= 4:
        recent_ret = (mids_5m[-1] - mids_5m[-3]) / (mids_5m[-3] + EPS)
        older_ret  = (mids_5m[-3] - mids_5m[-5]) / (mids_5m[-5] + EPS) if n >= 6 else 0
        momentum_accel = recent_ret - older_ret
    else:
        momentum_accel = 0.0

    return {
        "item_id": item_id,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "mid_price": mid_price,
        "spread_pct": spread_pct,
        "return_30m": return_30m,
        "return_1h": return_1h,
        "return_4h": return_4h,
        "return_12h": return_12h,
        "return_24h": return_24h,
        "return_7d": return_7d,
        "vwap_7d": vwap_7d,
        "vwap_deviation": vwap_deviation,
        "volatility_24h": volatility_24h,
        "vol_zscore": vol_zscore,
        "vol_trend": vol_trend,
        "queue_imbalance": queue_imbalance,
        "momentum_accel": momentum_accel,
        "hourly_sell_vol": hourly_sell_vol,
        "hourly_buy_vol": hourly_buy_vol,
        "sell_moving_week": sell_wk,
        "buy_moving_week": buy_wk,
        "n_pts_5m": len(mids_5m),
    }
