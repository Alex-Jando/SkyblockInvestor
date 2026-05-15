"""
Walk-forward backtest over stored historical data.

Does NOT call any external APIs. Uses whatever is already in data/local.db.
Runs in seconds — no rate limiting involved.

Usage:
    python backtest.py                  # use all stored 2h candles (up to 7d)
    python backtest.py --start-equity 50000000

How it works
------------
1. Load all 2h candles (interval_m=120) from DB, group by item.
2. Find the global time range and step forward in 2h increments.
3. At each step t:
   a. For each watchlist item, compute features using only rows with ts <= t.
   b. Generate buy signals (reusing the real signals_v2 logic).
   c. Simulate fills for open orders using the actual subsequent candles.
   d. Record portfolio equity.
4. Print a full performance report.

Fill simulation (backtest mode)
--------------------------------
For a BUY order at price P placed at step t:
  - Walk forward through candles after t.
  - Each candle where sell_price <= P contributes fills:
      qty_filled += hourly_vol * FILL_SHARE_FRAC * 2h  (2h per candle)
  - Order is fully filled once qty_filled >= qty_ordered.
  - Time-stop at MAX_HOLD_HOURS; stop-loss at sell_price < stop_price.

SELL order fills mirror the same logic using buy_price.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from db_local import DB_PATH
from signals_v2 import (
    BZ_TAX, TARGET_NET_RETURN, MAX_LOSS, MAX_HOLD_HOURS, SELL_TIME_STOP,
    FILL_SHARE_FRAC, MAX_POSITION_PCT, MIN_ORDER_COINS, MAX_ORDER_COINS,
    MAX_REPRICE_COUNT, REPRICE_STALE_MIN, BUY_REPRICE_MAX_SLIP,
    MIN_SELL_NET_RETURN,
    generate_signals, _buy_order_price, _sell_order_price, _stop_price,
    _size_order,
)

EPS = 1e-9
CANDLE_H = 2.0  # 2h candles


# ── Helpers ───────────────────────────────────────────────────────────────────

def _isoparse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _midprice(row: dict) -> float:
    bp = row.get("buy_price") or 0
    sp = row.get("sell_price") or 0
    if bp <= 0 or sp <= 0:
        return max(bp, sp)
    return (bp + sp) / 2


# ── Feature computation (time-bounded, no real DB) ────────────────────────────

def _compute_features_from_rows(
    item_id: str,
    rows_5m: list[dict],
    rows_2h: list[dict],
    spread_row: dict | None,
) -> dict | None:
    """
    Equivalent to features_v2.compute_features but takes lists of dicts
    instead of querying the DB. Rows must already be filtered to ts <= now.
    """
    if len(rows_5m) < 12:
        # Fall back to 2h rows if we have no 5m data (bootstrap phase)
        if len(rows_2h) < 6:
            return None
        # Synthesise pseudo-5m features from 2h rows
        rows_5m_eff = rows_2h[-24:]  # use last 48h of 2h candles
    else:
        rows_5m_eff = rows_5m[-288:]  # last 24h

    if not rows_5m_eff:
        return None

    def _ret(a: dict, b: dict) -> float:
        pa = _midprice(a)
        pb = _midprice(b)
        if pa <= 0:
            return 0.0
        return (pb - pa) / pa

    n = len(rows_5m_eff)
    cur = rows_5m_eff[-1]
    cur_mid = _midprice(cur)

    def _ret_back(steps: int) -> float:
        if steps >= n:
            return 0.0
        ref = rows_5m_eff[-steps - 1]
        return _ret(ref, cur)

    # steps at 5-min resolution
    return_30m  = _ret_back(6)
    return_1h   = _ret_back(12)
    return_4h   = _ret_back(min(48, n - 1))
    return_12h  = _ret_back(min(144, n - 1))
    return_24h  = _ret_back(n - 1)

    # 7d return from 2h rows
    n2 = len(rows_2h)
    return_7d = 0.0
    if n2 >= 2:
        return_7d = _ret(rows_2h[0], rows_2h[-1])

    # VWAP from 2h rows (last 7d)
    if rows_2h:
        vwap_7d = sum(_midprice(r) for r in rows_2h) / len(rows_2h)
        vwap_deviation = (cur_mid - vwap_7d) / (vwap_7d + EPS)
    else:
        vwap_7d = cur_mid
        vwap_deviation = 0.0

    # Volatility (std of log returns, 5-min)
    log_rets = []
    for i in range(1, len(rows_5m_eff)):
        p0 = _midprice(rows_5m_eff[i - 1])
        p1 = _midprice(rows_5m_eff[i])
        if p0 > 0 and p1 > 0:
            log_rets.append(math.log(p1 / p0))
    if len(log_rets) >= 2:
        mean_lr = sum(log_rets) / len(log_rets)
        volatility_24h = math.sqrt(sum((x - mean_lr) ** 2 for x in log_rets) / len(log_rets))
    else:
        volatility_24h = 0.0

    # Volume stats
    sell_wk_values = [r.get("sell_wk") or 0 for r in rows_5m_eff if r.get("sell_wk")]
    buy_wk_values  = [r.get("buy_wk")  or 0 for r in rows_5m_eff if r.get("buy_wk")]
    sell_moving_week = sell_wk_values[-1] if sell_wk_values else 0
    buy_moving_week  = buy_wk_values[-1]  if buy_wk_values  else 0
    hourly_sell_vol  = sell_moving_week / (7 * 24) if sell_moving_week else 0
    hourly_buy_vol   = buy_moving_week  / (7 * 24) if buy_moving_week  else 0

    # Vol z-score: compare last 12pt vol to full-period vol
    if len(sell_wk_values) >= 12:
        recent_vol = sum(sell_wk_values[-12:]) / 12
        full_vol   = sum(sell_wk_values) / len(sell_wk_values)
        if full_vol > 0:
            vol_zscore = (recent_vol - full_vol) / (full_vol * 0.3 + EPS)
        else:
            vol_zscore = 0.0
    else:
        vol_zscore = 0.0

    # Volume trend
    half = len(rows_5m_eff) // 2
    if half >= 6:
        v1 = [r.get("sell_wk") or 0 for r in rows_5m_eff[:half]]
        v2 = [r.get("sell_wk") or 0 for r in rows_5m_eff[half:]]
        vol_trend = (sum(v2) / max(len(v2), 1)) / (sum(v1) / max(len(v1), 1) + EPS) - 1
    else:
        vol_trend = 0.0

    # Queue imbalance
    bq = cur.get("buy_vol_q") or 0
    sq = cur.get("sell_vol_q") or 0
    total_q = bq + sq
    queue_imbalance = (bq - sq) / (total_q + EPS) if total_q > 0 else 0.0

    # Momentum acceleration (2nd derivative of 5-min returns)
    if len(rows_5m_eff) >= 3:
        r1 = _ret(rows_5m_eff[-3], rows_5m_eff[-2])
        r2 = _ret(rows_5m_eff[-2], rows_5m_eff[-1])
        momentum_accel = r2 - r1
    else:
        momentum_accel = 0.0

    spread_pct = ((cur.get("sell_price") or 0) - (cur.get("buy_price") or 0)) / ((cur.get("buy_price") or EPS) + EPS)

    if not cur.get("sell_price") or not cur.get("buy_price"):
        return None

    return {
        "item_id":         item_id,
        "sell_price":      cur["sell_price"],
        "buy_price":       cur["buy_price"],
        "spread_pct":      max(0.0, spread_pct),
        "return_30m":      return_30m,
        "return_1h":       return_1h,
        "return_4h":       return_4h,
        "return_12h":      return_12h,
        "return_24h":      return_24h,
        "return_7d":       return_7d,
        "vwap_7d":         vwap_7d,
        "vwap_deviation":  vwap_deviation,
        "volatility_24h":  volatility_24h,
        "vol_zscore":      vol_zscore,
        "vol_trend":       vol_trend,
        "queue_imbalance": queue_imbalance,
        "momentum_accel":  momentum_accel,
        "hourly_sell_vol": hourly_sell_vol,
        "hourly_buy_vol":  hourly_buy_vol,
        "sell_moving_week": sell_moving_week,
        "buy_moving_week":  buy_moving_week,
        "n_pts_5m":        len(rows_5m_eff),
        # spread compat
        "is_manipulated":  0,
        "est_buy_fill_s":  3600 / (hourly_sell_vol + EPS),
    }


# ── Backtest fill simulation ───────────────────────────────────────────────────

class BtOrder:
    """Lightweight order object for backtest simulation."""
    _next_id = 1

    def __init__(
        self, item_id: str, side: str, order_price: float, qty: float,
        target_price: float, stop_price: float, placed_ts: datetime,
        cost_basis: float | None = None, parent_id: int | None = None,
    ):
        self.id              = BtOrder._next_id
        BtOrder._next_id    += 1
        self.item_id         = item_id
        self.side            = side
        self.order_price     = order_price
        self.original_price  = order_price
        self.qty             = qty
        self.qty_filled      = 0.0
        self.target_price    = target_price
        self.stop_price      = stop_price
        self.placed_ts       = placed_ts
        self.cost_basis      = cost_basis or order_price
        self.parent_id       = parent_id
        self.status          = "OPEN"   # OPEN / PARTIAL / FILLED / CANCELLED / EXPIRED
        self.exit_reason     = None
        self.closed_ts: datetime | None = None
        self.reprice_count   = 0
        self.last_fill_ts    = placed_ts

    @property
    def qty_remaining(self) -> float:
        return self.qty - self.qty_filled

    @property
    def age_h(self) -> float:
        if self.closed_ts:
            return (self.closed_ts - self.placed_ts).total_seconds() / 3600
        return 0.0


def simulate_order_on_candles(
    order: BtOrder,
    candles: list[dict],   # sorted by ts ascending, all ts > order.placed_ts
    step_ts_list: list[datetime],
) -> tuple[float, str]:
    """
    Walk through future candles to determine fill outcome.
    Returns (net_pnl_coins, exit_reason).
    net_pnl = 0 for BUY (cost already accounted for); gain/loss for SELL.
    """
    last_reprice_ts = order.placed_ts

    for i, (candle, step_ts) in enumerate(zip(candles, step_ts_list)):
        if order.status in ("FILLED", "CANCELLED", "EXPIRED"):
            break

        age_h     = (step_ts - order.placed_ts).total_seconds() / 3600
        elapsed_h = (step_ts - order.last_fill_ts).total_seconds() / 3600

        sell_wk     = candle.get("sell_wk") or 0
        buy_wk      = candle.get("buy_wk")  or 0
        hourly_sell = sell_wk / (7 * 24) if sell_wk else 0
        hourly_buy  = buy_wk  / (7 * 24) if buy_wk  else 0

        cur_sell = candle["sell_price"]
        cur_buy  = candle["buy_price"]

        # ── BUY order ────────────────────────────────────────────────────────
        if order.side == "BUY":
            if order.order_price >= cur_sell > 0:
                fill_rate_h = hourly_sell * FILL_SHARE_FRAC
                new_fills   = min(order.qty_remaining, fill_rate_h * elapsed_h)
                if new_fills > 0.5:
                    order.qty_filled  += new_fills
                    order.last_fill_ts = step_ts
                    if order.qty_filled >= order.qty * 0.99:
                        order.status    = "FILLED"
                        order.closed_ts = step_ts
                        order.exit_reason = "filled"
                    else:
                        order.status = "PARTIAL"
            else:
                # Check reprice
                reprice_stale = (step_ts - last_reprice_ts).total_seconds() / 60 >= REPRICE_STALE_MIN
                max_chase     = order.original_price * (1 + BUY_REPRICE_MAX_SLIP)
                new_price     = round(cur_sell + 0.1, 1) if cur_sell > 0 else None
                if (
                    reprice_stale
                    and new_price and new_price != order.order_price
                    and new_price <= max_chase
                    and order.reprice_count < MAX_REPRICE_COUNT
                ):
                    order.order_price   = new_price
                    order.reprice_count += 1
                    last_reprice_ts      = step_ts
                elif age_h > MAX_HOLD_HOURS:
                    order.status      = "EXPIRED"
                    order.closed_ts   = step_ts
                    order.exit_reason = "time_stop_unfilled"

        # ── SELL order ───────────────────────────────────────────────────────
        elif order.side == "SELL":
            # Stop-loss first
            if cur_sell > 0 and cur_sell < order.stop_price:
                order.qty_filled  += order.qty_remaining
                order.status       = "CANCELLED"
                order.closed_ts    = step_ts
                order.exit_reason  = "stop_loss"
                order.order_price  = cur_sell  # instasell price
                break

            if order.order_price <= cur_buy > 0:
                fill_rate_h = hourly_buy * FILL_SHARE_FRAC
                new_fills   = min(order.qty_remaining, fill_rate_h * elapsed_h)
                if new_fills > 0.5:
                    order.qty_filled  += new_fills
                    order.last_fill_ts = step_ts
                    if order.qty_filled >= order.qty * 0.99:
                        order.status    = "FILLED"
                        order.closed_ts = step_ts
                        order.exit_reason = "filled"
                    else:
                        order.status = "PARTIAL"
            else:
                # Reprice down
                reprice_stale = (step_ts - last_reprice_ts).total_seconds() / 60 >= REPRICE_STALE_MIN
                new_price     = round(cur_buy - 0.1, 1) if cur_buy > 0 else None
                still_ok      = (
                    new_price is not None
                    and (new_price * (1 - BZ_TAX)) / (order.cost_basis + EPS) - 1
                        >= MIN_SELL_NET_RETURN
                )
                if (
                    reprice_stale and still_ok
                    and new_price != order.order_price
                    and order.reprice_count < MAX_REPRICE_COUNT
                ):
                    order.order_price   = new_price
                    order.reprice_count += 1
                    last_reprice_ts      = step_ts
                elif age_h > SELL_TIME_STOP:
                    exit_p = cur_sell if cur_sell > 0 else order.order_price * 0.97
                    order.order_price  = exit_p
                    order.qty_filled  += order.qty_remaining
                    order.status       = "EXPIRED"
                    order.closed_ts    = step_ts
                    order.exit_reason  = "time_stop_sell"

    return order


def _realised_pnl(buy_order: BtOrder, sell_order: BtOrder | None) -> float | None:
    """
    Returns realised net P&L in coins once both sides are done.
    None if trade isn't complete yet.
    """
    if sell_order and sell_order.status in ("FILLED", "CANCELLED", "EXPIRED"):
        qty        = sell_order.qty_filled
        proceeds   = qty * sell_order.order_price * (1 - BZ_TAX)
        cost       = qty * buy_order.cost_basis
        return proceeds - cost
    return None


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(start_equity: float = 100_000_000.0, max_positions: int = 5) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Load all 2h candles
    raw = conn.execute(
        """SELECT item_id, ts, buy_price, sell_price,
                  buy_vol_q, sell_vol_q, buy_wk, sell_wk
           FROM price_history
           WHERE interval_m=120
           ORDER BY ts ASC"""
    ).fetchall()

    if not raw:
        print("No 2h candle data in DB. Run the bot first to bootstrap history.")
        return

    # Also load 5-min candles for feature enrichment
    raw_5m = conn.execute(
        """SELECT item_id, ts, buy_price, sell_price,
                  buy_vol_q, sell_vol_q, buy_wk, sell_wk
           FROM price_history
           WHERE interval_m=5
           ORDER BY ts ASC"""
    ).fetchall()
    conn.close()

    # Group by item
    candles_by_item: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        candles_by_item[r["item_id"]].append(dict(r))

    candles_5m_by_item: dict[str, list[dict]] = defaultdict(list)
    for r in raw_5m:
        candles_5m_by_item[r["item_id"]].append(dict(r))

    all_items = sorted(candles_by_item.keys())

    # Global time axis (every 2h from earliest to latest)
    all_ts_strs = sorted({r["ts"] for r in raw})
    all_ts      = [_isoparse(t) for t in all_ts_strs]

    print(f"Loaded {len(raw)} 2h candles across {len(all_items)} items")
    print(f"Time range: {all_ts[0].strftime('%Y-%m-%d')} → {all_ts[-1].strftime('%Y-%m-%d')}")
    print(f"Steps: {len(all_ts)} × 2h = {len(all_ts)*2}h ({len(all_ts)*2/24:.1f} days)")
    print(f"Starting equity: {start_equity/1e6:.1f}M coins\n")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    equity          = start_equity
    free_cash       = start_equity
    equity_curve: list[tuple[datetime, float]] = [(all_ts[0], equity)]
    open_orders:  list[BtOrder] = []
    closed_trades: list[dict]   = []

    # We need at least 84 steps (7d) of lookback before we start making
    # decisions — but we'll start earlier and let the feature filter do its work
    MIN_LOOKBACK_STEPS = 12  # 24h of 2h candles

    for step_idx, step_ts in enumerate(all_ts):
        step_ts_str = step_ts.isoformat()

        # ── Simulate fills for open orders up to this step ───────────────────
        for order in list(open_orders):
            if order.status in ("FILLED", "CANCELLED", "EXPIRED"):
                continue

            item_candles  = candles_by_item.get(order.item_id, [])
            future_candles = [
                c for c in item_candles
                if _isoparse(c["ts"]) > order.last_fill_ts
                   and _isoparse(c["ts"]) <= step_ts
            ]
            future_ts = [_isoparse(c["ts"]) for c in future_candles]

            if future_candles:
                simulate_order_on_candles(order, future_candles, future_ts)

            if order.status in ("FILLED", "CANCELLED", "EXPIRED"):
                if order.side == "BUY":
                    if order.status == "FILLED":
                        # Auto-place sell order
                        sell_price = _sell_order_price(order.cost_basis)
                        sell_order = BtOrder(
                            item_id     = order.item_id,
                            side        = "SELL",
                            order_price = sell_price,
                            qty         = order.qty_filled,
                            target_price= sell_price,
                            stop_price  = _stop_price(order.cost_basis),
                            placed_ts   = step_ts,
                            cost_basis  = order.cost_basis,
                            parent_id   = order.id,
                        )
                        open_orders.append(sell_order)
                    else:
                        # Expired/cancelled buy — refund reserved cash
                        refund    = order.qty_remaining * order.order_price
                        free_cash += refund

                elif order.side == "SELL":
                    proceeds   = order.qty_filled * order.order_price * (1 - BZ_TAX)
                    free_cash += proceeds

                    # Find linked buy order
                    parent = next(
                        (o for o in open_orders if o.id == order.parent_id), None
                    )
                    if parent is None:
                        parent = next(
                            (t.get("buy_order") for t in closed_trades
                             if t.get("buy_order") and t["buy_order"].id == order.parent_id),
                            None,
                        )
                    pnl  = _realised_pnl(parent, order) if parent else None
                    net  = (order.order_price * (1 - BZ_TAX)) / ((parent.cost_basis if parent else order.cost_basis) + EPS) - 1
                    closed_trades.append({
                        "item_id":    order.item_id,
                        "buy_ts":     parent.placed_ts if parent else step_ts,
                        "sell_ts":    order.closed_ts or step_ts,
                        "buy_price":  parent.cost_basis if parent else order.cost_basis,
                        "sell_price": order.order_price,
                        "qty":        order.qty_filled,
                        "net_return": net,
                        "pnl_coins":  pnl,
                        "exit_reason": order.exit_reason,
                        "reprice_count_sell": order.reprice_count,
                        "reprice_count_buy": parent.reprice_count if parent else 0,
                        "buy_order":  parent,
                    })
                    if pnl is not None:
                        equity = free_cash + sum(
                            o.qty_remaining * (
                                list(candles_by_item.get(o.item_id, [{}]))[-1].get("sell_price", 0) or 0
                            )
                            for o in open_orders
                            if o.status not in ("FILLED", "CANCELLED", "EXPIRED")
                        )

        # Remove closed orders from open list
        open_orders = [o for o in open_orders
                       if o.status not in ("FILLED", "CANCELLED", "EXPIRED")]

        # ── Build current price map for this step ────────────────────────────
        current_prices: dict[str, dict] = {}
        for item_id, clist in candles_by_item.items():
            candidates = [c for c in clist if _isoparse(c["ts"]) <= step_ts]
            if candidates:
                latest = candidates[-1]
                current_prices[item_id] = {
                    "buy_price":  latest["buy_price"],
                    "sell_price": latest["sell_price"],
                    "sell_wk":    latest.get("sell_wk") or 0,
                    "buy_wk":     latest.get("buy_wk")  or 0,
                }

        # ── Mark-to-market equity ─────────────────────────────────────────────
        holdings = sum(
            o.qty_remaining * (current_prices.get(o.item_id, {}).get("sell_price") or o.order_price)
            for o in open_orders
            if o.status not in ("FILLED", "CANCELLED", "EXPIRED")
        ) + sum(
            o.qty_remaining * o.order_price
            for o in open_orders
            if o.side == "BUY" and o.status in ("OPEN", "PARTIAL")
        )
        equity = free_cash + holdings
        equity_curve.append((step_ts, equity))

        # ── Only generate signals after enough lookback ───────────────────────
        occupied = {o.item_id for o in open_orders}
        slots    = max(0, max_positions - len(occupied))
        if step_idx < MIN_LOOKBACK_STEPS or slots == 0 or free_cash < MIN_ORDER_COINS:
            continue

        # ── Build features for each item using data up to step_ts ────────────
        features_list: list[dict] = []
        spread_map: dict[str, dict] = {}

        for item_id in all_items:
            c2h = [c for c in candles_by_item[item_id] if _isoparse(c["ts"]) <= step_ts]
            c5m = [c for c in candles_5m_by_item.get(item_id, [])
                   if _isoparse(c["ts"]) <= step_ts]

            feat = _compute_features_from_rows(item_id, c5m, c2h, None)
            if feat:
                features_list.append(feat)
                spread_map[item_id] = {
                    "is_manipulated": 0,
                    "est_buy_fill_s": feat["est_buy_fill_s"],
                    "buy_price":      feat["buy_price"],
                    "sell_price":     feat["sell_price"],
                }

        if not features_list:
            continue

        signals = generate_signals(
            features_list, spread_map, occupied, free_cash, step_ts_str
        )
        signals = signals[:slots]

        for sig in signals:
            cost = sig["qty"] * sig["order_price"]
            if cost > free_cash:
                continue

            free_cash -= cost

            order = BtOrder(
                item_id     = sig["item_id"],
                side        = "BUY",
                order_price = sig["order_price"],
                qty         = sig["qty"],
                target_price= sig["target_price"],
                stop_price  = sig["stop_price"],
                placed_ts   = step_ts,
            )
            open_orders.append(order)

    # ── Final P&L for still-open orders at last price ─────────────────────────
    for order in open_orders:
        if order.side == "SELL" and order.status in ("OPEN", "PARTIAL"):
            # Force close at last available price
            last_price = (
                list(candles_by_item.get(order.item_id, [{}]))[-1].get("sell_price")
                or order.order_price
            )
            proceeds    = order.qty_remaining * last_price * (1 - BZ_TAX)
            free_cash  += proceeds
            parent = next((o for o in open_orders if o.side == "BUY"
                           and o.item_id == order.item_id
                           and o.status == "FILLED"), None)
            net = (last_price * (1 - BZ_TAX)) / ((parent.cost_basis if parent else order.cost_basis) + EPS) - 1
            closed_trades.append({
                "item_id":    order.item_id,
                "buy_ts":     parent.placed_ts if parent else order.placed_ts,
                "sell_ts":    all_ts[-1],
                "buy_price":  parent.cost_basis if parent else order.cost_basis,
                "sell_price": last_price,
                "qty":        order.qty_remaining,
                "net_return": net,
                "pnl_coins":  proceeds - order.qty_remaining * (parent.cost_basis if parent else order.cost_basis),
                "exit_reason": "end_of_data",
                "reprice_count_sell": order.reprice_count,
                "reprice_count_buy": parent.reprice_count if parent else 0,
                "buy_order":  parent,
            })
        elif order.side == "BUY" and order.status in ("OPEN", "PARTIAL"):
            # Refund unfilled buy
            free_cash += order.qty_remaining * order.order_price

    # ── Report ────────────────────────────────────────────────────────────────
    final_equity = free_cash
    total_return = (final_equity - start_equity) / (start_equity + EPS) * 100

    complete_trades = [
        t for t in closed_trades
        if t.get("pnl_coins") is not None
    ]
    winning = [t for t in complete_trades if t["net_return"] > 0]
    losing  = [t for t in complete_trades if t["net_return"] <= 0]

    print("═" * 65)
    print("  BACKTEST RESULTS")
    print("═" * 65)
    print(f"  Period   : {all_ts[0].strftime('%Y-%m-%d')} → {all_ts[-1].strftime('%Y-%m-%d')}"
          f"  ({len(all_ts)*2/24:.1f} days)")
    print(f"  Start    : {start_equity/1e6:.2f}M coins")
    print(f"  End      : {final_equity/1e6:.2f}M coins")
    print(f"  Total P&L: {total_return:+.2f}%  "
          f"({(final_equity-start_equity)/1e6:+.2f}M coins)")
    print()

    if complete_trades:
        win_rate  = len(winning) / len(complete_trades) * 100
        avg_ret   = sum(t["net_return"] for t in complete_trades) / len(complete_trades) * 100
        avg_win   = sum(t["net_return"] for t in winning) / max(len(winning), 1) * 100
        avg_loss  = sum(t["net_return"] for t in losing)  / max(len(losing),  1) * 100
        best      = max(complete_trades, key=lambda t: t["net_return"])
        worst     = min(complete_trades, key=lambda t: t["net_return"])

        # Max drawdown
        peak = equity_curve[0][1]
        max_dd = 0.0
        for _, eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / (peak + EPS) * 100
            if dd > max_dd:
                max_dd = dd

        print(f"  Trades   : {len(complete_trades)} closed  "
              f"({len(winning)} wins / {len(losing)} losses)")
        print(f"  Win rate : {win_rate:.1f}%")
        print(f"  Avg ret  : {avg_ret:+.2f}%  "
              f"(wins {avg_win:+.2f}%  /  losses {avg_loss:+.2f}%)")
        print(f"  Max draw : {max_dd:.2f}%")
        print()
        print(f"  Best     : {best['item_id']}  {best['net_return']*100:+.2f}%  "
              f"exit={best['exit_reason']}")
        print(f"  Worst    : {worst['item_id']}  {worst['net_return']*100:+.2f}%  "
              f"exit={worst['exit_reason']}")

        # Exit reason breakdown
        reasons: dict[str, int] = defaultdict(int)
        for t in complete_trades:
            reasons[t["exit_reason"]] += 1
        print()
        print("  Exit reasons:")
        for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25} {cnt:>4}x")

        # Top 10 trades by |pnl|
        top10 = sorted(complete_trades, key=lambda t: abs(t["pnl_coins"] or 0), reverse=True)[:10]
        print()
        print("  Top trades:")
        for t in top10:
            hold_h = (t["sell_ts"] - t["buy_ts"]).total_seconds() / 3600
            print(f"    {t['item_id']:<35} {t['net_return']*100:+.2f}%  "
                  f"{(t['pnl_coins'] or 0)/1000:+.0f}k coins  "
                  f"hold {hold_h:.0f}h  exit={t['exit_reason']}")
    else:
        print("  No completed trades (data window may be too short to close positions).")

    print()
    print(f"  Signals evaluated: {len(all_ts) - MIN_LOOKBACK_STEPS} steps × watchlist")
    print(f"  Total signals placed: {len([o for o in open_orders]) + len(closed_trades)}")
    print("═" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest on stored historical data")
    parser.add_argument("--start-equity", type=float, default=100_000_000,
                        help="Starting paper coins (default 100M)")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Max simultaneous positions (default 5)")
    args = parser.parse_args()
    run_backtest(start_equity=args.start_equity, max_positions=args.max_positions)
