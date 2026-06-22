"""
Signal generation + order fill simulation.

Philosophy
----------
The bot mimics a real player placing *limit orders* on the BZ:
  - BUY order  → placed at sell_price + 0.1  (just above current top buyer)
  - SELL order → placed at target price that achieves the desired net return

Fill simulation
---------------
Every run we estimate how much of each open order has filled based on:
  - Direction: buy fills when our price >= current sell_price; sell fills when
    our price <= current buy_price.
  - Rate: hourly_sell/buy_vol × FILL_SHARE_FRAC × elapsed_hours
    FILL_SHARE_FRAC reflects that we're near the top of the queue but not
    guaranteed all volume (other players outbid us sometimes).

BZ Tax: 1.25% applied to all sell-side proceeds (both sell orders and any
instasell used for stop-loss exits).
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from db_local import (
    create_order, get_open_orders, log_signal, update_order,
    order_book_depth_at_price, two_latest_order_book_ts,
    reprice_order,
)

EPS = 1e-9

# ── Constants ─────────────────────────────────────────────────────────────────
BZ_TAX           = 0.0125   # 1.25% on all sell proceeds
TARGET_NET_RETURN = 0.035   # target 3.5% net return per trade
MAX_LOSS          = 0.020   # stop-loss: 2% below entry price
MAX_HOLD_HOURS    = 72.0    # time-stop for open orders (3 days)
SELL_TIME_STOP    = 96.0    # time-stop for sell orders (4 days) before forced close

# Fill simulation
FILL_SHARE_FRAC   = 0.35    # fraction of hourly market volume we capture when at top of queue
TARGET_FILL_HOURS = 1.5     # target: order fills within this many hours (affects sizing)

# Repricing
REPRICE_STALE_MIN    = 30       # minutes of no fill progress before repricing
MAX_REPRICE_COUNT    = 5        # never reprice more than this many times per order
BUY_REPRICE_MAX_SLIP = 0.01     # BUY: don't chase more than 1% above original price
MIN_SELL_NET_RETURN  = 0.005    # SELL: reprice only if we still net ≥0.5% after tax

# Portfolio sizing
MAX_POSITION_PCT  = 0.10    # max 10% of equity per single position
MIN_ORDER_COINS   = 50_000  # skip if order value < 50k coins
MAX_ORDER_COINS   = 15_000_000  # cap at 15M coins per order

# Signal filters
MIN_RETURN_4H       = 0.002   # 4h momentum must be positive (>0.2%)
MIN_RETURN_24H      = 0.003   # 24h momentum must be positive (>0.3%)
MAX_VWAP_DEV        = 0.04    # not more than 4% above 7-day VWAP (overbought guard)
MIN_VOL_ZSCORE      = -1.0    # volume not collapsing
MAX_SPREAD_PCT      = 0.06    # skip wide-spread items
MAX_VOLATILITY_5M   = 0.005   # max 5-min std dev (~0.5%); very noisy items are unreliable
MAX_BUY_FILL_HOURS  = 6.0     # skip illiquid items with >6h estimated fill time


# ── Price math ────────────────────────────────────────────────────────────────

def _buy_order_price(bid_price: float) -> float:
    """Place a Bazaar buy order just above the current top buyer/bid."""
    return round(bid_price + 0.1, 1)


def _sell_order_price(entry_price: float, net_return: float = TARGET_NET_RETURN) -> float:
    """
    Calculate sell order price so that after 1.25% BZ tax we achieve net_return.
      net_return = sell_price * (1 - BZ_TAX) / entry_price - 1
      => sell_price = entry_price * (1 + net_return) / (1 - BZ_TAX)
    """
    return round(entry_price * (1 + net_return) / (1 - BZ_TAX), 1)


def _stop_price(entry_price: float) -> float:
    return round(entry_price * (1 - MAX_LOSS), 1)


# ── Position sizing ────────────────────────────────────────────────────────────

def _size_order(features: dict, equity: float) -> tuple[int, float, str]:
    """
    Returns (qty, order_value_coins, reasoning).
    qty=0 means skip this item.

    Rules (tightest cap wins):
      1. Volume cap: fill within TARGET_FILL_HOURS at FILL_SHARE_FRAC of hourly vol
      2. Portfolio cap: MAX_POSITION_PCT of equity
      3. Absolute min/max
    """
    buy_price = _buy_order_price(features["buy_price"])
    hourly_sell = features.get("hourly_sell_vol", 0)

    # 1. Volume-based maximum
    if hourly_sell > 0:
        vol_qty = int(hourly_sell * FILL_SHARE_FRAC * TARGET_FILL_HOURS)
    else:
        vol_qty = 10  # very low-volume fallback

    # 2. Portfolio-based maximum
    max_coins = min(equity * MAX_POSITION_PCT, MAX_ORDER_COINS)
    port_qty = int(max_coins / (buy_price + EPS))

    qty = min(vol_qty, port_qty)
    order_value = qty * buy_price

    if order_value < MIN_ORDER_COINS:
        return 0, 0.0, f"too small ({order_value:.0f} < {MIN_ORDER_COINS} min)"

    reason = (
        f"vol_cap={vol_qty} (hourly_sell={hourly_sell:.0f}×{FILL_SHARE_FRAC}×{TARGET_FILL_HOURS}h)  "
        f"port_cap={port_qty} ({MAX_POSITION_PCT*100:.0f}% of {equity:.0f})"
    )
    return max(1, qty), order_value, reason


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signals(
    features_list: list[dict],
    spread_map: dict[str, dict],    # item_id → latest spread row
    occupied_items: set[str],       # items already in open orders
    equity: float,
    now_ts: str,
) -> list[dict]:
    """
    Evaluate watchlist items and return buy signal dicts.
    Signals are sorted by confidence (highest first).
    """
    signals: list[dict] = []

    for feat in features_list:
        item_id = feat["item_id"]

        if item_id in occupied_items:
            continue

        spread = spread_map.get(item_id, {})

        # ── Hard filters ────────────────────────────────────────────────────
        if spread.get("is_manipulated"):
            continue

        if feat["spread_pct"] > MAX_SPREAD_PCT:
            continue

        if feat["volatility_24h"] > MAX_VOLATILITY_5M:
            continue

        est_fill_s = spread.get("est_buy_fill_s") or 0
        if est_fill_s > 0 and est_fill_s > MAX_BUY_FILL_HOURS * 3600:
            continue

        # ── Momentum filters ─────────────────────────────────────────────────
        failures: list[str] = []

        if feat["return_4h"] < MIN_RETURN_4H:
            failures.append(f"4h={feat['return_4h']*100:.3f}%<{MIN_RETURN_4H*100:.2f}%")

        if feat["return_24h"] < MIN_RETURN_24H:
            failures.append(f"24h={feat['return_24h']*100:.3f}%<{MIN_RETURN_24H*100:.2f}%")

        if feat["vwap_deviation"] > MAX_VWAP_DEV:
            failures.append(f"vwap_dev={feat['vwap_deviation']*100:.2f}%>{MAX_VWAP_DEV*100:.0f}%")

        if feat["vol_zscore"] < MIN_VOL_ZSCORE:
            failures.append(f"vol_z={feat['vol_zscore']:.2f}<{MIN_VOL_ZSCORE}")

        if failures:
            continue

        # ── Sizing ───────────────────────────────────────────────────────────
        qty, order_value, size_reason = _size_order(feat, equity)
        if qty <= 0:
            continue

        buy_price  = _buy_order_price(feat["buy_price"])
        target_sell = _sell_order_price(buy_price)
        stop        = _stop_price(buy_price)

        # ── Confidence score (0–1) ────────────────────────────────────────────
        # Weighted sum of normalised signal components
        c_4h    = min(1.0, max(0.0, feat["return_4h"]  / 0.015))   # full score at 1.5%
        c_24h   = min(1.0, max(0.0, feat["return_24h"] / 0.025))   # full score at 2.5%
        c_vwap  = min(1.0, max(0.0, 1 - feat["vwap_deviation"] / MAX_VWAP_DEV))
        c_vol   = min(1.0, max(0.0, (feat["vol_zscore"] + 2) / 4))
        c_sprd  = min(1.0, max(0.0, 1 - feat["spread_pct"] / MAX_SPREAD_PCT))
        c_accel = min(1.0, max(0.0, feat["momentum_accel"] / 0.005 + 0.5))
        c_qi    = min(1.0, max(0.0, feat["queue_imbalance"] + 0.5))

        confidence = (
            0.25 * c_4h
            + 0.25 * c_24h
            + 0.15 * c_vwap
            + 0.15 * c_vol
            + 0.10 * c_sprd
            + 0.05 * c_accel
            + 0.05 * c_qi
        )

        expected_fill_hours = qty / (feat["hourly_sell_vol"] * FILL_SHARE_FRAC + EPS)
        expected_hold_hours = min(MAX_HOLD_HOURS,
                                  max(4.0, 12.0 / (abs(feat["return_24h"]) / 0.01 + EPS)))

        signals.append({
            "ts": now_ts,
            "item_id": item_id,
            "action": "BUY",
            "order_price": buy_price,
            "target_price": target_sell,
            "stop_price": stop,
            "qty": qty,
            "order_value": order_value,
            "confidence": round(confidence, 4),
            "expected_net_return": TARGET_NET_RETURN,
            "expected_fill_hours": round(expected_fill_hours, 2),
            "expected_hold_hours": round(expected_hold_hours, 1),
            "reasoning": (
                f"4h={feat['return_4h']*100:+.3f}%  "
                f"24h={feat['return_24h']*100:+.3f}%  "
                f"vwap_dev={feat['vwap_deviation']*100:+.2f}%  "
                f"vol_z={feat['vol_zscore']:+.2f}  "
                f"spread={feat['spread_pct']*100:.2f}%  "
                f"vol_5m_std={feat['volatility_24h']*100:.3f}%  "
                f"fill_est={est_fill_s/3600:.1f}h | {size_reason}"
            ),
        })

    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return signals


# ── Order fill simulation ─────────────────────────────────────────────────────

FillEvent = dict[str, Any]


def update_fills(
    conn: sqlite3.Connection,
    current_prices: dict[str, dict],    # item_id → {buy_price, sell_price, sell_wk, buy_wk}
    now: datetime,
    free_cash: float,
) -> tuple[float, list[FillEvent]]:
    """
    For every open order, simulate how much has filled since last check.
    Updates DB and returns (updated_free_cash, events_list).

    Cash flow:
      - BUY fill:  cash was already deducted when order was placed; no change here
      - SELL fill: cash += qty_filled × sell_price × (1 - BZ_TAX)
      - BUY  expire/cancel: cash += unfilled_qty × order_price  (refund reservation)
      - SELL stop-loss:     cash += qty_remaining × current_sell_price × (1 - BZ_TAX)
    """
    open_orders = get_open_orders(conn)
    events: list[FillEvent] = []

    for order in open_orders:
        item_id = order["item_id"]
        prices  = current_prices.get(item_id)
        if not prices:
            continue

        cur_buy  = prices.get("buy_price", 0) or 0
        cur_sell = prices.get("sell_price", 0) or 0
        sell_wk  = prices.get("sell_wk", 0) or 0
        buy_wk   = prices.get("buy_wk", 0) or 0

        hourly_sell = sell_wk / (7 * 24) if sell_wk else 0
        hourly_buy  = buy_wk  / (7 * 24) if buy_wk  else 0

        updated_at = datetime.fromisoformat(order["updated_at"])
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        elapsed_h = max(0.0, (now - updated_at).total_seconds() / 3600)

        created_at = datetime.fromisoformat(order["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_h = (now - created_at).total_seconds() / 3600

        qty_remaining = order["qty_ordered"] - order["qty_filled"]

        # ── Order book delta fill estimate ───────────────────────────────────
        # Compare how much volume disappeared at/below (for BUY) or at/above
        # (for SELL) our order price between the two most recent snapshots.
        # If only one snapshot exists, fall back to volume-based estimate.
        ob_curr_ts, ob_prev_ts = two_latest_order_book_ts(conn, item_id)

        def _ob_fill_estimate(side_query: str, op: float) -> float | None:
            """
            Returns consumed volume at our price tier, or None if not enough
            snapshots to compute a delta.
            side_query: 'SELL' for BUY orders, 'BUY' for SELL orders.
            """
            if not ob_curr_ts or not ob_prev_ts:
                return None
            curr_depth = order_book_depth_at_price(conn, item_id, side_query, op, ob_curr_ts)
            prev_depth = order_book_depth_at_price(conn, item_id, side_query, op, ob_prev_ts)
            consumed   = max(0.0, prev_depth - curr_depth)
            return consumed * FILL_SHARE_FRAC

        # ── BUY order ────────────────────────────────────────────────────────
        if order["side"] == "BUY":
            if cur_buy > 0 and order["order_price"] >= cur_buy:
                # Our buy order is at/above the current bid, so instant-sellers
                # can fill us over time. Do not require crossing the ask.
                # Use order book delta if available; otherwise fall back to
                # weekly-volume estimate so fills never stall on first run.
                ob_est = _ob_fill_estimate("SELL", order["order_price"])
                if ob_est is not None:
                    new_fills = min(qty_remaining, ob_est)
                else:
                    fill_rate_h = hourly_sell * FILL_SHARE_FRAC
                    new_fills   = min(qty_remaining, fill_rate_h * elapsed_h)

                if new_fills > 0.5:  # threshold to avoid micro-updates
                    new_total   = order["qty_filled"] + new_fills
                    old_cb      = order["cost_basis_avg"] or order["order_price"]
                    new_cb      = (
                        (order["qty_filled"] * old_cb + new_fills * order["order_price"])
                        / (new_total + EPS)
                    )
                    is_done     = new_total >= order["qty_ordered"] * 0.99
                    new_status  = "FILLED" if is_done else "PARTIAL"

                    update_order(conn, order["id"],
                                 qty_filled=new_total,
                                 cost_basis_avg=new_cb,
                                 status=new_status)

                    events.append({
                        "type":        "BUY_FILL",
                        "order_id":    order["id"],
                        "item_id":     item_id,
                        "qty_filled":  new_fills,
                        "total_filled": new_total,
                        "qty_ordered": order["qty_ordered"],
                        "status":      new_status,
                        "price":       order["order_price"],
                    })

                    if is_done:
                        # Automatically place sell order
                        target_p = order["target_price"] or _sell_order_price(new_cb)
                        stop_p   = order["stop_price"]   or _stop_price(new_cb)
                        sell_id  = create_order(conn, {
                            "item_id":          item_id,
                            "side":             "SELL",
                            "order_price":      target_p,
                            "qty":              new_total,
                            "parent_order_id":  order["id"],
                            "target_price":     target_p,
                            "stop_price":       stop_p,
                            "cost_basis_avg":   new_cb,
                        })
                        events.append({
                            "type":           "SELL_ORDER_PLACED",
                            "order_id":       sell_id,
                            "item_id":        item_id,
                            "sell_price":     target_p,
                            "qty":            new_total,
                            "cost_basis":     new_cb,
                            "expected_net":   round(target_p * (1 - BZ_TAX) / new_cb - 1, 4),
                        })

            else:
                # Price moved away — order not at top of queue.
                # Reprice up toward cur_sell if we've been stale long enough.
                reprice_count = order["reprice_count"] or 0
                stale = elapsed_h * 60 >= REPRICE_STALE_MIN
                original_price = order["original_order_price"] or order["order_price"]
                max_chase      = round(original_price * (1 + BUY_REPRICE_MAX_SLIP), 1)
                new_buy_price  = round(cur_buy + 0.1, 1) if cur_buy > 0 else None

                if (
                    stale
                    and new_buy_price is not None
                    and new_buy_price != order["order_price"]
                    and new_buy_price <= max_chase
                    and reprice_count < MAX_REPRICE_COUNT
                ):
                    reprice_order(conn, order["id"], new_buy_price)
                    events.append({
                        "type":       "REPRICED",
                        "order_id":   order["id"],
                        "item_id":    item_id,
                        "side":       "BUY",
                        "old_price":  order["order_price"],
                        "new_price":  new_buy_price,
                        "reprice_n":  reprice_count + 1,
                    })

                elif age_h > MAX_HOLD_HOURS:
                    refund = qty_remaining * order["order_price"]
                    free_cash += refund
                    update_order(conn, order["id"],
                                 status="EXPIRED",
                                 exit_reason="time_stop_unfilled")
                    events.append({
                        "type":      "BUY_EXPIRED",
                        "order_id":  order["id"],
                        "item_id":   item_id,
                        "refund":    refund,
                    })

        # ── SELL order ───────────────────────────────────────────────────────
        elif order["side"] == "SELL":
            if cur_sell > 0 and order["order_price"] <= cur_sell:
                # Our sell order is at/below the current ask, so instant-buyers
                # can fill us over time. Do not require dropping to the bid.
                ob_est = _ob_fill_estimate("BUY", order["order_price"])
                if ob_est is not None:
                    new_fills = min(qty_remaining, ob_est)
                else:
                    fill_rate_h = hourly_buy * FILL_SHARE_FRAC
                    new_fills   = min(qty_remaining, fill_rate_h * elapsed_h)

                if new_fills > 0.5:
                    new_total  = order["qty_filled"] + new_fills
                    is_done    = new_total >= order["qty_ordered"] * 0.99
                    new_status = "FILLED" if is_done else "PARTIAL"

                    proceeds   = new_fills * order["order_price"] * (1 - BZ_TAX)
                    free_cash += proceeds

                    update_order(conn, order["id"],
                                 qty_filled=new_total,
                                 status=new_status)

                    cost_basis = order["cost_basis_avg"] or order["order_price"]
                    net_ret    = (order["order_price"] * (1 - BZ_TAX)) / (cost_basis + EPS) - 1

                    events.append({
                        "type":        "SELL_FILL",
                        "order_id":    order["id"],
                        "item_id":     item_id,
                        "qty_filled":  new_fills,
                        "total_filled": new_total,
                        "qty_ordered": order["qty_ordered"],
                        "status":      new_status,
                        "sell_price":  order["order_price"],
                        "proceeds":    proceeds,
                        "net_return":  net_ret,
                    })

            else:
                # ── Reprice check ────────────────────────────────────────────
                # Move the ask down toward the top buyer if we've been stale,
                # as long as the new price is still profitable.
                reprice_count = order["reprice_count"] or 0
                stale = elapsed_h * 60 >= REPRICE_STALE_MIN
                cost_basis = order["cost_basis_avg"] or order["order_price"]
                new_sell_price = round(cur_sell - 0.1, 1) if cur_sell > 0 else None
                still_profitable = (
                    new_sell_price is not None
                    and (new_sell_price * (1 - BZ_TAX)) / (cost_basis + EPS) - 1
                        >= MIN_SELL_NET_RETURN
                )

                if (
                    stale
                    and still_profitable
                    and new_sell_price != order["order_price"]
                    and reprice_count < MAX_REPRICE_COUNT
                ):
                    reprice_order(conn, order["id"], new_sell_price)
                    events.append({
                        "type":       "REPRICED",
                        "order_id":   order["id"],
                        "item_id":    item_id,
                        "side":       "SELL",
                        "old_price":  order["order_price"],
                        "new_price":  new_sell_price,
                        "reprice_n":  reprice_count + 1,
                    })

                else:
                    # ── Stop-loss check ──────────────────────────────────────
                    stop_p = order["stop_price"] or _stop_price(cost_basis)
                    if cur_sell > 0 and cur_sell < stop_p:
                        # Instasell remaining qty at current sell_price
                        proceeds   = qty_remaining * cur_sell * (1 - BZ_TAX)
                        free_cash += proceeds
                        net_ret    = (cur_sell * (1 - BZ_TAX)) / (cost_basis + EPS) - 1

                        update_order(conn, order["id"],
                                     status="CANCELLED",
                                     exit_reason="stop_loss")

                        events.append({
                            "type":        "STOP_LOSS",
                            "order_id":    order["id"],
                            "item_id":     item_id,
                            "exit_price":  cur_sell,
                            "qty":         qty_remaining,
                            "proceeds":    proceeds,
                            "net_return":  net_ret,
                        })

                    # ── Time-stop check ──────────────────────────────────────
                    elif age_h > SELL_TIME_STOP:
                        # Can't reprice profitably and time ran out — force close
                        exit_price = cur_sell if cur_sell > 0 else (order["order_price"] * 0.97)
                        proceeds   = qty_remaining * exit_price * (1 - BZ_TAX)
                        free_cash += proceeds
                        net_ret    = (exit_price * (1 - BZ_TAX)) / (cost_basis + EPS) - 1

                        update_order(conn, order["id"],
                                     status="EXPIRED",
                                     exit_reason="time_stop_sell")

                        events.append({
                            "type":        "SELL_EXPIRED",
                            "order_id":    order["id"],
                            "item_id":     item_id,
                            "exit_price":  exit_price,
                            "proceeds":    proceeds,
                            "net_return":  net_ret,
                        })

    return free_cash, events


# ── Holdings value ────────────────────────────────────────────────────────────

def compute_holdings_value(
    conn: sqlite3.Connection,
    current_prices: dict[str, dict],
) -> float:
    """
    Mark-to-market value of all open positions:
      - Partially/fully filled BUY orders not yet sold
      - Open/partial SELL orders (qty not yet sold, valued at current sell_price)
    """
    value = 0.0

    # Filled portion of buy orders that are now being sold (open sell orders)
    sell_orders = conn.execute(
        "SELECT * FROM orders WHERE side='SELL' AND status IN ('OPEN','PARTIAL')"
    ).fetchall()
    for o in sell_orders:
        prices    = current_prices.get(o["item_id"], {})
        mark      = prices.get("sell_price") or o["order_price"]
        qty_held  = o["qty_ordered"] - o["qty_filled"]
        if qty_held > 0 and mark > 0:
            value += qty_held * mark

    # Partially filled buy orders (the filled portion we hold)
    partial_buys = conn.execute(
        "SELECT * FROM orders WHERE side='BUY' AND status='PARTIAL'"
    ).fetchall()
    for o in partial_buys:
        prices = current_prices.get(o["item_id"], {})
        mark   = prices.get("sell_price") or o["cost_basis_avg"] or o["order_price"]
        qty    = o["qty_filled"]
        if qty > 0 and mark > 0:
            value += qty * mark

    # Unfilled open BUY orders — cash is already deducted from free_cash when
    # placed, so we must add it back here or equity will show a phantom loss
    # until the order fills or is refunded.
    open_buys = conn.execute(
        "SELECT * FROM orders WHERE side='BUY' AND status='OPEN'"
    ).fetchall()
    for o in open_buys:
        unfilled = o["qty_ordered"] - o["qty_filled"]
        if unfilled > 0:
            value += unfilled * o["order_price"]

    return value
