"""
SkyBlock Investor v2 — main orchestrator.

Run modes:
  python -m worker.main_v2            # continuous loop (local dev)
  python -m worker.main_v2 --once     # single run (CI / GitHub Actions)

Architecture:
  1. On every run:  fetch Hypixel snapshot + Coflnet spread
  2. Refresh day-history (5-min) for watchlist items (up to 30 per run)
  3. Bootstrap (first run / stale): also fetch 2h week-history for all watchlist items
  4. Simulate order fills from new price data
  5. Generate new BUY signals where equity allows
  6. Print signals to terminal + append to data/signals.log
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure worker/ is importable regardless of cwd
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coflnet_api import fetch_spread, fetch_history_day, fetch_history_week
from db_local import (
    init_db, get_conn,
    upsert_price_rows, upsert_spread_rows,
    latest_spread,
    spread_age_seconds, get_open_orders,
    create_order, log_signal, log_portfolio,
    get_state, set_state,
    upsert_order_book, prune_old_order_book,
)
from features_v2 import compute_features
from signals_v2 import (
    generate_signals, update_fills, compute_holdings_value,
    _buy_order_price, MAX_REPRICE_COUNT,
)
from hypixel_api import fetch_bazaar

load_dotenv(Path(_HERE).resolve().parents[0] / ".env")

LOG_DIR  = Path(_HERE).resolve().parent / "data"
LOG_FILE = LOG_DIR / "signals.log"

# ── Settings ──────────────────────────────────────────────────────────────────

PAPER_START_COINS  = float(os.environ.get("PAPER_START_COINS", "100000000"))
HYPIXEL_API_KEY    = os.environ.get("HYPIXEL_API_KEY", "")
RUN_INTERVAL_S     = int(os.environ.get("RUN_INTERVAL_S", "600"))   # 10 min default
WATCHLIST_SIZE     = int(os.environ.get("WATCHLIST_SIZE", "80"))
HISTORY_REFRESH_S  = int(os.environ.get("HISTORY_REFRESH_S", "300"))  # re-fetch day hist every 5 min
WEEK_REFRESH_H     = float(os.environ.get("WEEK_REFRESH_H", "2"))      # refresh week hist every 2h
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "5"))

# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
        ],
    )


def _signal_file() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_FILE


# ── Watchlist management ──────────────────────────────────────────────────────

def _build_watchlist(spread_rows: list[dict]) -> list[str]:
    """
    Select top WATCHLIST_SIZE items from spread data.
    Filter: not manipulated, has both buy/sell price, fill < 6h.
    Score: profit_per_hour / sqrt(est_buy_fill_s) — high profit, fast fill.
    """
    MAX_FILL_S = 6 * 3600
    candidates = [
        r for r in spread_rows
        if not r["is_manipulated"]
        and r.get("buy_price") and r.get("sell_price")
        and r.get("est_buy_fill_s") and r["est_buy_fill_s"] < MAX_FILL_S
        and r.get("profit_per_hour", 0) > 0
    ]
    candidates.sort(
        key=lambda r: r["profit_per_hour"] / (r["est_buy_fill_s"] ** 0.5 + 1),
        reverse=True,
    )
    return [r["item_id"] for r in candidates[:WATCHLIST_SIZE]]


# ── Hypixel snapshot → DB ─────────────────────────────────────────────────────

def ingest_hypixel(conn: Any) -> dict[str, dict]:
    """Fetch Hypixel BZ snapshot, store to DB, return item_id → price dict."""
    if not HYPIXEL_API_KEY:
        logging.warning("HYPIXEL_API_KEY not set; skipping Hypixel fetch")
        return {}

    payload = fetch_bazaar(HYPIXEL_API_KEY)
    now_ts  = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows_out: list[dict] = []
    price_map: dict[str, dict] = {}

    ob_rows: list[dict] = []
    products = payload.get("products", {})
    for item_id, product in products.items():
        qs = product.get("quick_status", {})
        bp = qs.get("buyPrice")
        sp = qs.get("sellPrice")
        bv = qs.get("buyVolume")
        sv = qs.get("sellVolume")
        buy_wk  = qs.get("buyMovingWeek")
        sell_wk = qs.get("sellMovingWeek")
        if not bp or not sp or bp <= 0 or sp <= 0:
            continue
        rows_out.append({
            "ts":        now_ts,
            "item_id":   item_id,
            "source":    "hypixel",
            "interval_m": 0,
            "buy_price": bp,
            "sell_price": sp,
            "buy_vol_q": bv,
            "sell_vol_q": sv,
            "buy_wk":    buy_wk,
            "sell_wk":   sell_wk,
        })
        price_map[item_id] = {
            "buy_price":  bp,
            "sell_price": sp,
            "buy_vol_q":  bv,
            "sell_vol_q": sv,
            "buy_wk":     buy_wk  or 0,
            "sell_wk":    sell_wk or 0,
        }

        # Store order book levels for fill accuracy
        # sell_summary = people offering to SELL (fills our BUY orders)
        for level in product.get("sell_summary", []):
            price  = level.get("pricePerUnit")
            amount = level.get("amount")
            if price and amount:
                ob_rows.append({"ts": now_ts, "item_id": item_id,
                                "side": "SELL", "price": price, "amount": amount})
        # buy_summary = people offering to BUY (fills our SELL orders)
        for level in product.get("buy_summary", []):
            price  = level.get("pricePerUnit")
            amount = level.get("amount")
            if price and amount:
                ob_rows.append({"ts": now_ts, "item_id": item_id,
                                "side": "BUY", "price": price, "amount": amount})

    inserted = upsert_price_rows(conn, rows_out)
    if ob_rows:
        upsert_order_book(conn, ob_rows)
        conn.commit()
        prune_old_order_book(conn, keep_hours=4)  # keep last 4h of book snapshots
        conn.commit()
    logging.info("Hypixel snapshot: %d items (%d price rows, %d book levels)",
                 len(rows_out), inserted, len(ob_rows))
    return price_map


def _enrich_price_map_with_vol(price_map: dict, conn: Any) -> dict:
    """Add sell_wk / buy_wk from latest Coflnet day-history rows."""
    for item_id in list(price_map.keys()):
        row = conn.execute(
            """SELECT buy_wk, sell_wk FROM price_history
               WHERE item_id=? AND interval_m=5
               ORDER BY ts DESC LIMIT 1""",
            (item_id,),
        ).fetchone()
        if row:
            price_map[item_id]["buy_wk"]  = row["buy_wk"]  or 0
            price_map[item_id]["sell_wk"] = row["sell_wk"] or 0
    return price_map


# ── Print / log helpers ───────────────────────────────────────────────────────

LINE = "═" * 70

def _fmt_coins(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return f"{n:.1f}"


def print_header(equity: float, start: float, free_cash: float, holdings: float) -> None:
    pnl = (equity - start) / (start + 1e-9) * 100
    print(LINE)
    print(f" SkyBlock Investor  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f" Equity: {_fmt_coins(equity)} coins  "
          f"(cash {_fmt_coins(free_cash)} + held {_fmt_coins(holdings)})  "
          f"P&L: {pnl:+.2f}% vs {_fmt_coins(start)} start")
    print(LINE)


def print_event(evt: dict) -> None:
    t = evt["type"]
    if t == "BUY_FILL":
        pct = evt["total_filled"] / evt["qty_ordered"] * 100
        print(f"  🔄 BUY  fill  [{evt['item_id']}]  "
              f"+{evt['qty_filled']:.0f} units @ {evt['price']:.1f}  "
              f"({pct:.0f}% of {evt['qty_ordered']:.0f})")
    elif t == "SELL_ORDER_PLACED":
        print(f"  📋 SELL placed [{evt['item_id']}]  "
              f"{evt['qty']:.0f} units @ {evt['sell_price']:.1f}  "
              f"(cost {evt['cost_basis']:.1f}, exp net {evt['expected_net']*100:+.2f}%)")
    elif t == "SELL_FILL":
        net_pct = evt["net_return"] * 100
        icon = "✅" if net_pct > 0 else "🔴"
        print(f"  {icon} SELL fill [{evt['item_id']}]  "
              f"+{evt['qty_filled']:.0f} units @ {evt['sell_price']:.1f}  "
              f"net {net_pct:+.2f}%  proceeds {_fmt_coins(evt['proceeds'])}")
    elif t == "STOP_LOSS":
        print(f"  🛑 STOP LOSS   [{evt['item_id']}]  "
              f"{evt['qty']:.0f} units @ {evt['exit_price']:.1f}  "
              f"net {evt['net_return']*100:+.2f}%")
    elif t in ("SELL_EXPIRED", "BUY_EXPIRED"):
        print(f"  ⏰ EXPIRED     [{evt['item_id']}]  type={t}")
    elif t == "REPRICED":
        direction = "▲" if evt["side"] == "BUY" else "▼"
        print(f"  {direction}  REPRICED      [{evt['item_id']}]  {evt['side']}  "
              f"{evt['old_price']:.1f} → {evt['new_price']:.1f}  "
              f"(#{evt['reprice_n']} of {MAX_REPRICE_COUNT})")


def print_signal(sig: dict) -> None:
    print(f"\n  📈 BUY SIGNAL: {sig['item_id']}")
    print(f"     Order :  place BUY ORDER at {sig['order_price']:.1f} coins each")
    print(f"     Qty   :  {sig['qty']:.0f} units  (≈ {_fmt_coins(sig['order_value'])} coins)")
    print(f"     Target:  {sig['target_price']:.1f} coins  "
          f"(+{sig['expected_net_return']*100:.1f}% net after 1.25% tax)")
    print(f"     Stop  :  cancel buy order / exit if price drops below "
          f"{sig['stop_price']:.1f}  (-2.0%)")
    print(f"     Fill  :  est. {sig['expected_fill_hours']:.1f}h to fill  |  "
          f"hold ≈ {sig['expected_hold_hours']:.0f}h")
    print(f"     Conf  :  {sig['confidence']:.3f}")
    print(f"     Why   :  {sig['reasoning']}")


def _write_signal_log(sig: dict) -> None:
    with _signal_file().open("a", encoding="utf-8") as f:
        ts = sig["ts"]
        f.write(
            f"{ts}  BUY  {sig['item_id']:<35} "
            f"@ {sig['order_price']:.1f}  qty={sig['qty']:.0f}  "
            f"target={sig['target_price']:.1f}  stop={sig['stop_price']:.1f}  "
            f"conf={sig['confidence']:.3f}  "
            f"net_ret={sig['expected_net_return']*100:.1f}%\n"
        )


# ── History refresh ───────────────────────────────────────────────────────────

def maybe_refresh_history(
    conn: Any,
    watchlist: list[str],
    force_week: bool = False,
) -> int:
    """
    Fetch day-history (5-min) for watchlist items that need a refresh.
    Also fetches week-history (2h) if force_week=True or data is stale.
    Returns number of items refreshed.
    """
    WEEK_REFRESH_S = WEEK_REFRESH_H * 3600
    refreshed = 0

    # Items needing day refresh: no 5-min data in last HISTORY_REFRESH_S seconds
    to_refresh_day: list[str] = []
    to_refresh_week: list[str] = []

    for item_id in watchlist:
        latest_5m = conn.execute(
            """SELECT MAX(ts) ts FROM price_history
               WHERE item_id=? AND interval_m=5""",
            (item_id,),
        ).fetchone()
        age_s = float("inf")
        if latest_5m and latest_5m["ts"]:
            ts = datetime.fromisoformat(latest_5m["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
        if age_s > HISTORY_REFRESH_S:
            to_refresh_day.append(item_id)

        if force_week:
            latest_2h = conn.execute(
                """SELECT MAX(ts) ts FROM price_history
                   WHERE item_id=? AND interval_m=120""",
                (item_id,),
            ).fetchone()
            week_age = float("inf")
            if latest_2h and latest_2h["ts"]:
                ts2 = datetime.fromisoformat(latest_2h["ts"])
                if ts2.tzinfo is None:
                    ts2 = ts2.replace(tzinfo=timezone.utc)
                week_age = (datetime.now(timezone.utc) - ts2).total_seconds()
            if week_age > WEEK_REFRESH_S:
                to_refresh_week.append(item_id)

    total = len(to_refresh_day) + len(to_refresh_week)
    if total == 0:
        return 0

    logging.info(
        "Refreshing history: %d day, %d week items",
        len(to_refresh_day), len(to_refresh_week),
    )

    # Interleave week and day fetches (week is less urgent, do day first)
    for item_id in to_refresh_day:
        rows = fetch_history_day(item_id)
        if rows:
            upsert_price_rows(conn, rows)
            conn.commit()  # commit each item so progress survives a kill
            refreshed += 1

    for item_id in to_refresh_week:
        rows = fetch_history_week(item_id)
        if rows:
            upsert_price_rows(conn, rows)
            conn.commit()  # commit each item so progress survives a kill
            refreshed += 1

    return refreshed


# ── Main run ──────────────────────────────────────────────────────────────────

def run_once(run_num: int = 0) -> None:
    now = datetime.now(timezone.utc)
    now_ts = now.replace(microsecond=0).isoformat()
    logging.info("── Run #%d  %s ──", run_num, now_ts)

    with get_conn() as conn:

        # ── Portfolio state ──────────────────────────────────────────────────
        free_cash = get_state(conn, "free_cash", PAPER_START_COINS)
        start_eq  = get_state(conn, "start_equity", PAPER_START_COINS)

        # ── Spread refresh (every run, one request) ──────────────────────────
        if spread_age_seconds(conn) > 300:  # refresh if >5 min stale
            logging.info("Fetching Coflnet spread…")
            spread_rows = fetch_spread()
            if spread_rows:
                upsert_spread_rows(conn, spread_rows)
                conn.commit()  # persist spread immediately
                logging.info("Spread: %d items stored", len(spread_rows))

        spread_rows_db = latest_spread(conn)
        spread_map: dict[str, dict] = {
            r["item_id"]: dict(r) for r in spread_rows_db
        }

        # ── Watchlist (rebuild every 30 min or if empty) ─────────────────────
        watchlist: list[str] = get_state(conn, "watchlist", [])
        wl_age = get_state(conn, "watchlist_age_ts", "2000-01-01T00:00:00+00:00")
        wl_ts = datetime.fromisoformat(wl_age)
        if wl_ts.tzinfo is None:
            wl_ts = wl_ts.replace(tzinfo=timezone.utc)
        wl_stale = (now - wl_ts).total_seconds() > 1800 or not watchlist

        if wl_stale and spread_map:
            watchlist = _build_watchlist(list(spread_map.values()))
            set_state(conn, "watchlist", watchlist)
            set_state(conn, "watchlist_age_ts", now_ts)
            conn.commit()  # persist watchlist immediately
            logging.info("Watchlist updated: %d items", len(watchlist))

        # First-ever run: force week-history bootstrap
        is_first_run = not get_state(conn, "bootstrapped")
        if is_first_run:
            logging.info("First run — bootstrapping week-history for %d items…", len(watchlist))
            # Mark bootstrapped BEFORE fetching so a mid-run kill doesn't
            # cause a full re-bootstrap on the next start.
            set_state(conn, "bootstrapped", True)
            conn.commit()

        refreshed = maybe_refresh_history(conn, watchlist, force_week=is_first_run)

        # ── Hypixel snapshot ─────────────────────────────────────────────────
        price_map = ingest_hypixel(conn)
        # Note: buy_wk/sell_wk now populated directly from Hypixel quick_status
        # (no longer need _enrich_price_map_with_vol)

        # ── Update order fills ───────────────────────────────────────────────
        free_cash, events = update_fills(conn, price_map, now, free_cash)

        if events:
            print()
            for evt in events:
                print_event(evt)

        # ── Compute portfolio ────────────────────────────────────────────────
        holdings_val = compute_holdings_value(conn, price_map)
        equity       = free_cash + holdings_val

        print()
        print_header(equity, start_eq, free_cash, holdings_val)

        # ── Current occupied items (open buy + sell orders) ──────────────────
        open_orders   = get_open_orders(conn)
        occupied      = {o["item_id"] for o in open_orders}
        n_buy_open    = sum(1 for o in open_orders if o["side"] == "BUY")
        n_sell_open   = sum(1 for o in open_orders if o["side"] == "SELL")
        available_slots = max(0, MAX_OPEN_POSITIONS - n_sell_open - n_buy_open)

        logging.info(
            "Portfolio | equity=%s  cash=%s  held=%s  buy_orders=%d  sell_orders=%d  slots=%d",
            _fmt_coins(equity), _fmt_coins(free_cash), _fmt_coins(holdings_val),
            n_buy_open, n_sell_open, available_slots,
        )

        # ── Generate features for watchlist ──────────────────────────────────
        features_list: list[dict] = []
        for item_id in watchlist:
            feat = compute_features(conn, item_id)
            if feat:
                # Merge spread data
                sd = spread_map.get(item_id, {})
                feat["is_manipulated"] = sd.get("is_manipulated", 0)
                feat["est_buy_fill_s"] = sd.get("est_buy_fill_s") or 0
                features_list.append(feat)

        logging.info("Features computed for %d/%d watchlist items", len(features_list), len(watchlist))

        # ── Generate buy signals ──────────────────────────────────────────────
        if available_slots > 0 and free_cash > 50_000:
            signals = generate_signals(features_list, spread_map, occupied, free_cash, now_ts)
            signals = signals[:available_slots]  # honour slot limit

            if signals:
                print(f"\n  ── {len(signals)} new signal(s) ──")
            else:
                print("\n  (no new BUY signals this run)")

            with _signal_file().open("a", encoding="utf-8") as f:
                f.write(f"\n{'─'*70}\n")
                f.write(f"Run #{run_num}  {now_ts}  "
                        f"equity={_fmt_coins(equity)}  slots={available_slots}\n")

            for sig in signals:
                # Log to terminal
                print_signal(sig)

                # Deduct from free_cash (reserve funds for order)
                order_cost = sig["qty"] * sig["order_price"]
                if order_cost > free_cash:
                    logging.warning(
                        "Insufficient cash for %s (need %s, have %s)",
                        sig["item_id"], _fmt_coins(order_cost), _fmt_coins(free_cash),
                    )
                    continue

                free_cash -= order_cost

                # Create order in DB
                order_id = create_order(conn, {
                    "item_id":      sig["item_id"],
                    "side":         "BUY",
                    "order_price":  sig["order_price"],
                    "qty":          sig["qty"],
                    "target_price": sig["target_price"],
                    "stop_price":   sig["stop_price"],
                })

                # Log signal
                log_signal(conn, {
                    **sig,
                    "reasoning": sig["reasoning"][:500],
                })
                _write_signal_log(sig)

                logging.info(
                    "New order #%d: BUY %s  qty=%d  price=%.1f  target=%.1f",
                    order_id, sig["item_id"], sig["qty"],
                    sig["order_price"], sig["target_price"],
                )
        else:
            if available_slots <= 0:
                print("\n  (max positions reached; no new buys)")
            else:
                print("\n  (insufficient free cash)")

        # ── Persist portfolio state ───────────────────────────────────────────
        set_state(conn, "free_cash", free_cash)
        if not get_state(conn, "start_equity"):
            set_state(conn, "start_equity", PAPER_START_COINS)

        log_portfolio(conn, free_cash, holdings_val)

    print(LINE)
    print(f"  Watchlist: {len(watchlist)} items  |  "
          f"Open buy orders: {n_buy_open}  |  Sell orders: {n_sell_open}")
    print(f"  History refreshed: {refreshed} items this run")
    print(LINE)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    init_db()

    parser = argparse.ArgumentParser(description="SkyBlock Investor v2")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_num = 0
    while True:
        try:
            run_once(run_num)
        except KeyboardInterrupt:
            logging.info("Stopped by user.")
            break
        except Exception as exc:
            logging.exception("Run #%d failed: %s", run_num, exc)

        run_num += 1

        if args.once:
            break

        next_run = datetime.now(timezone.utc).replace(microsecond=0)
        logging.info("Sleeping %ds until next run…", RUN_INTERVAL_S)
        print(f"\n  Next run in {RUN_INTERVAL_S}s  "
              f"({datetime.now(timezone.utc).strftime('%H:%M:%S UTC')})\n")
        time.sleep(RUN_INTERVAL_S)


if __name__ == "__main__":
    main()
