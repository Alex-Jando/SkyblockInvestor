from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row


def get_connection(database_url: str | None = None) -> psycopg.Connection:
    dsn = database_url or os.environ.get("SUPABASE_DATABASE_URL")
    if not dsn:
        raise ValueError("SUPABASE_DATABASE_URL is required.")
    return psycopg.connect(dsn, row_factory=dict_row, prepare_threshold=None)


def upsert_bazaar_snapshots(conn: psycopg.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    sql = """
        insert into bazaar_snapshots (
          ts, day, item_id, buy_price, sell_price, buy_volume, sell_volume, mid_price
        )
        values (
          %(ts)s, %(day)s, %(item_id)s, %(buy_price)s, %(sell_price)s, %(buy_volume)s, %(sell_volume)s, %(mid_price)s
        )
        on conflict (item_id, day) do update
        set
          ts = excluded.ts,
          buy_price = excluded.buy_price,
          sell_price = excluded.sell_price,
          buy_volume = excluded.buy_volume,
          sell_volume = excluded.sell_volume,
          mid_price = excluded.mid_price
    """

    with conn.cursor() as cur:
        cur.executemany(sql, rows)


def load_snapshots_history(conn: psycopg.Connection) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """
            select
              ts, day, item_id, buy_price, sell_price, buy_volume, sell_volume, mid_price
            from bazaar_snapshots
            order by day asc, item_id asc
            """
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def replace_item_signals(
    conn: psycopg.Connection,
    day: date,
    ts: datetime,
    rows: list[dict[str, Any]],
) -> None:
    with conn.cursor() as cur:
        cur.execute("delete from item_signals where day = %s", (day,))
        if not rows:
            return

        payload = []
        for row in rows:
            payload.append(
                {
                    "ts": ts,
                    "day": day,
                    "horizon_days": int(row["horizon_days"]),
                    "item_id": str(row["item_id"]),
                    "expected_return": float(row["expected_return"]),
                    "confidence": float(row["confidence"]),
                    "liquidity_score": float(row["liquidity_score"]),
                    "spread_pct": float(row["spread_pct"]),
                    "imbalance": float(row["imbalance"]),
                    "volatility_30d": float(row["volatility_30d"]),
                    "max_alloc_pct_feasible": float(row["max_alloc_pct_feasible"]),
                    "model_version": str(row["model_version"]),
                }
            )

        cur.executemany(
            """
            insert into item_signals (
              ts, day, horizon_days, item_id, expected_return, confidence, liquidity_score,
              spread_pct, imbalance, volatility_30d, max_alloc_pct_feasible, model_version
            )
            values (
              %(ts)s, %(day)s, %(horizon_days)s, %(item_id)s, %(expected_return)s, %(confidence)s, %(liquidity_score)s,
              %(spread_pct)s, %(imbalance)s, %(volatility_30d)s, %(max_alloc_pct_feasible)s, %(model_version)s
            )
            """,
            payload,
        )


def upsert_basket_and_items(
    conn: psycopg.Connection,
    ts: datetime,
    day: date,
    decision_horizon_days: int,
    model_version: str,
    notes: str,
    buy_items: list[dict[str, Any]],
    sell_items: list[dict[str, Any]],
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into baskets (ts, day, decision_horizon_days, model_version, notes)
            values (%s, %s, %s, %s, %s)
            on conflict (day) do update
            set
              ts = excluded.ts,
              decision_horizon_days = excluded.decision_horizon_days,
              model_version = excluded.model_version,
              notes = excluded.notes
            returning id
            """,
            (ts, day, decision_horizon_days, model_version, notes),
        )
        basket_id = int(cur.fetchone()["id"])

        cur.execute("delete from basket_items where basket_id = %s", (basket_id,))

        item_rows = []
        for row in buy_items:
            item_rows.append(
                (
                    basket_id,
                    str(row["item_id"]),
                    "BUY",
                    float(row["weight_pct"]),
                    float(row["expected_return"]),
                    float(row["confidence"]),
                    float(row["liquidity_score"]),
                    float(row["spread_pct"]),
                    float(row["max_alloc_pct_feasible"]),
                )
            )
        for row in sell_items:
            item_rows.append(
                (
                    basket_id,
                    str(row["item_id"]),
                    "SELL",
                    0.0,
                    float(row["expected_return"]),
                    float(row["confidence"]),
                    float(row["liquidity_score"]),
                    float(row["spread_pct"]),
                    float(row["max_alloc_pct_feasible"]),
                )
            )

        if item_rows:
            cur.executemany(
                """
                insert into basket_items (
                  basket_id, item_id, action, weight_pct, expected_return, confidence,
                  liquidity_score, spread_pct, max_alloc_pct_feasible
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                item_rows,
            )

    return basket_id


def load_current_prices(conn: psycopg.Connection, day: date) -> dict[str, dict[str, float]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select item_id, buy_price, sell_price
            from bazaar_snapshots
            where day = %s
            """,
            (day,),
        )
        rows = cur.fetchall()

    return {
        str(row["item_id"]): {
            "buy_price": float(row["buy_price"]),
            "sell_price": float(row["sell_price"]),
        }
        for row in rows
    }


def fetch_previous_holdings(conn: psycopg.Connection, day: date) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select day
            from paper_portfolio_holdings
            where day < %s
            order by day desc
            limit 1
            """,
            (day,),
        )
        latest = cur.fetchone()
        if latest is None:
            return []

        prev_day = latest["day"]
        cur.execute(
            """
            select item_id, qty, cost_basis, market_value
            from paper_portfolio_holdings
            where day = %s
            order by item_id
            """,
            (prev_day,),
        )
        return cur.fetchall()


def fetch_previous_equity(conn: psycopg.Connection, day: date) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            select day, equity_value, cash_value, holdings_value, cumulative_return_pct, daily_return_pct, max_drawdown_pct
            from paper_portfolio_equity
            where day < %s
            order by day desc
            limit 1
            """,
            (day,),
        )
        return cur.fetchone()


def fetch_portfolio_state(conn: psycopg.Connection, day: date, start_equity: float) -> tuple[float, float]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select coalesce(max(equity_value), %s) as peak
            from paper_portfolio_equity
            where day < %s
            """,
            (start_equity, day),
        )
        peak = float(cur.fetchone()["peak"])

        cur.execute(
            """
            select max_drawdown_pct
            from paper_portfolio_equity
            where day < %s
            order by day desc
            limit 1
            """,
            (day,),
        )
        latest = cur.fetchone()
        previous_mdd = float(latest["max_drawdown_pct"]) if latest is not None else 0.0

    return peak, previous_mdd


def upsert_equity(conn: psycopg.Connection, row: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into paper_portfolio_equity (
              ts, day, equity_value, cash_value, holdings_value,
              cumulative_return_pct, daily_return_pct, max_drawdown_pct
            )
            values (%(ts)s, %(day)s, %(equity_value)s, %(cash_value)s, %(holdings_value)s,
                    %(cumulative_return_pct)s, %(daily_return_pct)s, %(max_drawdown_pct)s)
            on conflict (day) do update
            set
              ts = excluded.ts,
              equity_value = excluded.equity_value,
              cash_value = excluded.cash_value,
              holdings_value = excluded.holdings_value,
              cumulative_return_pct = excluded.cumulative_return_pct,
              daily_return_pct = excluded.daily_return_pct,
              max_drawdown_pct = excluded.max_drawdown_pct
            """,
            row,
        )


def replace_holdings(conn: psycopg.Connection, day: date, holdings: list[dict[str, Any]]) -> None:
    with conn.cursor() as cur:
        cur.execute("delete from paper_portfolio_holdings where day = %s", (day,))
        if not holdings:
            return
        cur.executemany(
            """
            insert into paper_portfolio_holdings (day, item_id, qty, cost_basis, market_value)
            values (%(day)s, %(item_id)s, %(qty)s, %(cost_basis)s, %(market_value)s)
            """,
            holdings,
        )
