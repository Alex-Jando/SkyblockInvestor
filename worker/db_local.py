from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "local.db"

SCHEMA = """
-- All price snapshots (Hypixel spot + Coflnet historical intervals)
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    item_id     TEXT    NOT NULL,
    source      TEXT    NOT NULL,           -- 'hypixel' | 'coflnet'
    interval_m  INTEGER NOT NULL DEFAULT 0, -- 0=spot, 5=5min, 120=2h
    buy_price   REAL,
    sell_price  REAL,
    buy_vol_q   INTEGER,                    -- queue depth (order book)
    sell_vol_q  INTEGER,
    buy_wk      INTEGER,                    -- buyMovingWeek  (weekly traded units)
    sell_wk     INTEGER,                    -- sellMovingWeek
    UNIQUE(item_id, ts, source, interval_m)
);
CREATE INDEX IF NOT EXISTS idx_ph_item_ts  ON price_history(item_id, ts);
CREATE INDEX IF NOT EXISTS idx_ph_ts_desc  ON price_history(ts DESC);

-- Coflnet spread snapshots (one row per item per fetch)
CREATE TABLE IF NOT EXISTS spread_snapshot (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                    TEXT    NOT NULL,
    item_id               TEXT    NOT NULL,
    buy_price             REAL,
    sell_price            REAL,
    median_buy_price      REAL,
    profit_per_hour       REAL,
    est_buy_fill_s        REAL,
    est_sell_fill_s       REAL,
    is_manipulated        INTEGER NOT NULL DEFAULT 0,
    item_name             TEXT,
    UNIQUE(item_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_ss_ts_desc ON spread_snapshot(ts DESC);

-- Orders: both BUY (entry) and SELL (exit) sides
CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    item_id         TEXT    NOT NULL,
    side            TEXT    NOT NULL CHECK(side IN ('BUY','SELL')),
    order_price     REAL    NOT NULL,   -- limit price we placed
    qty_ordered     REAL    NOT NULL,
    qty_filled      REAL    NOT NULL DEFAULT 0,
    cost_basis_avg  REAL,               -- avg fill price (null until first fill)
    status          TEXT    NOT NULL DEFAULT 'OPEN'
                    CHECK(status IN ('OPEN','PARTIAL','FILLED','CANCELLED','EXPIRED')),
    parent_order_id       INTEGER,        -- SELL order's linked BUY order id
    target_price          REAL,
    stop_price            REAL,
    exit_reason           TEXT,
    closed_at             TEXT,
    original_order_price  REAL,           -- price at initial placement (pre-reprice)
    reprice_count         INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (parent_order_id) REFERENCES orders(id)
);
CREATE INDEX IF NOT EXISTS idx_orders_open   ON orders(status) WHERE status IN ('OPEN','PARTIAL');
CREATE INDEX IF NOT EXISTS idx_orders_item   ON orders(item_id, status);

-- Portfolio snapshots over time
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    free_cash       REAL    NOT NULL,
    holdings_value  REAL    NOT NULL,
    equity          REAL    NOT NULL,
    n_buy_orders    INTEGER NOT NULL DEFAULT 0,
    n_sell_orders   INTEGER NOT NULL DEFAULT 0,
    daily_pnl_pct   REAL
);

-- All signals generated (for logging / backtesting)
CREATE TABLE IF NOT EXISTS signals_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL,
    item_id             TEXT    NOT NULL,
    action              TEXT    NOT NULL,   -- 'BUY' | 'SKIP' | 'HOLD'
    order_price         REAL,
    target_price        REAL,
    stop_price          REAL,
    qty                 REAL,
    confidence          REAL,
    expected_net_return REAL,
    expected_fill_hours REAL,
    expected_hold_hours REAL,
    reasoning           TEXT
);

-- Key-value store for persistent state
CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Order book depth snapshots (top levels from Hypixel per run)
-- side='SELL' rows = offers to sell (relevant for our BUY orders)
-- side='BUY'  rows = offers to buy  (relevant for our SELL orders)
CREATE TABLE IF NOT EXISTS order_book_snapshot (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT    NOT NULL,
    item_id TEXT    NOT NULL,
    side    TEXT    NOT NULL CHECK(side IN ('BUY','SELL')),
    price   REAL    NOT NULL,
    amount  INTEGER NOT NULL,
    UNIQUE(ts, item_id, side, price)
);
CREATE INDEX IF NOT EXISTS idx_obs_item_ts ON order_book_snapshot(item_id, ts DESC);
"""


def _db_path() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(_db_path()), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Idempotent migrations for columns added after initial release
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema without dropping existing data."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(orders)")}
    if "original_order_price" not in existing:
        conn.execute("ALTER TABLE orders ADD COLUMN original_order_price REAL")
    if "reprice_count" not in existing:
        conn.execute("ALTER TABLE orders ADD COLUMN reprice_count INTEGER NOT NULL DEFAULT 0")
    conn.commit()


# ── Price history ────────────────────────────────────────────────────────────

def upsert_price_rows(conn: sqlite3.Connection, rows: list[dict]) -> int:
    cur = conn.executemany(
        """INSERT OR IGNORE INTO price_history
           (ts, item_id, source, interval_m, buy_price, sell_price,
            buy_vol_q, sell_vol_q, buy_wk, sell_wk)
           VALUES (:ts,:item_id,:source,:interval_m,:buy_price,:sell_price,
                   :buy_vol_q,:sell_vol_q,:buy_wk,:sell_wk)""",
        rows,
    )
    return cur.rowcount


def get_price_history(
    conn: sqlite3.Connection,
    item_id: str,
    interval_m: int = 5,
    hours: int = 24,
) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM price_history
           WHERE item_id=? AND interval_m=?
             AND ts >= datetime('now',?)
           ORDER BY ts ASC""",
        (item_id, interval_m, f"-{hours} hours"),
    ).fetchall()


def latest_hypixel_snapshot(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return the most-recent Hypixel spot row for every item."""
    return conn.execute(
        """SELECT ph.*
           FROM price_history ph
           INNER JOIN (
               SELECT item_id, MAX(ts) ts
               FROM price_history WHERE source='hypixel'
               GROUP BY item_id
           ) latest USING(item_id, ts)""",
    ).fetchall()


# ── Spread data ──────────────────────────────────────────────────────────────

def upsert_spread_rows(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR IGNORE INTO spread_snapshot
           (ts, item_id, buy_price, sell_price, median_buy_price,
            profit_per_hour, est_buy_fill_s, est_sell_fill_s,
            is_manipulated, item_name)
           VALUES (:ts,:item_id,:buy_price,:sell_price,:median_buy_price,
                   :profit_per_hour,:est_buy_fill_s,:est_sell_fill_s,
                   :is_manipulated,:item_name)""",
        rows,
    )


def latest_spread(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM spread_snapshot
           WHERE ts=(SELECT MAX(ts) FROM spread_snapshot)""",
    ).fetchall()


def spread_age_seconds(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT MAX(ts) ts FROM spread_snapshot").fetchone()
    if not row or not row["ts"]:
        return float("inf")
    ts = datetime.fromisoformat(row["ts"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds()


# ── Orders ───────────────────────────────────────────────────────────────────

def create_order(conn: sqlite3.Connection, order: dict) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO orders
           (created_at, updated_at, item_id, side, order_price, qty_ordered,
            qty_filled, cost_basis_avg, status, parent_order_id,
            target_price, stop_price, original_order_price, reprice_count)
           VALUES (?,?,?,?,?,?, 0,NULL,'OPEN',?,?,?,?,0)""",
        (
            now, now,
            order["item_id"], order["side"], order["order_price"], order["qty"],
            order.get("parent_order_id"),
            order.get("target_price"), order.get("stop_price"),
            order["order_price"],  # original_order_price = initial price
        ),
    )
    return cur.lastrowid


def reprice_order(
    conn: sqlite3.Connection,
    order_id: int,
    new_price: float,
) -> None:
    """Update the limit price of an open order and increment reprice_count."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """UPDATE orders
           SET order_price=?,
               updated_at=?,
               reprice_count=reprice_count+1
           WHERE id=?""",
        (new_price, now, order_id),
    )


def get_open_orders(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM orders WHERE status IN ('OPEN','PARTIAL') ORDER BY created_at"
    ).fetchall()


def update_order(
    conn: sqlite3.Connection,
    order_id: int,
    *,
    qty_filled: float | None = None,
    cost_basis_avg: float | None = None,
    status: str | None = None,
    exit_reason: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    closed_at = now if status in ("FILLED", "CANCELLED", "EXPIRED") else None
    conn.execute(
        """UPDATE orders
           SET qty_filled=COALESCE(?,qty_filled),
               cost_basis_avg=COALESCE(?,cost_basis_avg),
               status=COALESCE(?,status),
               exit_reason=COALESCE(?,exit_reason),
               updated_at=?,
               closed_at=COALESCE(?,closed_at)
           WHERE id=?""",
        (qty_filled, cost_basis_avg, status, exit_reason, now, closed_at, order_id),
    )


# ── Order book snapshots ──────────────────────────────────────────────────────

def upsert_order_book(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """Store order book levels. rows: [{ts, item_id, side, price, amount}]"""
    conn.executemany(
        """INSERT OR IGNORE INTO order_book_snapshot (ts, item_id, side, price, amount)
           VALUES (:ts, :item_id, :side, :price, :amount)""",
        rows,
    )


def order_book_depth_at_price(
    conn: sqlite3.Connection,
    item_id: str,
    side: str,
    order_price: float,
    ts: str,
) -> float:
    """
    Return total volume available on the opposite side at/better-than order_price
    at the given snapshot timestamp.

    For a BUY order at price P:  side='SELL', sum amount WHERE price <= P
    For a SELL order at price P: side='BUY',  sum amount WHERE price >= P
    """
    if side == "SELL":
        row = conn.execute(
            """SELECT COALESCE(SUM(amount), 0) total
               FROM order_book_snapshot
               WHERE item_id=? AND side='SELL' AND price<=? AND ts=?""",
            (item_id, order_price, ts),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT COALESCE(SUM(amount), 0) total
               FROM order_book_snapshot
               WHERE item_id=? AND side='BUY' AND price>=? AND ts=?""",
            (item_id, order_price, ts),
        ).fetchone()
    return float(row[0]) if row else 0.0


def two_latest_order_book_ts(conn: sqlite3.Connection, item_id: str) -> tuple[str | None, str | None]:
    """Return (current_ts, previous_ts) for order book snapshots for this item."""
    rows = conn.execute(
        """SELECT DISTINCT ts FROM order_book_snapshot
           WHERE item_id=? ORDER BY ts DESC LIMIT 2""",
        (item_id,),
    ).fetchall()
    curr = rows[0][0] if len(rows) >= 1 else None
    prev = rows[1][0] if len(rows) >= 2 else None
    return curr, prev


def prune_old_order_book(conn: sqlite3.Connection, keep_hours: int = 4) -> None:
    """Delete order book snapshots older than keep_hours to keep DB lean."""
    conn.execute(
        "DELETE FROM order_book_snapshot WHERE ts < datetime('now', ?)",
        (f"-{keep_hours} hours",),
    )


# ── Portfolio ────────────────────────────────────────────────────────────────

def log_portfolio(
    conn: sqlite3.Connection,
    free_cash: float,
    holdings_value: float,
    daily_pnl_pct: float | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    n_buy = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE side='BUY' AND status IN ('OPEN','PARTIAL')"
    ).fetchone()[0]
    n_sell = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE side='SELL' AND status IN ('OPEN','PARTIAL')"
    ).fetchone()[0]
    conn.execute(
        """INSERT INTO portfolio_snapshots
           (ts,free_cash,holdings_value,equity,n_buy_orders,n_sell_orders,daily_pnl_pct)
           VALUES (?,?,?,?,?,?,?)""",
        (now, free_cash, holdings_value, free_cash + holdings_value,
         n_buy, n_sell, daily_pnl_pct),
    )


# ── Signals ──────────────────────────────────────────────────────────────────

def log_signal(conn: sqlite3.Connection, sig: dict) -> None:
    conn.execute(
        """INSERT INTO signals_log
           (ts,item_id,action,order_price,target_price,stop_price,qty,
            confidence,expected_net_return,expected_fill_hours,expected_hold_hours,reasoning)
           VALUES (:ts,:item_id,:action,:order_price,:target_price,:stop_price,:qty,
                   :confidence,:expected_net_return,:expected_fill_hours,
                   :expected_hold_hours,:reasoning)""",
        sig,
    )


# ── App state (key/value) ────────────────────────────────────────────────────

def get_state(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute(
        "SELECT value FROM app_state WHERE key=?", (key,)
    ).fetchone()
    return json.loads(row["value"]) if row else default


def set_state(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO app_state(key,value) VALUES(?,?)",
        (key, json.dumps(value)),
    )
