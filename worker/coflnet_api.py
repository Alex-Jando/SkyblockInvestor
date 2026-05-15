"""
Coflnet sky.coflnet.com API client.

Rate limits (IP-based):
  - 30 requests per 10 seconds
  - 100 requests per minute
"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import requests

BASE = "https://sky.coflnet.com/api"
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "SkyblockInvestor/2.0"


# ── Rate limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """Sliding-window rate limiter: 28/10s and 95/60s (slight headroom below hard limits)."""

    def __init__(self, per_10s: int = 28, per_60s: int = 95) -> None:
        self._per_10s = per_10s
        self._per_60s = per_60s
        self._w10: deque[float] = deque()
        self._w60: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        # Purge expired timestamps
        while self._w10 and now - self._w10[0] >= 10.0:
            self._w10.popleft()
        while self._w60 and now - self._w60[0] >= 60.0:
            self._w60.popleft()

        # Wait if needed (10s window)
        if len(self._w10) >= self._per_10s:
            wait = 10.0 - (now - self._w10[0]) + 0.05
            if wait > 0:
                logging.debug("Rate-limit 10s: sleeping %.2fs", wait)
                time.sleep(wait)
                now = time.monotonic()
                while self._w10 and now - self._w10[0] >= 10.0:
                    self._w10.popleft()

        # Wait if needed (60s window)
        if len(self._w60) >= self._per_60s:
            wait = 60.0 - (now - self._w60[0]) + 0.05
            if wait > 0:
                logging.debug("Rate-limit 60s: sleeping %.2fs", wait)
                time.sleep(wait)
                now = time.monotonic()
                while self._w60 and now - self._w60[0] >= 60.0:
                    self._w60.popleft()

        now = time.monotonic()
        self._w10.append(now)
        self._w60.append(now)


_limiter = _RateLimiter()


def _get(path: str, params: dict | None = None, retries: int = 3) -> Any:
    url = f"{BASE}{path}"
    for attempt in range(1, retries + 1):
        _limiter.acquire()
        try:
            resp = _SESSION.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 10 * attempt
                logging.warning("429 from Coflnet, sleeping %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            logging.warning("Coflnet request failed (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(2 ** attempt)
    return None  # unreachable


# ── Public endpoints ──────────────────────────────────────────────────────────

def fetch_spread() -> list[dict]:
    """
    GET /api/flip/bazaar/spread
    Returns all Bazaar items with spread, fill estimates, and manipulation flag.
    One request covers all items.
    """
    now_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    data = _get("/flip/bazaar/spread")
    rows: list[dict] = []
    for entry in data or []:
        flip = entry.get("flip") or {}
        tag = flip.get("itemTag", "")
        if not tag:
            continue
        rows.append({
            "ts": now_ts,
            "item_id": tag,
            "buy_price": flip.get("buyPrice"),
            "sell_price": flip.get("sellPrice"),
            "median_buy_price": flip.get("medianBuyPrice"),
            "profit_per_hour": flip.get("profitPerHour"),
            "est_buy_fill_s": flip.get("estimatedBuyFillSeconds"),
            "est_sell_fill_s": flip.get("estimatedSellFillSeconds"),
            "is_manipulated": 1 if entry.get("isManipulated") else 0,
            "item_name": entry.get("itemName", tag),
        })
    return rows


def fetch_history_day(item_id: str) -> list[dict]:
    """
    GET /api/bazaar/{itemTag}/history/day
    5-minute interval OHLC for the last 24h (~288 points).
    """
    try:
        data = _get(f"/bazaar/{item_id}/history/day")
    except Exception as exc:
        logging.warning("Coflnet day-history failed for %s: %s", item_id, exc)
        return []
    return _parse_history(item_id, data or [], interval_m=5)


def fetch_history_week(item_id: str) -> list[dict]:
    """
    GET /api/bazaar/{itemTag}/history/week
    2-hour interval OHLC for the last 7 days (~84 points).
    """
    try:
        data = _get(f"/bazaar/{item_id}/history/week")
    except Exception as exc:
        logging.warning("Coflnet week-history failed for %s: %s", item_id, exc)
        return []
    return _parse_history(item_id, data or [], interval_m=120)


def fetch_bazaar_tags() -> list[str]:
    """GET /api/items/bazaar/tags — list of all tradeable item tags."""
    return _get("/items/bazaar/tags") or []


def _parse_history(item_id: str, data: list[dict], interval_m: int) -> list[dict]:
    rows: list[dict] = []
    for pt in data:
        ts = pt.get("timestamp", "")
        if not ts:
            continue
        # Normalise timestamp to UTC isoformat without microseconds
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts = dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
        rows.append({
            "ts": ts,
            "item_id": item_id,
            "source": "coflnet",
            "interval_m": interval_m,
            "buy_price": pt.get("buy"),
            "sell_price": pt.get("sell"),
            "buy_vol_q": pt.get("buyVolume"),
            "sell_vol_q": pt.get("sellVolume"),
            "buy_wk": pt.get("buyMovingWeek"),
            "sell_wk": pt.get("sellMovingWeek"),
        })
    return rows
