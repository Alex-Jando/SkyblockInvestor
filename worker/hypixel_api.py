from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

API_URL = "https://api.hypixel.net/skyblock/bazaar"
EPS = 1e-9


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN check
        return None
    return parsed


def fetch_bazaar(api_key: str, max_retries: int = 5, timeout_seconds: int = 30) -> dict[str, Any]:
    if not api_key:
        raise ValueError("HYPIXEL_API_KEY is required.")

    session = requests.Session()
    backoff_seconds = 1.0
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                API_URL,
                params={"key": api_key},
                timeout=timeout_seconds,
            )

            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"Hypixel API returned retryable status {response.status_code}"
                )

            response.raise_for_status()
            payload = response.json()

            if not payload.get("success", False):
                raise ValueError("Hypixel API payload indicates success=false.")

            products = payload.get("products")
            if not isinstance(products, dict) or not products:
                raise ValueError("Hypixel API payload missing products.")

            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_for = backoff_seconds * (1 + 0.1 * attempt)
            logging.warning(
                "Hypixel fetch failed on attempt %s/%s: %s. Retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
            backoff_seconds *= 2

    raise RuntimeError(f"Failed to fetch Hypixel bazaar data: {last_error}") from last_error


def parse_snapshot_rows(
    payload: dict[str, Any], run_ts: datetime | None = None
) -> tuple[list[dict[str, Any]], int]:
    ts = run_ts or datetime.now(timezone.utc)
    day = ts.date()

    rows: list[dict[str, Any]] = []
    skipped = 0

    products = payload.get("products", {})
    for item_id, product in products.items():
        if not isinstance(product, dict):
            skipped += 1
            continue

        quick_status = product.get("quick_status", {})
        if not isinstance(quick_status, dict):
            skipped += 1
            continue

        buy_price = _to_float(quick_status.get("buyPrice"))
        sell_price = _to_float(quick_status.get("sellPrice"))
        buy_volume = _to_float(quick_status.get("buyVolume"))
        sell_volume = _to_float(quick_status.get("sellVolume"))

        if buy_price is None or sell_price is None or buy_price <= 0 or sell_price <= 0:
            skipped += 1
            continue

        mid_price = (buy_price + sell_price) / 2.0
        if mid_price <= EPS:
            skipped += 1
            continue

        rows.append(
            {
                "ts": ts,
                "day": day,
                "item_id": str(item_id),
                "buy_price": buy_price,
                "sell_price": sell_price,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "mid_price": mid_price,
            }
        )

    return rows, skipped
