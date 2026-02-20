from __future__ import annotations

from datetime import date, datetime
from typing import Any

EPS = 1e-9


def simulate_daily_rebalance(
    day: date,
    ts: datetime,
    buy_items: list[dict[str, Any]],
    current_prices: dict[str, dict[str, float]],
    previous_holdings: list[dict[str, Any]],
    previous_equity: dict[str, Any] | None,
    start_equity: float,
    historical_peak_equity: float,
    previous_max_drawdown_pct: float,
) -> tuple[dict[str, float | date | datetime], list[dict[str, float | date | str]]]:
    previous_equity_value = (
        float(previous_equity["equity_value"]) if previous_equity is not None else float(start_equity)
    )

    if previous_holdings:
        cash = 0.0
        for holding in previous_holdings:
            item_id = str(holding["item_id"])
            qty = float(holding["qty"])
            default_sell = float(holding["market_value"]) / max(qty, EPS)
            sell_price = current_prices.get(item_id, {}).get("sell_price", default_sell)
            cash += qty * max(sell_price, 0.0)
    else:
        cash = previous_equity_value

    equity_before_buy = cash

    new_holdings: list[dict[str, float | date | str]] = []
    holdings_value = 0.0

    for item in buy_items:
        item_id = str(item["item_id"])
        weight_pct = float(item["weight_pct"])
        if weight_pct <= 0:
            continue

        price_info = current_prices.get(item_id)
        if not price_info:
            continue

        buy_price = float(price_info.get("buy_price", 0.0))
        sell_price = float(price_info.get("sell_price", 0.0))
        if buy_price <= 0 or sell_price <= 0:
            continue

        allocation_coins = equity_before_buy * weight_pct
        qty = allocation_coins / buy_price
        market_value = qty * sell_price

        cash -= allocation_coins
        holdings_value += market_value

        new_holdings.append(
            {
                "day": day,
                "item_id": item_id,
                "qty": qty,
                "cost_basis": buy_price,
                "market_value": market_value,
            }
        )

    equity_value = cash + holdings_value
    cumulative_return_pct = (equity_value / max(start_equity, EPS) - 1.0) * 100.0

    if previous_equity is None:
        daily_return_pct = 0.0
    else:
        daily_return_pct = (equity_value / max(previous_equity_value, EPS) - 1.0) * 100.0

    peak_equity = max(historical_peak_equity, equity_value, start_equity)
    current_drawdown_pct = (equity_value / max(peak_equity, EPS) - 1.0) * 100.0
    max_drawdown_pct = min(previous_max_drawdown_pct, current_drawdown_pct)

    equity_row: dict[str, float | date | datetime] = {
        "ts": ts,
        "day": day,
        "equity_value": equity_value,
        "cash_value": cash,
        "holdings_value": holdings_value,
        "cumulative_return_pct": cumulative_return_pct,
        "daily_return_pct": daily_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }

    return equity_row, new_holdings
