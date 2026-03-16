from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _normalize_with_caps(scores: np.ndarray, caps: np.ndarray, total_target: float) -> np.ndarray:
    if scores.size == 0 or total_target <= 0:
        return np.array([])
    scores = np.maximum(scores, 0.0)
    if scores.sum() <= EPS:
        scores = np.ones_like(scores)

    weights = total_target * scores / (scores.sum() + EPS)
    weights = np.minimum(weights, caps)
    remaining = total_target - weights.sum()
    if remaining > 1e-9:
        room = np.maximum(caps - weights, 0.0)
        if room.sum() > EPS:
            weights += remaining * room / (room.sum() + EPS)
    return weights


def _metric_lookup(safe_universe: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    if safe_universe.empty:
        return metrics

    cols = [
        "item_id",
        "score",
        "rank_score",
        "expected_return_7d",
        "confidence_7d",
        "return_3d",
        "volume_collapse_frac",
        "spread_pct",
        "max_alloc_pct_feasible",
    ]
    rows = safe_universe[cols].to_dict("records")
    for row in rows:
        item_id = str(row["item_id"])
        metrics[item_id] = {
            "score": float(row.get("score", 0.0)),
            "rank_score": float(row.get("rank_score", 9999.0)),
            "expected_return_7d": float(row.get("expected_return_7d", 0.0)),
            "confidence_7d": float(row.get("confidence_7d", 0.0)),
            "return_3d": float(row.get("return_3d", 0.0)),
            "volume_collapse_frac": float(row.get("volume_collapse_frac", 1.0)),
            "spread_pct": float(row.get("spread_pct", 0.0)),
            "max_alloc_pct_feasible": float(row.get("max_alloc_pct_feasible", 0.0)),
        }
    return metrics


def simulate_daily_rebalance(
    day: date,
    ts: datetime,
    buy_items: list[dict[str, Any]],
    safe_universe: pd.DataFrame,
    current_prices: dict[str, dict[str, float]],
    previous_holdings: list[dict[str, Any]],
    previous_equity: dict[str, Any] | None,
    start_equity: float,
    historical_peak_equity: float,
    previous_max_drawdown_pct: float,
    position_state: dict[str, Any] | None,
    market_regime: str,
    rebalance_band: float = 0.08,
) -> tuple[
    dict[str, float | date | datetime],
    list[dict[str, float | date | str]],
    dict[str, Any],
    dict[str, Any],
]:
    state = position_state or {}
    entry_days_raw = state.get("entry_days", {}) or {}
    cooldowns_raw = state.get("cooldowns", {}) or {}

    entry_days: dict[str, date] = {
        str(k): v for k, v in ((k, _to_date(v)) for k, v in entry_days_raw.items()) if v is not None
    }
    cooldowns: dict[str, date] = {
        str(k): v for k, v in ((k, _to_date(v)) for k, v in cooldowns_raw.items()) if v is not None
    }

    prev_equity_value = float(previous_equity["equity_value"]) if previous_equity is not None else float(start_equity)
    cash = float(previous_equity["cash_value"]) if previous_equity is not None else float(start_equity)

    qty_map: dict[str, float] = {}
    cost_map: dict[str, float] = {}
    initial_qty_map: dict[str, float] = {}
    for holding in previous_holdings:
        item_id = str(holding["item_id"])
        qty = float(holding["qty"])
        if qty <= 0:
            continue
        qty_map[item_id] = qty
        initial_qty_map[item_id] = qty
        cost_map[item_id] = float(holding["cost_basis"])

    marked_holdings_value = 0.0
    for item_id, qty in qty_map.items():
        sell_price = float(current_prices.get(item_id, {}).get("sell_price", 0.0))
        if sell_price > 0:
            marked_holdings_value += qty * sell_price
    equity_before = cash + marked_holdings_value
    if equity_before <= EPS:
        equity_before = prev_equity_value

    metrics = _metric_lookup(safe_universe)

    # Exit / keep evaluation for existing positions.
    hold_threshold = 0.0
    sell_threshold = -0.01
    rank_limit = 5 + 3
    keep_items: set[str] = set()
    exit_reasons: dict[str, str] = {}

    for item_id in list(qty_map.keys()):
        m = metrics.get(
            item_id,
            {
                "score": -1.0,
                "rank_score": 9999.0,
                "expected_return_7d": -0.02,
                "confidence_7d": 0.0,
                "return_3d": -0.01,
                "volume_collapse_frac": 0.0,
                "spread_pct": 0.05,
                "max_alloc_pct_feasible": 0.0,
            },
        )
        entry_day = entry_days.get(item_id, day - timedelta(days=999))
        hold_days = max(0, (day - entry_day).days)

        expected_7d = m["expected_return_7d"]
        confidence_7d = m["confidence_7d"]
        return_3d = m["return_3d"]
        collapse_frac = m["volume_collapse_frac"]
        score = m["score"]
        rank_score = m["rank_score"]

        if hold_days < 2:
            keep_items.add(item_id)
            continue

        reason = None
        if expected_7d < -0.01:
            reason = "negative_expected_return"
        elif confidence_7d < 0.35 and return_3d < 0:
            reason = "low_conf_negative_momentum"
        elif collapse_frac < 0.6:
            reason = "volume_collapse"
        elif score < sell_threshold:
            reason = "score_below_sell_threshold"
        elif rank_score > rank_limit:
            reason = "rank_drop"
        elif hold_days >= 14:
            reason = "max_hold_reeval"

        if reason is not None:
            exit_reasons[item_id] = reason
            continue

        if expected_7d > 0 or score > hold_threshold or confidence_7d >= 0.5:
            keep_items.add(item_id)
        else:
            exit_reasons[item_id] = "weak_hold_signal"

    # Prepare candidate entries from today's BUY list.
    buy_df = pd.DataFrame(buy_items)
    if not buy_df.empty:
        buy_df["item_id"] = buy_df["item_id"].astype(str)
        for col in ["score", "spread_pct", "max_alloc_pct_feasible"]:
            if col not in buy_df.columns:
                buy_df[col] = 0.0
            buy_df[col] = pd.to_numeric(buy_df[col], errors="coerce").fillna(0.0)
        buy_df = buy_df.sort_values("score", ascending=False)

    selected: list[str] = list(keep_items)
    max_holdings = 5

    for _, row in buy_df.iterrows() if not buy_df.empty else []:
        item_id = str(row["item_id"])
        cooldown_until = cooldowns.get(item_id)
        if cooldown_until is not None and day <= cooldown_until:
            continue
        if item_id in selected:
            continue

        new_score = float(row["score"])
        new_spread = float(row["spread_pct"])
        if len(selected) < max_holdings:
            selected.append(item_id)
            continue

        replace_target = None
        replace_score = None
        for cur_item in selected:
            cur_entry_day = entry_days.get(cur_item, day - timedelta(days=999))
            cur_hold_days = max(0, (day - cur_entry_day).days)
            if cur_hold_days < 2:
                continue
            cur_metric = metrics.get(cur_item, {})
            cur_score = float(cur_metric.get("score", -1.0))
            if replace_target is None or cur_score < float(replace_score):
                replace_target = cur_item
                replace_score = cur_score

        if replace_target is None:
            continue
        old_spread = float(metrics.get(replace_target, {}).get("spread_pct", 0.05))
        replacement_margin = max(0.01, 0.75 * old_spread + 0.75 * new_spread)
        if new_score >= float(replace_score) + replacement_margin:
            selected.remove(replace_target)
            selected.append(item_id)
            exit_reasons[replace_target] = "replacement_upgrade"

    selected = selected[:max_holdings]
    selected_count = len(selected)

    if selected_count == 0:
        invest_cap = 0.0
    elif selected_count == 1:
        invest_cap = 0.20
    elif selected_count == 2:
        invest_cap = 0.40
    else:
        invest_cap = 0.80
    if market_regime == "RISK_OFF":
        invest_cap = min(invest_cap, 0.60)

    # Build target weights over selected names.
    if selected_count > 0:
        selected_scores = []
        selected_caps = []
        for item_id in selected:
            metric = metrics.get(item_id, {})
            fallback_score = float(
                buy_df.loc[buy_df["item_id"] == item_id, "score"].iloc[0]
                if not buy_df.empty and (buy_df["item_id"] == item_id).any()
                else 0.0
            )
            score = float(metric.get("score", fallback_score))
            cap = float(metric.get("max_alloc_pct_feasible", 0.30))
            cap = min(0.30, max(0.0, cap))
            if market_regime == "RISK_OFF":
                cap = min(cap, 0.22)
            selected_scores.append(max(score, 0.001))
            selected_caps.append(cap)
        target_weights_arr = _normalize_with_caps(
            np.asarray(selected_scores),
            np.asarray(selected_caps),
            total_target=invest_cap,
        )
    else:
        target_weights_arr = np.array([])
    target_weights = {item_id: float(target_weights_arr[i]) for i, item_id in enumerate(selected)}

    # Execute mandatory exits first.
    trades_executed = 0
    for item_id, reason in list(exit_reasons.items()):
        qty = qty_map.get(item_id, 0.0)
        if qty <= 0:
            continue
        sell_price = float(current_prices.get(item_id, {}).get("sell_price", 0.0))
        if sell_price <= 0:
            continue
        proceeds = qty * sell_price
        cash += proceeds
        trades_executed += 1

        cost_basis = float(cost_map.get(item_id, sell_price))
        loss_exit = sell_price < cost_basis
        if loss_exit:
            cooldown_days = 5 if reason == "volume_collapse" else 3
            cooldowns[item_id] = day + timedelta(days=cooldown_days)

        qty_map[item_id] = 0.0
        entry_days.pop(item_id, None)

    # Recompute equity after exits at bid marks.
    holdings_value_after_exits = 0.0
    for item_id, qty in qty_map.items():
        if qty <= 0:
            continue
        sell_price = float(current_prices.get(item_id, {}).get("sell_price", 0.0))
        if sell_price > 0:
            holdings_value_after_exits += qty * sell_price
    equity_for_targeting = cash + holdings_value_after_exits
    if equity_for_targeting <= EPS:
        equity_for_targeting = equity_before

    # Delta rebalancing with band.
    for item_id in selected:
        price_info = current_prices.get(item_id)
        if not price_info:
            continue
        buy_price = float(price_info.get("buy_price", 0.0))
        sell_price = float(price_info.get("sell_price", 0.0))
        if buy_price <= 0 or sell_price <= 0:
            continue

        qty = float(qty_map.get(item_id, 0.0))
        current_value = qty * sell_price
        target_value = equity_for_targeting * float(target_weights.get(item_id, 0.0))
        current_weight = current_value / max(equity_for_targeting, EPS)
        target_weight = target_value / max(equity_for_targeting, EPS)
        delta_weight = target_weight - current_weight

        if abs(delta_weight) <= rebalance_band:
            continue

        if delta_weight > 0:
            desired_buy_value = target_value - current_value
            buy_value = min(desired_buy_value, cash)
            if buy_value <= EPS:
                continue
            buy_qty = buy_value / buy_price
            old_qty = qty
            old_cost = float(cost_map.get(item_id, buy_price))
            new_qty = old_qty + buy_qty
            new_cost = ((old_qty * old_cost) + (buy_qty * buy_price)) / max(new_qty, EPS)
            qty_map[item_id] = new_qty
            cost_map[item_id] = new_cost
            cash -= buy_value
            trades_executed += 1
        else:
            desired_sell_value = current_value - target_value
            sell_value = min(desired_sell_value, current_value)
            if sell_value <= EPS:
                continue
            sell_qty = min(qty, sell_value / sell_price)
            if sell_qty <= EPS:
                continue
            qty_map[item_id] = max(0.0, qty - sell_qty)
            cash += sell_qty * sell_price
            trades_executed += 1
            if qty_map[item_id] <= EPS:
                qty_map[item_id] = 0.0
                entry_days.pop(item_id, None)

    # Update entry day for new positions.
    for item_id, qty in qty_map.items():
        if qty <= EPS:
            continue
        if initial_qty_map.get(item_id, 0.0) <= EPS:
            entry_days[item_id] = day

    # Build holdings rows and final equity.
    holdings_rows: list[dict[str, float | date | str]] = []
    holdings_value = 0.0
    for item_id, qty in qty_map.items():
        if qty <= EPS:
            continue
        sell_price = float(current_prices.get(item_id, {}).get("sell_price", 0.0))
        if sell_price <= 0:
            continue
        market_value = qty * sell_price
        holdings_value += market_value
        holdings_rows.append(
            {
                "day": day,
                "item_id": item_id,
                "qty": qty,
                "cost_basis": float(cost_map.get(item_id, sell_price)),
                "market_value": market_value,
            }
        )

    equity_value = cash + holdings_value
    cumulative_return_pct = (equity_value / max(start_equity, EPS) - 1.0) * 100.0
    daily_return_pct = (
        0.0
        if previous_equity is None
        else (equity_value / max(prev_equity_value, EPS) - 1.0) * 100.0
    )

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

    # Clean expired cooldowns.
    cooldowns = {k: v for k, v in cooldowns.items() if day <= v}
    new_state = {
        "entry_days": {k: v.isoformat() for k, v in entry_days.items()},
        "cooldowns": {k: v.isoformat() for k, v in cooldowns.items()},
    }
    diagnostics = {
        "selected_count": selected_count,
        "invest_cap": invest_cap,
        "trades_executed": trades_executed,
        "rebalance_band": rebalance_band,
        "cash_weight": float(cash / max(equity_value, EPS)),
        "market_regime": market_regime,
    }
    return equity_row, holdings_rows, new_state, diagnostics
