from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9
STATS_COLUMNS = [
    "expected_return_7d",
    "confidence_7d",
    "spread_pct",
    "liquidity_score",
    "volatility_30d",
    "imbalance",
    "max_alloc_pct_feasible",
    "volume_daily",
    "turnover_coins",
]


@dataclass
class BasketConfig:
    mode: str
    spread_max: float
    liquidity_min: float
    vol_max: float
    volume_drop_frac: float
    min_expected_return_buy: float
    conf_min_buy: float
    min_weight_pct: float
    max_weight_pct: float
    sell_neg_threshold: float
    sell_neg_threshold_14d: float
    min_basket_size: int


def log_df_stats(stage_name: str, df: pd.DataFrame, cols: list[str] | None = None) -> None:
    columns = cols or STATS_COLUMNS
    logging.info("[FUNNEL] stage=%s | count=%s", stage_name, len(df))
    if df.empty:
        return

    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        nan_frac = float(series.isna().mean())
        non_nan = series.dropna()
        if non_nan.empty:
            logging.info("[FUNNEL] %s | min=nan max=nan nan_frac=%.3f", col, nan_frac)
            continue
        logging.info(
            "[FUNNEL] %s | min=%.6f max=%.6f nan_frac=%.3f",
            col,
            float(non_nan.min()),
            float(non_nan.max()),
            nan_frac,
        )


def prepare_decision_frame(current_features: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    frame = current_features.copy().set_index("item_id")
    for horizon in [1, 3, 7, 14, 30]:
        frame[f"expected_return_{horizon}d"] = np.nan
        frame[f"confidence_{horizon}d"] = np.nan

    if signals.empty:
        return frame.reset_index()

    signals = signals.copy()
    signals["horizon_days"] = pd.to_numeric(signals["horizon_days"], errors="coerce").astype("Int64")

    expected = signals.pivot_table(
        index="item_id",
        columns="horizon_days",
        values="expected_return",
        aggfunc="last",
    )
    confidence = signals.pivot_table(
        index="item_id",
        columns="horizon_days",
        values="confidence",
        aggfunc="last",
    )

    for horizon in [1, 3, 7, 14, 30]:
        if horizon in expected.columns:
            frame[f"expected_return_{horizon}d"] = expected[horizon]
        if horizon in confidence.columns:
            frame[f"confidence_{horizon}d"] = confidence[horizon]

    return frame.reset_index()


def _normalize_with_caps(scores: np.ndarray, caps: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.array([])

    scores = np.maximum(scores.astype(float), 0.0)
    caps = np.maximum(caps.astype(float), 0.0)
    if caps.sum() <= EPS:
        return np.zeros_like(scores, dtype=float)

    weights = np.zeros_like(scores, dtype=float)
    active = np.ones(scores.size, dtype=bool)
    budget = 1.0

    while budget > 1e-9 and active.any():
        idx = np.where(active)[0]
        local_scores = scores[idx]
        if local_scores.sum() <= EPS:
            local_scores = np.ones_like(local_scores, dtype=float)

        proposed = budget * local_scores / (local_scores.sum() + EPS)
        room = np.maximum(caps[idx] - weights[idx], 0.0)
        over = proposed > room + 1e-12

        if not np.any(over):
            weights[idx] += proposed
            break

        hit_idx = idx[over]
        weights[hit_idx] += room[over]
        active[hit_idx] = False
        budget = max(0.0, 1.0 - weights.sum())

    total = weights.sum()
    if total > EPS:
        weights /= total
    return weights


def _score(df: pd.DataFrame) -> pd.Series:
    return (
        df["expected_return_7d"] / (df["volatility_30d"].abs() + 0.01)
        * (1.0 + 0.2 * df["imbalance"])
        * (1.0 + 0.05 * df["liquidity_score"])
    )


def _finalize_weights(
    candidates: pd.DataFrame,
    cfg: BasketConfig,
    relax_cap_floor: bool = False,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    working = candidates.copy()
    working["score"] = pd.to_numeric(working["score"], errors="coerce").fillna(0.0)
    working = working[working["score"] > 0].copy()
    if working.empty:
        return working

    for _ in range(6):
        caps = np.minimum(cfg.max_weight_pct, working["max_alloc_pct_feasible"].to_numpy())
        if relax_cap_floor:
            caps = np.maximum(caps, cfg.min_weight_pct)

        weights = _normalize_with_caps(working["score"].to_numpy(), caps)
        working["weight_pct"] = weights

        keep_mask = working["weight_pct"] >= cfg.min_weight_pct - 1e-12
        if keep_mask.all():
            break
        working = working[keep_mask].copy()
        if working.empty:
            break

    if working.empty:
        return working

    if working["weight_pct"].sum() > EPS:
        working["weight_pct"] = working["weight_pct"] / working["weight_pct"].sum()
    return working.sort_values("weight_pct", ascending=False)


def _apply_stage(
    stage_name: str,
    df: pd.DataFrame,
    mask: pd.Series,
    funnel_counts: dict[str, int],
) -> pd.DataFrame:
    next_df = df[mask].copy()
    funnel_counts[stage_name] = len(next_df)
    log_df_stats(stage_name, next_df)
    if next_df.empty:
        logging.warning("[FUNNEL] STOP REASON: zero items at stage=%s", stage_name)
    return next_df


def build_basket(
    decision_frame: pd.DataFrame,
    blacklist: set[str],
    cfg: BasketConfig,
) -> tuple[list[dict], list[dict], str, dict[str, int], dict[str, int], str, list[dict[str, Any]]]:
    if decision_frame.empty:
        return [], [], "No decision frame available.", {}, {}, "No decision frame available.", []

    df = decision_frame.copy()
    default_values = {
        "expected_return_1d": 0.0,
        "expected_return_3d": 0.0,
        "expected_return_7d": 0.0,
        "expected_return_14d": 0.0,
        "expected_return_30d": 0.0,
        "confidence_1d": 0.0,
        "confidence_3d": 0.0,
        "confidence_7d": 0.0,
        "confidence_14d": 0.0,
        "confidence_30d": 0.0,
        "spread_pct": 0.0,
        "liquidity_score": 0.0,
        "volatility_30d": 0.05,
        "imbalance": 0.0,
        "max_alloc_pct_feasible": 0.0,
        "volume_daily": 0.0,
        "turnover_coins": 0.0,
        "volume_median_30d": 0.0,
        "return_3d": 0.0,
        "return_7d": 0.0,
        "turnover_min_pass": False,
    }
    for col, value in default_values.items():
        if col not in df.columns:
            df[col] = value
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(value)

    df["item_id"] = df["item_id"].astype(str)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(default_values)
    df["volatility_30d"] = df["volatility_30d"].clip(lower=0.005)
    df["spread_pct"] = df["spread_pct"].clip(lower=0.0)

    funnel_counts: dict[str, int] = {"joined": len(df)}
    log_df_stats("joined", df)

    exclusion_counter: Counter[str] = Counter()
    filtered = _apply_stage("after_blacklist", df, ~df["item_id"].isin(blacklist), funnel_counts)
    exclusion_counter["blacklisted"] += len(df) - len(filtered)
    if filtered.empty:
        return [], [], "No items survived blacklist.", dict(exclusion_counter), funnel_counts, "All items blacklisted.", []

    prev_len = len(filtered)
    filtered = _apply_stage("after_spread", filtered, filtered["spread_pct"] <= cfg.spread_max, funnel_counts)
    exclusion_counter["spread_too_wide"] += prev_len - len(filtered)

    prev_len = len(filtered)
    filtered = _apply_stage(
        "after_liquidity",
        filtered,
        filtered["liquidity_score"] >= cfg.liquidity_min,
        funnel_counts,
    )
    exclusion_counter["low_liquidity"] += prev_len - len(filtered)

    prev_len = len(filtered)
    filtered = _apply_stage("after_volatility", filtered, filtered["volatility_30d"] <= cfg.vol_max, funnel_counts)
    exclusion_counter["high_volatility"] += prev_len - len(filtered)

    prev_len = len(filtered)
    collapse_ok = (filtered["volume_median_30d"] <= 0) | (
        filtered["volume_daily"] >= cfg.volume_drop_frac * filtered["volume_median_30d"]
    )
    filtered = _apply_stage("after_volume_collapse", filtered, collapse_ok, funnel_counts)
    exclusion_counter["volume_collapse"] += prev_len - len(filtered)

    prev_len = len(filtered)
    filtered = _apply_stage("after_turnover_floor", filtered, filtered["turnover_min_pass"] > 0, funnel_counts)
    exclusion_counter["turnover_too_low"] += prev_len - len(filtered)

    if filtered.empty:
        diagnosis = "No BUY items: risk/liquidity/spread/turnover filters removed all candidates."
        return [], [], diagnosis, dict(exclusion_counter), funnel_counts, diagnosis, []

    safe_universe = filtered.copy()
    safe_universe["score"] = _score(safe_universe)
    top_candidates = (
        safe_universe.sort_values("score", ascending=False)
        .head(5)[["item_id", "expected_return_7d", "score", "spread_pct", "liquidity_score", "max_alloc_pct_feasible"]]
        .to_dict("records")
    )

    buy_by_return = _apply_stage(
        "after_expected_return",
        safe_universe,
        safe_universe["expected_return_7d"] >= cfg.min_expected_return_buy,
        funnel_counts,
    )
    buy_primary = _apply_stage(
        "after_confidence",
        buy_by_return,
        buy_by_return["confidence_7d"] >= cfg.conf_min_buy,
        funnel_counts,
    )
    buy_primary = buy_primary[buy_primary["score"] > 0].copy()

    fallback_path = "PRIMARY"
    buy_pool = buy_primary.copy()
    if len(buy_pool) < cfg.min_basket_size:
        fallback_path = "FALLBACK_TOP_SCORE"
        buy_pool = safe_universe.sort_values("score", ascending=False).head(12).copy()

    weighted = _finalize_weights(buy_pool, cfg=cfg, relax_cap_floor=False)

    if len(weighted) < 1:
        fallback_path = "FALLBACK_STABILITY"
        stable = safe_universe.sort_values(
            ["liquidity_score", "spread_pct", "volatility_30d"],
            ascending=[False, True, True],
        ).head(12).copy()
        stable["score"] = 1.0 + 0.1 * stable["liquidity_score"]
        weighted = _finalize_weights(stable, cfg=cfg, relax_cap_floor=True)

    buy_items = [
        {
            "item_id": str(row["item_id"]),
            "weight_pct": float(row["weight_pct"]),
            "expected_return": float(row["expected_return_7d"]),
            "confidence": float(row["confidence_7d"]),
            "liquidity_score": float(row["liquidity_score"]),
            "spread_pct": float(row["spread_pct"]),
            "max_alloc_pct_feasible": float(row["max_alloc_pct_feasible"]),
        }
        for _, row in weighted.iterrows()
    ]

    sell_candidates = safe_universe[
        (safe_universe["expected_return_7d"] < cfg.sell_neg_threshold)
        | ((safe_universe["return_3d"] < 0) & (safe_universe["confidence_7d"] < 0.35))
    ].copy()

    if sell_candidates.empty:
        fallback_path = f"{fallback_path}+SELL_WATCHLIST"
        sell_candidates = safe_universe.sort_values("score", ascending=True).head(25).copy()

    sell_candidates = sell_candidates.sort_values(
        by=["expected_return_7d", "expected_return_14d", "expected_return_30d"],
        ascending=[True, True, True],
    ).head(50)
    sell_items = [
        {
            "item_id": str(row["item_id"]),
            "weight_pct": 0.0,
            "expected_return": float(row["expected_return_7d"]),
            "confidence": float(row["confidence_7d"]),
            "liquidity_score": float(row["liquidity_score"]),
            "spread_pct": float(row["spread_pct"]),
            "max_alloc_pct_feasible": float(row["max_alloc_pct_feasible"]),
        }
        for _, row in sell_candidates.iterrows()
    ]

    if not buy_items:
        diagnosis = (
            "No BUY items after expected_return/confidence/weight-cap checks. "
            "Fallback paths were exhausted."
        )
    elif funnel_counts.get("after_expected_return", 0) == 0:
        diagnosis = (
            "No BUY items after expected_return threshold. "
            "Consider lowering MIN_EXPECTED_RETURN_BUY or tuning heuristic spread penalty."
        )
    elif funnel_counts.get("after_confidence", 0) == 0:
        diagnosis = (
            "No BUY items after confidence threshold. "
            "Consider lowering CONF_MIN_BUY or increasing liquidity allowance."
        )
    else:
        diagnosis = "BUY funnel produced candidates."

    notes = (
        f"Decision horizon: 7d. Mode: {cfg.mode}. "
        f"Safe universe: {len(safe_universe)}. BUY count: {len(buy_items)}. "
        f"SELL count: {len(sell_items)}. Path: {fallback_path}."
    )

    return (
        buy_items,
        sell_items,
        notes,
        dict(exclusion_counter),
        funnel_counts,
        diagnosis,
        top_candidates,
    )
