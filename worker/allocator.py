from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-9


@dataclass
class BasketConfig:
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


def prepare_decision_frame(current_features: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    frame = current_features.copy().set_index("item_id")

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
        frame[f"expected_return_{horizon}d"] = expected[horizon] if horizon in expected.columns else np.nan
        frame[f"confidence_{horizon}d"] = confidence[horizon] if horizon in confidence.columns else np.nan

    frame = frame.reset_index()
    return frame


def _normalize_with_caps(scores: np.ndarray, caps: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.array([])

    scores = np.maximum(scores.astype(float), 0.0)
    caps = np.maximum(caps.astype(float), 0.0)
    weights = np.zeros_like(scores, dtype=float)

    active = np.ones(scores.size, dtype=bool)
    budget = 1.0

    while budget > 1e-9 and active.any():
        idx = np.where(active)[0]
        active_scores = scores[idx]
        score_sum = active_scores.sum()
        if score_sum <= EPS:
            break

        proposed = budget * active_scores / score_sum
        room = np.maximum(caps[idx] - weights[idx], 0.0)
        over = proposed > room + 1e-12

        if not np.any(over):
            weights[idx] += proposed
            budget = 0.0
            break

        hit_idx = idx[over]
        weights[hit_idx] += room[over]
        active[hit_idx] = False
        budget = max(0.0, 1.0 - weights.sum())

    return weights


def _score_buy_candidates(df: pd.DataFrame) -> pd.Series:
    risk_adjusted = df["expected_return_7d"] / (df["volatility_30d"].abs() + 0.01)
    demand_boost = 1.0 + 0.2 * df["imbalance"]
    liquidity_boost = 1.0 + 0.05 * df["liquidity_score"]
    return risk_adjusted * demand_boost * liquidity_boost


def _compute_reason(row: pd.Series, blacklist: set[str], cfg: BasketConfig) -> list[str]:
    reasons: list[str] = []
    item_id = str(row["item_id"])
    if item_id in blacklist:
        reasons.append("blacklisted")
    if row["spread_pct"] > cfg.spread_max:
        reasons.append("spread_too_wide")
    if row["liquidity_score"] < cfg.liquidity_min:
        reasons.append("low_liquidity")
    if row["volatility_30d"] > cfg.vol_max:
        reasons.append("high_volatility")
    if (
        pd.notna(row.get("volume_median_30d"))
        and row.get("volume_median_30d", 0.0) > 0
        and row.get("volume_daily", 0.0) < cfg.volume_drop_frac * row.get("volume_median_30d", 0.0)
    ):
        reasons.append("volume_collapse")
    if row["max_alloc_pct_feasible"] < cfg.min_weight_pct:
        reasons.append("allocation_not_feasible")
    return reasons


def _finalize_weights(candidates: pd.DataFrame, cfg: BasketConfig) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    working = candidates.copy()
    working = working[working["score"] > 0].copy()
    if working.empty:
        return working

    while not working.empty:
        scores = working["score"].to_numpy()
        caps = np.minimum(cfg.max_weight_pct, working["max_alloc_pct_feasible"].to_numpy())
        weights = _normalize_with_caps(scores, caps)

        working["weight_pct"] = weights
        keep_mask = working["weight_pct"] >= cfg.min_weight_pct - 1e-12
        if keep_mask.all():
            break
        working = working[keep_mask].copy()

    return working.sort_values("weight_pct", ascending=False)


def build_basket(
    decision_frame: pd.DataFrame,
    blacklist: set[str],
    cfg: BasketConfig,
) -> tuple[list[dict], list[dict], str, dict[str, int]]:
    if decision_frame.empty:
        return [], [], "No decision frame available.", {}

    df = decision_frame.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "spread_pct",
            "liquidity_score",
            "volatility_30d",
            "max_alloc_pct_feasible",
            "expected_return_7d",
            "confidence_7d",
            "expected_return_14d",
        ]
    ).copy()

    exclusion_counter: Counter[str] = Counter()
    pass_mask = []
    for _, row in df.iterrows():
        reasons = _compute_reason(row, blacklist, cfg)
        if reasons:
            exclusion_counter.update(reasons)
            pass_mask.append(False)
        else:
            pass_mask.append(True)
    df["passes_risk"] = pass_mask
    filtered = df[df["passes_risk"]].copy()

    buy = filtered[
        (filtered["expected_return_7d"] >= cfg.min_expected_return_buy)
        & (filtered["confidence_7d"] >= cfg.conf_min_buy)
    ].copy()
    buy["score"] = _score_buy_candidates(buy)
    buy = buy[buy["score"] > 0].copy()

    if not buy.empty and np.minimum(cfg.max_weight_pct, buy["max_alloc_pct_feasible"]).sum() < 1.0:
        supplemental = filtered[
            (~filtered["item_id"].isin(buy["item_id"]))
            & (filtered["expected_return_7d"] > -0.005)
            & (filtered["confidence_7d"] > 0.40)
        ].copy()
        supplemental["score"] = _score_buy_candidates(supplemental).clip(lower=0.0)
        supplemental = supplemental.sort_values(["score", "liquidity_score"], ascending=False)
        for _, row in supplemental.iterrows():
            buy = pd.concat([buy, row.to_frame().T], ignore_index=True)
            if np.minimum(cfg.max_weight_pct, buy["max_alloc_pct_feasible"]).sum() >= 1.0:
                break

    fallback_used = False
    if buy.empty:
        fallback = filtered[
            (filtered["expected_return_7d"] > -0.02)
            & (filtered["confidence_7d"] > 0.35)
        ].copy()
        fallback["score"] = (
            (1.0 / (fallback["spread_pct"] + 0.001))
            + 0.4 * fallback["liquidity_score"]
            + 15.0 * np.maximum(fallback["return_7d"], 0.0)
        )
        buy = fallback.sort_values("score", ascending=False).head(12).copy()
        fallback_used = True

    weighted = _finalize_weights(buy, cfg)
    if weighted.empty:
        buy_items: list[dict] = []
    else:
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

    sell = df[
        (df["expected_return_7d"] < cfg.sell_neg_threshold)
        | (df["expected_return_14d"] < cfg.sell_neg_threshold_14d)
        | ((df["confidence_7d"] < 0.35) & (df["return_7d"] < 0))
    ].copy()
    sell = sell[~sell["item_id"].isin(blacklist)].copy()
    sell = sell.sort_values(
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
        for _, row in sell.iterrows()
    ]

    note_parts = [
        "Decision horizon: 7d.",
        f"Risk-passed items: {len(filtered)}.",
        f"BUY count: {len(buy_items)}.",
        f"SELL count: {len(sell_items)}.",
    ]
    if fallback_used:
        note_parts.append("Fallback basket used due empty primary candidate set.")

    notes = " ".join(note_parts)
    return buy_items, sell_items, notes, dict(exclusion_counter)
