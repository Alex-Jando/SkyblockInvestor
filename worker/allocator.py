from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9
HORIZONS = [3, 7, 14]


@dataclass
class BasketConfig:
    mode: str
    market_regime: str
    spread_max: float
    liquidity_min: float
    vol_max: float
    min_weight_pct: float
    max_weight_pct: float
    sell_neg_threshold: float
    sell_neg_threshold_14d: float
    max_holdings: int = 5


def _clip(series: pd.Series, low: float, high: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=low, upper=high)


def log_df_stats(stage_name: str, df: pd.DataFrame) -> None:
    cols = [
        "expected_return_7d",
        "spread_pct",
        "liquidity_score",
        "volatility_30d",
        "confidence_7d",
        "max_alloc_pct_feasible",
        "turnover_coins",
        "score",
    ]
    logging.info("[FUNNEL] %s | count=%s", stage_name, len(df))
    if df.empty:
        return
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        non_na = series.dropna()
        if non_na.empty:
            logging.info("[FUNNEL] %s | min=nan max=nan nan_frac=1.000", col)
        else:
            logging.info(
                "[FUNNEL] %s | min=%.6f max=%.6f nan_frac=%.3f",
                col,
                float(non_na.min()),
                float(non_na.max()),
                float(series.isna().mean()),
            )


def prepare_decision_frame(current_features: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    frame = current_features.copy().set_index("item_id")
    for horizon in HORIZONS:
        frame[f"expected_return_{horizon}d"] = np.nan
        frame[f"confidence_{horizon}d"] = np.nan

    if signals.empty:
        return frame.reset_index()

    expected = signals.pivot_table(index="item_id", columns="horizon_days", values="expected_return", aggfunc="last")
    confidence = signals.pivot_table(index="item_id", columns="horizon_days", values="confidence", aggfunc="last")

    for horizon in HORIZONS:
        if horizon in expected.columns:
            frame[f"expected_return_{horizon}d"] = expected[horizon]
        if horizon in confidence.columns:
            frame[f"confidence_{horizon}d"] = confidence[horizon]
    return frame.reset_index()


def _score(df: pd.DataFrame) -> pd.Series:
    net_edge_3d = df["expected_return_3d"] - 1.2 * df["spread_pct"]
    net_edge_7d = df["expected_return_7d"] - 1.5 * df["spread_pct"]
    net_edge_14d = df["expected_return_14d"] - 1.8 * df["spread_pct"]

    quality_bonus = (
        0.10 * _clip(df["imbalance"], -1.0, 1.0)
        + 0.08 * _clip(df["volume_zscore_30d"], -3.0, 3.0)
        + 0.06 * _clip(df["volume_slope_30d"], -3.0, 3.0)
        + 0.05 * np.log1p(df["liquidity_score"].clip(lower=0.0))
    )

    risk_penalty = (
        0.25 * df["volatility_7d"].clip(lower=0.0)
        + 0.35 * df["volatility_30d"].clip(lower=0.0)
        + 0.20 * np.maximum(0.0, 1.0 - df["volume_collapse_frac"])
    )

    return 0.25 * net_edge_3d + 0.50 * net_edge_7d + 0.25 * net_edge_14d + quality_bonus - risk_penalty


def _normalize_with_caps(weights_raw: np.ndarray, caps: np.ndarray, invest_fraction: float) -> np.ndarray:
    if len(weights_raw) == 0 or invest_fraction <= 0:
        return np.array([])
    scores = np.maximum(weights_raw, 0.0)
    if scores.sum() <= EPS:
        scores = np.ones_like(scores, dtype=float)

    target = invest_fraction
    weights = target * scores / (scores.sum() + EPS)
    weights = np.minimum(weights, caps)
    remaining = target - weights.sum()
    if remaining > 1e-8:
        room = np.maximum(caps - weights, 0.0)
        if room.sum() > EPS:
            weights += remaining * room / (room.sum() + EPS)
    return weights


def _apply_mask(stage_name: str, df: pd.DataFrame, mask: pd.Series, funnel_counts: dict[str, int]) -> pd.DataFrame:
    out = df[mask].copy()
    funnel_counts[stage_name] = len(out)
    log_df_stats(stage_name, out)
    return out


def build_basket(
    decision_frame: pd.DataFrame,
    blacklist: set[str],
    cfg: BasketConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, int], dict[str, int], str, list[dict[str, Any]], pd.DataFrame]:
    if decision_frame.empty:
        return [], [], "No decision frame.", {}, {}, "No decision frame.", [], pd.DataFrame()

    df = decision_frame.copy()
    defaults = {
        "expected_return_3d": 0.0,
        "expected_return_7d": 0.0,
        "expected_return_14d": 0.0,
        "confidence_3d": 0.0,
        "confidence_7d": 0.0,
        "confidence_14d": 0.0,
        "spread_pct": 0.0,
        "liquidity_score": 0.0,
        "volatility_7d": 0.05,
        "volatility_30d": 0.05,
        "imbalance": 0.0,
        "volume_zscore_30d": 0.0,
        "volume_slope_30d": 0.0,
        "volume_collapse_frac": 1.0,
        "turnover_min_pass": 0.0,
        "max_alloc_pct_feasible": 0.0,
        "turnover_coins": 0.0,
        "return_3d": 0.0,
        "rt_cost": 0.0,
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(value)
    df["item_id"] = df["item_id"].astype(str)
    df["spread_pct"] = df["spread_pct"].clip(lower=0.0)
    df["score"] = _score(df)
    df["rank_score"] = df["score"].rank(ascending=False, method="first")

    funnel_counts: dict[str, int] = {"joined": len(df)}
    log_df_stats("joined", df)

    exclusions: Counter[str] = Counter()
    safe = _apply_mask("after_blacklist", df, ~df["item_id"].isin(blacklist), funnel_counts)
    exclusions["blacklisted"] += len(df) - len(safe)
    safe = _apply_mask("after_spread", safe, safe["spread_pct"] <= cfg.spread_max, funnel_counts)
    safe = _apply_mask("after_liquidity", safe, safe["liquidity_score"] >= cfg.liquidity_min, funnel_counts)
    safe = _apply_mask("after_volatility", safe, safe["volatility_30d"] <= cfg.vol_max, funnel_counts)
    safe = _apply_mask("after_volume_collapse", safe, safe["volume_collapse_frac"] >= 0.80, funnel_counts)
    safe = _apply_mask("after_turnover", safe, safe["turnover_min_pass"] > 0, funnel_counts)

    required_return = np.maximum(0.015, 1.5 * safe["spread_pct"])
    if cfg.market_regime == "RISK_OFF":
        required_return = np.maximum(required_return, 0.025)

    eligible = _apply_mask(
        "after_expected_return",
        safe,
        safe["expected_return_7d"] >= required_return,
        funnel_counts,
    )
    eligible = _apply_mask("after_confidence", eligible, eligible["confidence_7d"] >= 0.60, funnel_counts)
    eligible = _apply_mask(
        "after_feasible_cap",
        eligible,
        eligible["max_alloc_pct_feasible"] >= cfg.min_weight_pct,
        funnel_counts,
    )
    eligible = eligible.sort_values("score", ascending=False)

    top_candidates = eligible.head(5)[
        ["item_id", "expected_return_3d", "expected_return_7d", "expected_return_14d", "score", "spread_pct"]
    ].to_dict("records")

    invest_fraction = 0.0
    if len(eligible) == 1:
        invest_fraction = 0.20
    elif len(eligible) == 2:
        invest_fraction = 0.40
    elif len(eligible) >= 3:
        invest_fraction = 0.80
    if cfg.market_regime == "RISK_OFF":
        invest_fraction = min(invest_fraction, 0.60)

    buy_items: list[dict[str, Any]] = []
    diagnosis = "Eligible buys found."
    if len(eligible) < 3:
        diagnosis = "Fewer than 3 eligible items after filters; hold cash (no fallback basket)."
    else:
        selected = eligible.head(cfg.max_holdings).copy()
        caps = np.minimum(cfg.max_weight_pct, selected["max_alloc_pct_feasible"].to_numpy())
        if cfg.market_regime == "RISK_OFF":
            caps = np.minimum(caps, 0.22)
        weights = _normalize_with_caps(selected["score"].to_numpy(), caps, invest_fraction=invest_fraction)
        selected["weight_pct"] = weights
        selected = selected[selected["weight_pct"] >= cfg.min_weight_pct - 1e-12].copy()
        if selected["weight_pct"].sum() > EPS:
            selected["weight_pct"] = selected["weight_pct"] / selected["weight_pct"].sum() * invest_fraction

        buy_items = [
            {
                "item_id": str(row["item_id"]),
                "weight_pct": float(row["weight_pct"]),
                "expected_return": float(row["expected_return_7d"]),
                "confidence": float(row["confidence_7d"]),
                "liquidity_score": float(row["liquidity_score"]),
                "spread_pct": float(row["spread_pct"]),
                "max_alloc_pct_feasible": float(row["max_alloc_pct_feasible"]),
                "expected_return_3d": float(row["expected_return_3d"]),
                "expected_return_7d": float(row["expected_return_7d"]),
                "expected_return_14d": float(row["expected_return_14d"]),
                "score": float(row["score"]),
                "rank_score": int(row["rank_score"]),
                "volume_collapse_frac": float(row["volume_collapse_frac"]),
                "return_3d": float(row["return_3d"]),
            }
            for _, row in selected.iterrows()
        ]

    sell_candidates = safe[
        (safe["expected_return_7d"] < cfg.sell_neg_threshold)
        | ((safe["return_3d"] < 0) & (safe["confidence_7d"] < 0.35))
    ].copy()
    if sell_candidates.empty:
        sell_candidates = safe.sort_values("score", ascending=True).head(25).copy()
    sell_candidates = sell_candidates.sort_values(["expected_return_7d", "score"], ascending=[True, True]).head(50)

    sell_items = [
        {
            "item_id": str(row["item_id"]),
            "weight_pct": 0.0,
            "expected_return": float(row["expected_return_7d"]),
            "confidence": float(row["confidence_7d"]),
            "liquidity_score": float(row["liquidity_score"]),
            "spread_pct": float(row["spread_pct"]),
            "max_alloc_pct_feasible": float(row["max_alloc_pct_feasible"]),
            "expected_return_3d": float(row["expected_return_3d"]),
            "expected_return_7d": float(row["expected_return_7d"]),
            "expected_return_14d": float(row["expected_return_14d"]),
            "score": float(row["score"]),
            "rank_score": int(row["rank_score"]),
            "volume_collapse_frac": float(row["volume_collapse_frac"]),
            "return_3d": float(row["return_3d"]),
        }
        for _, row in sell_candidates.iterrows()
    ]

    notes = (
        f"Mode={cfg.mode}, regime={cfg.market_regime}, eligible={len(eligible)}, "
        f"buy={len(buy_items)}, sell={len(sell_items)}, invest_frac={invest_fraction:.2f}"
    )
    return buy_items, sell_items, notes, dict(exclusions), funnel_counts, diagnosis, top_candidates, safe
