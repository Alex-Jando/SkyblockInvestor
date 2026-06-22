from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

try:
    from market_math import BAZAAR_TAX, EPS, net_exit_return
except ModuleNotFoundError:
    from .market_math import BAZAAR_TAX, EPS, net_exit_return

HORIZONS = [3, 7, 14]

FEATURE_COLUMNS = [
    "spread_pct",
    "rt_cost",
    "liquidity_score",
    "imbalance",
    "volatility_7d",
    "volatility_30d",
    "return_3d",
    "return_7d",
    "return_14d",
    "volume_daily",
    "volume_zscore_30d",
    "volume_slope_30d",
    "volume_collapse_frac",
    "max_alloc_pct_feasible",
]


@dataclass
class ModelBundle:
    models: dict[int, GradientBoostingRegressor]
    residual_std: dict[int, float]
    model_version: str
    mode: str


def _build_training_set(features: pd.DataFrame, horizon_days: int, as_of_day: date) -> pd.DataFrame:
    df = features.copy()
    df = df.sort_values(["item_id", "day"]).reset_index(drop=True)
    grouped = df.groupby("item_id", group_keys=False)

    # Realistic long-only target: enter by paying today's ask (sell_price) and
    # exit by selling into the future bid (buy_price), after Bazaar tax. The old
    # target used future ask / current bid, which measured the spread backwards
    # and materially overstated expected profit.
    df["future_bid_price"] = grouped["buy_price"].shift(-horizon_days)
    df["target"] = [
        net_exit_return(entry_ask, exit_bid, tax=BAZAAR_TAX)
        for entry_ask, exit_bid in zip(df["sell_price"], df["future_bid_price"], strict=False)
    ]

    train = df[(df["day"] < as_of_day) & df["target"].notna()].copy()
    return train.dropna(subset=["buy_price", "sell_price"])


def _heuristic_expected_return(frame: pd.DataFrame, horizon_days: int) -> np.ndarray:
    base_7d = (
        0.60 * frame["return_7d"].fillna(0.0).to_numpy()
        + 0.25 * frame["return_3d"].fillna(0.0).to_numpy()
        + 0.20 * frame["imbalance"].fillna(0.0).to_numpy()
        - 0.30 * frame["spread_pct"].fillna(0.0).to_numpy()
        - 0.15 * frame["volatility_30d"].fillna(0.05).to_numpy()
    )
    horizon_scale = {3: 0.70, 7: 1.00, 14: 1.40}.get(horizon_days, 1.0)
    return base_7d * horizon_scale


def train_models(features: pd.DataFrame, as_of_day: date, min_history_days: int = 60) -> ModelBundle:
    unique_days = pd.Series(features["day"]).nunique()
    if unique_days < min_history_days:
        return ModelBundle(
            models={},
            residual_std={h: 0.05 for h in HORIZONS},
            model_version=f"{as_of_day.isoformat()}-heuristic-v2",
            mode="heuristic",
        )

    models: dict[int, GradientBoostingRegressor] = {}
    residual_std: dict[int, float] = {}
    for horizon in HORIZONS:
        train = _build_training_set(features, horizon_days=horizon, as_of_day=as_of_day)
        train = train.dropna(subset=FEATURE_COLUMNS + ["target"])
        if len(train) < 300:
            continue

        model = GradientBoostingRegressor(
            random_state=42,
            n_estimators=260,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
        )
        model.fit(train[FEATURE_COLUMNS], train["target"])
        preds = model.predict(train[FEATURE_COLUMNS])
        residual_std[horizon] = float(np.std(train["target"].to_numpy() - preds))
        models[horizon] = model

    if not models:
        return ModelBundle(
            models={},
            residual_std={h: 0.05 for h in HORIZONS},
            model_version=f"{as_of_day.isoformat()}-heuristic-v2",
            mode="heuristic",
        )

    for horizon in HORIZONS:
        residual_std.setdefault(horizon, 0.06)
    return ModelBundle(
        models=models,
        residual_std=residual_std,
        model_version=f"{as_of_day.isoformat()}-gbr-v2",
        mode="gbr",
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def predict_horizon_signals(
    current_features: pd.DataFrame,
    model_bundle: ModelBundle,
    spread_max: float,
    liquidity_min: float,
    liquidity_target: float,
) -> pd.DataFrame:
    if current_features.empty:
        return pd.DataFrame()

    frame = current_features.copy().reset_index(drop=True)
    x = frame[FEATURE_COLUMNS].fillna(0.0)
    out_frames: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        model = model_bundle.models.get(horizon)
        if model is None:
            expected_return = _heuristic_expected_return(frame, horizon_days=horizon)
            model_uncertainty = np.full(len(frame), model_bundle.residual_std.get(horizon, 0.06))
        else:
            expected_return = model.predict(x)
            model_uncertainty = np.full(len(frame), model_bundle.residual_std.get(horizon, 0.06))

        vol_30d = (
            frame["volatility_30d"]
            .fillna(frame["volatility_7d"])
            .fillna(0.05)
            .abs()
            .to_numpy()
        )
        spread = frame["spread_pct"].fillna(0.0).clip(lower=0.0).to_numpy()
        liquidity = frame["liquidity_score"].fillna(0.0).to_numpy()
        imbalance = frame["imbalance"].fillna(0.0).clip(-1.0, 1.0).to_numpy()
        volume_zscore = frame["volume_zscore_30d"].fillna(0.0).to_numpy()

        base_signal = expected_return / (vol_30d + model_uncertainty + EPS)
        confidence = _sigmoid(base_signal)
        confidence *= np.clip(1.0 / (1.0 + 3.0 * model_uncertainty), 0.50, 1.00)
        confidence = np.clip(confidence, 0.0, 1.0)

        signal_quality = (
            0.45 * np.clip(expected_return, -1.0, 1.0)
            + 0.20 * np.clip(imbalance, -1.0, 1.0)
            + 0.15 * np.clip(volume_zscore / 3.0, -1.0, 1.0)
            + 0.10 * np.clip(liquidity / max(liquidity_target, EPS), 0.0, 1.5)
            - 0.10 * np.clip(spread / max(spread_max, EPS), 0.0, 2.0)
        )

        out_frames.append(
            pd.DataFrame(
                {
                    "item_id": frame["item_id"],
                    "horizon_days": horizon,
                    "expected_return": expected_return,
                    "confidence": confidence,
                    "liquidity_score": frame["liquidity_score"].fillna(0.0).to_numpy(),
                    "spread_pct": frame["spread_pct"].fillna(0.0).clip(lower=0.0).to_numpy(),
                    "imbalance": imbalance,
                    "volatility_30d": frame["volatility_30d"].fillna(0.05).to_numpy(),
                    "max_alloc_pct_feasible": frame["max_alloc_pct_feasible"].fillna(0.0).to_numpy(),
                    "signal_quality": signal_quality,
                    "model_version": model_bundle.model_version,
                }
            )
        )

    return pd.concat(out_frames, ignore_index=True)
