from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

EPS = 1e-9
HORIZONS = [1, 3, 7, 14, 30]

FEATURE_COLUMNS = [
    "spread_pct",
    "liquidity_score",
    "imbalance",
    "volatility_7d",
    "volatility_30d",
    "return_1d",
    "return_3d",
    "return_7d",
    "return_14d",
    "return_30d",
    "volume_daily",
    "volume_zscore_30d",
    "volume_slope_30d",
    "max_alloc_pct_feasible",
]

# Cold-start expected_return_7d heuristic:
# + 0.60 * return_7d
# + 0.25 * return_3d
# + 0.20 * imbalance
# - 0.30 * spread_pct
# - 0.15 * volatility_30d
HEURISTIC_WEIGHTS = {
    "w1_return_7d": 0.60,
    "w2_return_3d": 0.25,
    "w3_imbalance": 0.20,
    "w4_spread": 0.30,
    "w5_volatility_30d": 0.15,
}


@dataclass
class ModelBundle:
    models: dict[int, GradientBoostingRegressor]
    model_version: str
    mode: str


def _build_training_set(features: pd.DataFrame, horizon_days: int, as_of_day: date) -> pd.DataFrame:
    df = features.copy()
    df = df.sort_values(["item_id", "day"]).reset_index(drop=True)
    grouped = df.groupby("item_id", group_keys=False)

    df["future_sell_price"] = grouped["sell_price"].shift(-horizon_days)
    df["target"] = (df["future_sell_price"] - df["buy_price"]) / df["buy_price"].clip(lower=EPS)

    train = df[(df["day"] < as_of_day) & df["target"].notna()].copy()
    train = train.dropna(subset=["buy_price", "sell_price"])
    return train


def _heuristic_expected_return(frame: pd.DataFrame, horizon_days: int) -> np.ndarray:
    w = HEURISTIC_WEIGHTS
    base_7d = (
        w["w1_return_7d"] * frame["return_7d"].fillna(0.0).to_numpy()
        + w["w2_return_3d"] * frame["return_3d"].fillna(0.0).to_numpy()
        + w["w3_imbalance"] * frame["imbalance"].fillna(0.0).to_numpy()
        - w["w4_spread"] * frame["spread_pct"].fillna(0.0).to_numpy()
        - w["w5_volatility_30d"] * frame["volatility_30d"].fillna(0.0).to_numpy()
    )
    horizon_scale = {
        1: 0.35,
        3: 0.70,
        7: 1.00,
        14: 1.40,
        30: 2.00,
    }.get(horizon_days, 1.00)
    return base_7d * horizon_scale


def train_models(features: pd.DataFrame, as_of_day: date, min_history_days: int = 60) -> ModelBundle:
    unique_days = pd.Series(features["day"]).nunique()
    if unique_days < min_history_days:
        return ModelBundle(models={}, model_version=f"{as_of_day.isoformat()}-heuristic-v1", mode="heuristic")

    models: dict[int, GradientBoostingRegressor] = {}
    for horizon in HORIZONS:
        train = _build_training_set(features, horizon_days=horizon, as_of_day=as_of_day)
        train = train.dropna(subset=FEATURE_COLUMNS + ["target"])

        if len(train) < 400:
            continue

        model = GradientBoostingRegressor(
            random_state=42,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
        )
        model.fit(train[FEATURE_COLUMNS], train["target"])
        models[horizon] = model

    if not models:
        return ModelBundle(models={}, model_version=f"{as_of_day.isoformat()}-heuristic-v1", mode="heuristic")

    return ModelBundle(models=models, model_version=f"{as_of_day.isoformat()}-gbr-v1", mode="gbr")


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
        else:
            expected_return = model.predict(x)

        vol_30d = (
            frame["volatility_30d"]
            .fillna(frame["volatility_7d"])
            .fillna(0.05)
            .abs()
            .to_numpy()
        )
        spread = frame["spread_pct"].fillna(0.0).to_numpy()
        liquidity = frame["liquidity_score"].fillna(0.0).to_numpy()

        confidence = _sigmoid(expected_return / (vol_30d + EPS))
        confidence *= np.clip(1.0 - (spread / max(spread_max, EPS)), 0.2, 1.0)
        confidence *= np.clip(liquidity / max(liquidity_target, EPS), 0.2, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)

        out = pd.DataFrame(
            {
                "item_id": frame["item_id"],
                "horizon_days": horizon,
                "expected_return": expected_return,
                "confidence": confidence,
                "liquidity_score": frame["liquidity_score"].fillna(0.0).to_numpy(),
                "spread_pct": frame["spread_pct"].fillna(0.0).to_numpy(),
                "imbalance": frame["imbalance"].fillna(0.0).to_numpy(),
                "volatility_30d": frame["volatility_30d"].fillna(0.0).to_numpy(),
                "max_alloc_pct_feasible": frame["max_alloc_pct_feasible"].fillna(0.0).to_numpy(),
                "model_version": model_bundle.model_version,
            }
        )
        out_frames.append(out)

    return pd.concat(out_frames, ignore_index=True)
