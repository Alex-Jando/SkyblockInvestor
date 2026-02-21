from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from allocator import BasketConfig, build_basket, prepare_decision_frame
from db import (
    fetch_portfolio_state,
    fetch_previous_equity,
    fetch_previous_holdings,
    get_connection,
    load_current_prices,
    load_snapshots_history,
    replace_holdings,
    replace_item_signals,
    upsert_basket_and_items,
    upsert_bazaar_snapshots,
    upsert_equity,
)
from features import build_features
from hypixel_api import fetch_bazaar, parse_snapshot_rows
from model import train_models, predict_horizon_signals
from portfolio import simulate_daily_rebalance

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass
class Settings:
    hypixel_api_key: str
    supabase_database_url: str
    paper_start_coins: float
    spread_max: float
    liquidity_min: float
    vol_max: float
    volume_drop_frac: float
    feasibility_factor: float
    turnover_min_frac: float
    turnover_cap_factor: float
    liquidity_target: float
    min_expected_return_buy: float
    conf_min_buy: float
    min_weight_pct: float
    max_weight_pct: float
    sell_neg_threshold: float
    sell_neg_threshold_14d: float
    min_basket_size: int

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        return float(raw)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        return int(raw)

    @classmethod
    def from_env(cls) -> "Settings":
        hypixel_api_key = os.environ.get("HYPIXEL_API_KEY", "").strip()
        supabase_database_url = os.environ.get("SUPABASE_DATABASE_URL", "").strip()
        if not hypixel_api_key:
            raise ValueError("HYPIXEL_API_KEY is required.")
        if not supabase_database_url:
            raise ValueError("SUPABASE_DATABASE_URL is required.")

        return cls(
            hypixel_api_key=hypixel_api_key,
            supabase_database_url=supabase_database_url,
            paper_start_coins=cls._env_float("PAPER_START_COINS", 100_000_000.0),
            spread_max=cls._env_float("SPREAD_MAX", 0.05),
            liquidity_min=cls._env_float("LIQUIDITY_MIN", 2.0),
            vol_max=cls._env_float("VOL_MAX", 0.25),
            volume_drop_frac=cls._env_float("VOLUME_DROP_FRAC", 0.2),
            feasibility_factor=cls._env_float("FEASIBILITY_FACTOR", 0.05),
            turnover_min_frac=cls._env_float("TURNOVER_MIN_FRAC", 0.05),
            turnover_cap_factor=cls._env_float("TURNOVER_CAP_FACTOR", 0.25),
            liquidity_target=cls._env_float("LIQUIDITY_TARGET", 2.5),
            min_expected_return_buy=cls._env_float("MIN_EXPECTED_RETURN_BUY", 0.01),
            conf_min_buy=cls._env_float("CONF_MIN_BUY", 0.55),
            min_weight_pct=cls._env_float("MIN_WEIGHT_PCT", 0.05),
            max_weight_pct=cls._env_float("MAX_WEIGHT_PCT", 0.30),
            sell_neg_threshold=cls._env_float("SELL_NEG_THRESHOLD", -0.01),
            sell_neg_threshold_14d=cls._env_float("SELL_NEG_THRESHOLD_14D", -0.01),
            min_basket_size=cls._env_int("MIN_BASKET_SIZE", 6),
        )


def _load_blacklist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        return set()
    return {str(item) for item in payload}


def _select_signal_items(
    decision_frame: pd.DataFrame, buy_items: list[dict], sell_items: list[dict]
) -> set[str]:
    if decision_frame.empty:
        return set()

    top_liq = (
        decision_frame.sort_values("liquidity_score", ascending=False)
        .head(500)["item_id"]
        .astype(str)
        .tolist()
    )
    selected = set(top_liq)
    selected.update(str(item["item_id"]) for item in buy_items)
    selected.update(str(item["item_id"]) for item in sell_items)
    return selected


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    settings = Settings.from_env()
    run_ts = datetime.now(timezone.utc).replace(microsecond=0)
    run_day = run_ts.date()

    logging.info("Fetching bazaar snapshot from Hypixel API.")
    bazaar_payload = fetch_bazaar(settings.hypixel_api_key)
    snapshot_rows, skipped_items = parse_snapshot_rows(bazaar_payload, run_ts=run_ts)
    if not snapshot_rows:
        raise RuntimeError("No valid bazaar rows found in payload.")

    blacklist_path = Path(__file__).resolve().parent / "risk_blacklist.json"
    blacklist = _load_blacklist(blacklist_path)

    basket_cfg: BasketConfig | None = None
    history_days = 0
    funnel_counts: dict[str, int] = {}
    diagnosis = ""

    with get_connection(settings.supabase_database_url) as conn:
        try:
            upsert_bazaar_snapshots(conn, snapshot_rows)

            history = load_snapshots_history(conn)
            if history.empty:
                raise RuntimeError("History is empty after snapshot upsert.")

            history_days = pd.Series(history["day"]).nunique()
            if history_days < 15:
                mode = "BOOTSTRAP_15"
                effective_min_expected_return_buy = 0.003
                effective_conf_min_buy = 0.45
                effective_liquidity_min = min(settings.liquidity_min, 1.5)
                effective_vol_max = max(settings.vol_max, 0.35)
                effective_volume_drop_frac = min(settings.volume_drop_frac, 0.10)
                effective_turnover_min_frac = min(settings.turnover_min_frac, 0.02)
                effective_turnover_cap_factor = max(settings.turnover_cap_factor, 0.35)
                effective_sell_neg_threshold = max(settings.sell_neg_threshold, -0.005)
            elif history_days < 30:
                mode = "BOOTSTRAP_30"
                effective_min_expected_return_buy = min(
                    settings.min_expected_return_buy, 0.007
                )
                effective_conf_min_buy = min(settings.conf_min_buy, 0.50)
                effective_liquidity_min = min(settings.liquidity_min, 1.8)
                effective_vol_max = max(settings.vol_max, 0.30)
                effective_volume_drop_frac = min(settings.volume_drop_frac, 0.15)
                effective_turnover_min_frac = min(settings.turnover_min_frac, 0.02)
                effective_turnover_cap_factor = max(settings.turnover_cap_factor, 0.35)
                effective_sell_neg_threshold = max(settings.sell_neg_threshold, -0.005)
            else:
                mode = "NORMAL"
                effective_min_expected_return_buy = settings.min_expected_return_buy
                effective_conf_min_buy = settings.conf_min_buy
                effective_liquidity_min = settings.liquidity_min
                effective_vol_max = settings.vol_max
                effective_volume_drop_frac = settings.volume_drop_frac
                effective_turnover_min_frac = settings.turnover_min_frac
                effective_turnover_cap_factor = settings.turnover_cap_factor
                effective_sell_neg_threshold = settings.sell_neg_threshold

            basket_cfg = BasketConfig(
                mode=mode,
                spread_max=settings.spread_max,
                liquidity_min=effective_liquidity_min,
                vol_max=effective_vol_max,
                volume_drop_frac=effective_volume_drop_frac,
                min_expected_return_buy=effective_min_expected_return_buy,
                conf_min_buy=effective_conf_min_buy,
                min_weight_pct=settings.min_weight_pct,
                max_weight_pct=settings.max_weight_pct,
                sell_neg_threshold=effective_sell_neg_threshold,
                sell_neg_threshold_14d=settings.sell_neg_threshold_14d,
                min_basket_size=settings.min_basket_size,
            )
            logging.info(
                "Mode | history_days=%s mode=%s min_expected=%.4f conf_min=%.2f liquidity_min=%.2f vol_max=%.3f "
                "volume_drop_frac=%.2f turnover_min_frac=%.3f turnover_cap_factor=%.2f sell_neg_threshold=%.4f",
                history_days,
                mode,
                effective_min_expected_return_buy,
                effective_conf_min_buy,
                effective_liquidity_min,
                effective_vol_max,
                effective_volume_drop_frac,
                effective_turnover_min_frac,
                effective_turnover_cap_factor,
                effective_sell_neg_threshold,
            )

            features = build_features(
                history,
                spread_max=settings.spread_max,
                paper_equity_coins=settings.paper_start_coins,
                turnover_min_frac=effective_turnover_min_frac,
                turnover_cap_factor=effective_turnover_cap_factor,
                max_weight_pct=settings.max_weight_pct,
            )

            current = features[features["day"] == run_day].copy()
            if current.empty:
                latest_day = features["day"].max()
                current = features[features["day"] == latest_day].copy()
                run_day = latest_day

            model_bundle = train_models(
                features=features, as_of_day=run_day, min_history_days=60
            )
            signals = predict_horizon_signals(
                current_features=current,
                model_bundle=model_bundle,
                spread_max=settings.spread_max,
                liquidity_min=effective_liquidity_min,
                liquidity_target=settings.liquidity_target,
            )

            decision = prepare_decision_frame(current_features=current, signals=signals)
            (
                buy_items,
                sell_items,
                notes,
                exclusion_counts,
                funnel_counts,
                diagnosis,
                top_candidates,
            ) = build_basket(
                decision_frame=decision,
                blacklist=blacklist,
                cfg=basket_cfg,
            )
            if top_candidates:
                logging.info(
                    "Top candidates before final allocation | %s", top_candidates
                )

            signal_items = _select_signal_items(decision, buy_items, sell_items)
            if signal_items:
                signal_rows = signals[signals["item_id"].isin(signal_items)].to_dict(
                    "records"
                )
            else:
                signal_rows = signals.to_dict("records")
            replace_item_signals(conn, day=run_day, ts=run_ts, rows=signal_rows)

            upsert_basket_and_items(
                conn,
                ts=run_ts,
                day=run_day,
                decision_horizon_days=7,
                model_version=model_bundle.model_version,
                notes=notes,
                buy_items=buy_items,
                sell_items=sell_items,
            )

            current_prices = load_current_prices(conn, day=run_day)
            previous_holdings = fetch_previous_holdings(conn, day=run_day)
            previous_equity = fetch_previous_equity(conn, day=run_day)
            historical_peak, previous_max_drawdown = fetch_portfolio_state(
                conn,
                day=run_day,
                start_equity=settings.paper_start_coins,
            )

            equity_row, holdings = simulate_daily_rebalance(
                day=run_day,
                ts=run_ts,
                buy_items=buy_items,
                current_prices=current_prices,
                previous_holdings=previous_holdings,
                previous_equity=previous_equity,
                start_equity=settings.paper_start_coins,
                historical_peak_equity=historical_peak,
                previous_max_drawdown_pct=previous_max_drawdown,
            )

            upsert_equity(conn, row=equity_row)
            replace_holdings(conn, day=run_day, holdings=holdings)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    top_picks = (
        ", ".join(
            f"{row['item_id']} ({row['weight_pct'] * 100:.1f}%)"
            for row in buy_items[:5]
        )
        or "No BUY picks"
    )
    logging.info("Worker completed.")
    logging.info(
        "Summary | stored_items=%s | skipped_items=%s | signal_items=%s | buy=%s | sell=%s | model=%s | "
        "history_days=%s | mode=%s | funnel=%s",
        len(snapshot_rows),
        skipped_items,
        len(signal_rows),
        len(buy_items),
        len(sell_items),
        model_bundle.model_version,
        history_days,
        basket_cfg.mode if basket_cfg else "UNKNOWN",
        funnel_counts,
    )
    logging.info("Top picks | %s", top_picks)
    logging.info(
        "Portfolio | equity=%.2f | cumulative_return_pct=%.3f | daily_return_pct=%.3f | max_drawdown_pct=%.3f",
        float(equity_row["equity_value"]),
        float(equity_row["cumulative_return_pct"]),
        float(equity_row["daily_return_pct"]),
        float(equity_row["max_drawdown_pct"]),
    )
    if exclusion_counts:
        logging.info("Risk exclusions | %s", exclusion_counts)
    if not buy_items:
        logging.warning("No BUY diagnosis | %s", diagnosis)


if __name__ == "__main__":
    main()
