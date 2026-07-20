"""
Risk Layer: Volatility Risk + Event Risk + Liquidity Risk -> Risk Aggregation.
"""
import numpy as np
import pandas as pd


def _volatility_risk(df: pd.DataFrame) -> float:
    """0 (low) - 1 (high) based on annualized volatility."""
    if df.empty or len(df) < 30:
        return 0.5
    returns = df["close"].pct_change().dropna()
    ann_vol = returns.tail(60).std() * np.sqrt(252)
    # 15% ann. vol -> low risk, 60%+ -> high risk (typical NSE large-cap range)
    return float(np.clip((ann_vol - 0.15) / 0.45, 0, 1))


def _liquidity_risk(df: pd.DataFrame) -> float:
    """0 (liquid, low risk) - 1 (illiquid, high risk) based on avg daily volume trend."""
    if df.empty or "volume" not in df or len(df) < 20:
        return 0.5
    recent_vol = df["volume"].tail(20).mean()
    older_vol = df["volume"].tail(60).mean() if len(df) >= 60 else recent_vol
    if older_vol == 0:
        return 0.5
    drop = (older_vol - recent_vol) / older_vol
    return float(np.clip(drop, 0, 1))  # falling volume -> rising liquidity risk


def _event_risk(flagged_risk_events: list) -> float:
    """0 (no flagged headlines) - 1 (multiple negative/legal/fraud headlines)."""
    return float(np.clip(len(flagged_risk_events) / 3, 0, 1))


def assess(price_history: pd.DataFrame, flagged_risk_events: list) -> dict:
    vol_risk = _volatility_risk(price_history)
    liq_risk = _liquidity_risk(price_history)
    evt_risk = _event_risk(flagged_risk_events)

    aggregate = 0.5 * vol_risk + 0.25 * liq_risk + 0.25 * evt_risk
    label = "Low" if aggregate < 0.33 else ("Medium" if aggregate < 0.66 else "High")

    return {
        "volatility_risk": round(vol_risk, 3),
        "liquidity_risk": round(liq_risk, 3),
        "event_risk": round(evt_risk, 3),
        "aggregate_risk_score": round(aggregate, 3),
        "risk_label": label,
    }
