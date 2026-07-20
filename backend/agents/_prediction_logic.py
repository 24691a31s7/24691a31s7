"""
Prediction Layer: Return Forecast Model + Profit Probability Model +
Loss Probability Model.

v1 implementation is a transparent, explainable weighted-score model built
from the technical/fundamental/sentiment scores plus recent realized
volatility - this is intentionally NOT a black-box ML model, so every
number the UI shows can be traced back to a reason (matches the
architecture's "explainable AI" goal).

To upgrade to real ML: replace `forecast_return()` with a trained
regressor (e.g. XGBoost/LightGBM per the tech stack in the architecture
diagram) that takes the same feature vector as input.
"""
import numpy as np
import pandas as pd


def _historical_volatility(df: pd.DataFrame, window: int = 20) -> float:
    if df.empty or len(df) < window + 1:
        return 0.02  # 2% default daily vol
    returns = df["close"].pct_change().dropna()
    return float(returns.tail(window).std())


def forecast_return(
    technical_score: float,
    fundamental_score: float,
    sentiment_score: float,
    price_history: pd.DataFrame,
    pattern_score: float = 0.0,
) -> dict:
    """
    Combines the analysis-layer scores into an expected next-session
    return, a confidence range, and profit/loss probabilities via a normal
    approximation seeded by recent realized volatility.

    pattern_score (optional): the Pattern Recognition Agent's -1..1 lean
    from detected chart/candlestick patterns. Weighted lowest of the four
    inputs - published pattern-reliability studies put chart patterns as
    the weakest standalone signal of the four (see pattern_recognition_agent.py).
    """
    # Weighted composite: technical signals matter most for next-session
    # moves, fundamentals for medium-term drift, sentiment as a catalyst,
    # patterns as a minor corroborating/contradicting signal.
    if pattern_score:
        composite = (0.40 * technical_score + 0.27 * fundamental_score
                     + 0.20 * sentiment_score + 0.13 * pattern_score)
    else:
        composite = 0.45 * technical_score + 0.30 * fundamental_score + 0.25 * sentiment_score
    composite = float(np.clip(composite, -1, 1))

    daily_vol = _historical_volatility(price_history)
    # Map composite score to an expected return, scaled by the stock's own volatility
    expected_return_pct = composite * daily_vol * 100 * 1.5  # 1.5x = conviction multiplier

    # Profit probability via normal CDF assuming mean = expected_return, std = daily_vol
    from math import erf, sqrt

    std_pct = daily_vol * 100
    if std_pct == 0:
        profit_prob = 50.0
    else:
        z = -expected_return_pct / (std_pct * sqrt(2))
        profit_prob = (1 - 0.5 * (1 + erf(z))) * 100  # P(return > 0)
    profit_prob = float(np.clip(profit_prob, 5, 95))
    loss_prob = round(100 - profit_prob, 2)

    range_low = round(expected_return_pct - std_pct, 2)
    range_high = round(expected_return_pct + std_pct, 2)

    return {
        "expected_return_pct": round(expected_return_pct, 2),
        "range_low_pct": range_low,
        "range_high_pct": range_high,
        "profit_probability_pct": round(profit_prob, 2),
        "loss_probability_pct": loss_prob,
        "composite_score": round(composite, 3),
    }
