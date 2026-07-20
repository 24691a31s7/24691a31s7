"""
Technical Analysis (Analysis Layer): RSI, MACD, EMA, Bollinger Bands,
support/resistance, trend direction. Pure pandas/numpy - no paid
indicator library needed. Returns a structured AgentResult.
"""
import numpy as np
import pandas as pd

from schemas import AgentResult


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    ema12, ema26 = _ema(series, 12), _ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    return macd_line, signal_line


def _bollinger(series: pd.Series, period: int = 20, k: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + k * std, mid, mid - k * std


def analyze(df: pd.DataFrame) -> dict:
    """
    df: OHLCV dataframe (lowercase columns) from market_data_agent.get_price_history
    Returns an AgentResult dict: {agent, score, confidence, reason, details}
    """
    if df.empty or len(df) < 60:
        return AgentResult(
            agent="technical", score=0.0, confidence=15.0,
            reason="Insufficient price history for reliable technical signals",
            details={"bars_available": len(df)},
        ).to_dict()

    close = df["close"]
    rsi = _rsi(close).iloc[-1]
    macd_line, signal_line = _macd(close)
    macd_hist = (macd_line - signal_line).iloc[-1]
    ema20, ema50 = _ema(close, 20).iloc[-1], _ema(close, 50).iloc[-1]
    upper, mid, lower = _bollinger(close)
    last_price = close.iloc[-1]

    support = round(float(df["low"].tail(60).min()), 2)
    resistance = round(float(df["high"].tail(60).max()), 2)

    votes = []
    reasons = []

    if rsi < 30:
        votes.append(1); reasons.append(f"RSI {rsi:.0f} oversold")
    elif rsi > 70:
        votes.append(-1); reasons.append(f"RSI {rsi:.0f} overbought")
    else:
        votes.append((50 - rsi) / 50 * -0.3)

    votes.append(1 if macd_hist > 0 else -1)
    reasons.append("MACD bullish crossover" if macd_hist > 0 else "MACD bearish crossover")

    if ema20 > ema50 and last_price > ema20:
        votes.append(1); reasons.append("Price above EMA20/EMA50 uptrend")
    elif ema20 < ema50 and last_price < ema20:
        votes.append(-1); reasons.append("Price below EMA20/EMA50 downtrend")
    else:
        votes.append(0)

    band_width = (upper.iloc[-1] - lower.iloc[-1]) or 1
    bb_pos = (last_price - mid.iloc[-1]) / band_width
    votes.append(max(-1, min(1, -bb_pos)))

    score = float(np.clip(np.mean(votes), -1, 1))

    # Confidence: more history + stronger agreement across the 4 votes = higher confidence
    agreement = 1 - (np.std(votes) / 2)  # votes are roughly in [-1,1]
    data_depth_factor = min(1.0, len(df) / 250)  # full confidence once we have ~1y of bars
    confidence = round(max(20.0, min(95.0, agreement * 70 + data_depth_factor * 25)), 1)

    return AgentResult(
        agent="technical",
        score=round(score, 3),
        confidence=confidence,
        reason=reasons[0] if reasons else "Mixed technical signals",
        details={
            "rsi_14": round(float(rsi), 2),
            "macd_histogram": round(float(macd_hist), 4),
            "ema20": round(float(ema20), 2),
            "ema50": round(float(ema50), 2),
            "support": support,
            "resistance": resistance,
            "trend": "uptrend" if ema20 > ema50 else "downtrend",
            "all_reasons": reasons,
        },
    ).to_dict()
