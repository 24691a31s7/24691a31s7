"""
Explanation Agent: turns the structured outputs of every upstream agent
into a short, human-readable reason list (matches the architecture
diagram's "Recommended BUY because: ..." format).
"""


def explain(
    recommendation: str,
    technical: dict,
    fundamental: dict,
    sentiment: dict,
    risk: dict,
    prediction: dict,
    pattern: dict = None,
) -> list[str]:
    reasons = []

    if pattern and pattern.get("patterns"):
        top = pattern["patterns"][0]
        reasons.append(
            f"{top['name']} pattern detected ({top['direction']}, "
            f"~{top['reliability_pct']}% historical reliability - not a guarantee)"
        )

    tdet = technical.get("details", {})
    if tdet.get("trend") == "uptrend":
        reasons.append(f"Price is in an uptrend (EMA20 {tdet.get('ema20')} > EMA50 {tdet.get('ema50')})")
    elif tdet.get("trend") == "downtrend":
        reasons.append(f"Price is in a downtrend (EMA20 {tdet.get('ema20')} < EMA50 {tdet.get('ema50')})")
    rsi = tdet.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            reasons.append(f"RSI at {rsi} indicates oversold conditions")
        elif rsi > 70:
            reasons.append(f"RSI at {rsi} indicates overbought conditions")

    fdet = fundamental.get("details", {})
    if fdet.get("roe_pct") and fdet["roe_pct"] > 18:
        reasons.append(f"Strong fundamentals - ROE {fdet['roe_pct']}%")
    elif fdet.get("pe_ratio") and fdet["pe_ratio"] > 40:
        reasons.append(f"Valuation looks stretched - P/E {fdet['pe_ratio']}")

    if sentiment.get("score", 0) > 0.2:
        reasons.append("Positive news sentiment")
    elif sentiment.get("score", 0) < -0.2:
        reasons.append("Negative news sentiment")
    flagged = sentiment.get("details", {}).get("flagged_risk_events", [])
    if flagged:
        reasons.append(f"Flagged risk headline(s): {flagged[0]}")

    reasons.append(f"Risk profile: {risk.get('risk_label', 'Unknown')}")
    reasons.append(
        f"Expected return {prediction.get('expected_return_pct')}% "
        f"(profit probability {prediction.get('profit_probability_pct')}%)"
    )

    if not reasons:
        reasons.append("Insufficient signal strength for a strong conviction call")

    return reasons


def headline(recommendation: str, symbol: str) -> str:
    verbs = {
        "BUY": "Recommended BUY",
        "SELL": "Recommended SELL",
        "HOLD": "Recommended HOLD",
        "WATCHLIST": "Added to WATCHLIST",
    }
    return f"{verbs.get(recommendation, recommendation)} for {symbol} because:"
