"""
Fundamental Analysis (Analysis Layer): P/E, P/B, ROE, ROA, EPS growth,
Debt/Equity, profit margin. Data source: services/data_service (yfinance
`.info`, cached 6h - swap in FMP for higher reliability if you have a key).
"""
from schemas import AgentResult
from services import data_service

PE_GOOD, PE_OK = 25, 40
ROE_GOOD, ROE_OK = 18, 12
DE_GOOD, DE_OK = 0.5, 1.5


def analyze(symbol: str) -> dict:
    info = data_service.get_fundamentals(symbol)
    if not info:
        return AgentResult(
            agent="fundamental", score=0.0, confidence=10.0,
            reason="Fundamental data unavailable", details={},
        ).to_dict()

    pe = info.get("trailingPE")
    pb = info.get("priceToBook")
    roe = (info.get("returnOnEquity") or 0) * 100
    de = info.get("debtToEquity")
    eps_growth = (info.get("earningsQuarterlyGrowth") or 0) * 100
    profit_margin = (info.get("profitMargins") or 0) * 100

    votes = []
    reasons = []
    available_metrics = 0
    total_metrics = 5

    if pe is not None:
        available_metrics += 1
        votes.append(1 if pe < PE_GOOD else (0 if pe < PE_OK else -1))
        if pe < PE_GOOD:
            reasons.append(f"Attractive valuation - P/E {pe:.1f}")
    if roe:
        available_metrics += 1
        votes.append(1 if roe > ROE_GOOD else (0 if roe > ROE_OK else -1))
        if roe > ROE_GOOD:
            reasons.append(f"Strong ROE {roe:.1f}%")
    if de is not None:
        available_metrics += 1
        de_ratio = de / 100 if de > 5 else de
        votes.append(1 if de_ratio < DE_GOOD else (0 if de_ratio < DE_OK else -1))
    if eps_growth:
        available_metrics += 1
        votes.append(1 if eps_growth > 15 else (0 if eps_growth > 0 else -1))
        if eps_growth > 15:
            reasons.append(f"EPS growth {eps_growth:.1f}%")
    if profit_margin:
        available_metrics += 1
        votes.append(1 if profit_margin > 15 else (0 if profit_margin > 5 else -1))

    score = round(sum(votes) / len(votes), 3) if votes else 0.0
    # Confidence scales with how many of the 5 metrics were actually available
    confidence = round(20 + (available_metrics / total_metrics) * 75, 1)

    return AgentResult(
        agent="fundamental",
        score=score,
        confidence=confidence,
        reason=reasons[0] if reasons else "Fundamentals broadly neutral",
        details={
            "pe_ratio": round(pe, 2) if pe else None,
            "pb_ratio": round(pb, 2) if pb else None,
            "roe_pct": round(roe, 2) if roe else None,
            "debt_to_equity": round(de, 2) if de else None,
            "eps_growth_pct": round(eps_growth, 2) if eps_growth else None,
            "profit_margin_pct": round(profit_margin, 2) if profit_margin else None,
            "market_cap": info.get("marketCap"),
            "metrics_available": f"{available_metrics}/{total_metrics}",
            "all_reasons": reasons,
        },
    ).to_dict()
