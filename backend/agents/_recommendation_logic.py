"""
Recommendation Agent: combines every upstream signal into a final
BUY / SELL / HOLD / WATCHLIST decision, plus a position-sizing suggestion.

Position sizing uses a standard fixed-fractional risk model:
    shares = (capital * risk_per_trade%) / (entry_price * stop_loss_distance%)
capped by max allocation per stock. This is plain arithmetic parameterized
by the user's own capital/risk inputs from the UI - not a personalized
recommendation, and it does not place any order (alerts only).

IMPORTANT: this whole system is a heuristic/educational decision-support
tool, not investment advice. See README "Disclaimer" section.
"""
from config import settings


def decide(
    expected_return_pct: float,
    profit_probability_pct: float,
    risk_label: str,
    confidence_pct: float,
    conflict_detected: bool,
) -> str:
    if confidence_pct < 40 or conflict_detected:
        return "WATCHLIST"

    if expected_return_pct >= 0.8 and profit_probability_pct >= 65 and risk_label in ("Low", "Medium"):
        return "BUY"
    if expected_return_pct <= -0.8 and profit_probability_pct <= 40:
        return "SELL"
    if 0 <= expected_return_pct < 0.8 and risk_label == "Low":
        return "HOLD"
    if expected_return_pct < 0 and risk_label != "High":
        return "HOLD"
    return "WATCHLIST"


def size_position(
    entry_price: float,
    risk_label: str,
    capital: float = None,
    risk_per_trade_pct: float = None,
) -> dict:
    """
    Fixed-fractional position sizing. Stop-loss distance widens with risk
    label since higher-risk names need more room before a stop is hit.
    """
    capital = capital or settings.DEFAULT_CAPITAL_INR
    risk_per_trade_pct = risk_per_trade_pct or settings.DEFAULT_RISK_PER_TRADE_PCT

    stop_distance_pct = {"Low": 3.0, "Medium": 5.0, "High": 8.0}.get(risk_label, 5.0)
    risk_amount = capital * (risk_per_trade_pct / 100)
    stop_loss_price = round(entry_price * (1 - stop_distance_pct / 100), 2)
    per_share_risk = entry_price - stop_loss_price

    if per_share_risk <= 0:
        return {"quantity": 0, "stop_loss_price": stop_loss_price, "target_price": entry_price}

    raw_qty = int(risk_amount / per_share_risk)

    # Cap by max allocation per stock
    max_alloc = capital * (settings.MAX_ALLOCATION_PER_STOCK_PCT / 100)
    max_qty_by_alloc = int(max_alloc / entry_price) if entry_price else 0
    quantity = max(0, min(raw_qty, max_qty_by_alloc))

    target_price = round(entry_price * (1 + (stop_distance_pct * 1.5) / 100), 2)  # ~1.5:1 reward:risk

    return {
        "quantity": quantity,
        "estimated_investment_inr": round(quantity * entry_price, 2),
        "stop_loss_price": stop_loss_price,
        "target_price": target_price,
        "risk_amount_inr": round(risk_amount, 2),
    }
