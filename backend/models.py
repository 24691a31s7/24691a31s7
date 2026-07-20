from typing import Optional

from pydantic import BaseModel


class StockAnalysis(BaseModel):
    symbol: str
    name: str
    sector: str
    last_price: Optional[float] = None
    day_change_pct: Optional[float] = None
    expected_return_pct: float
    range_low_pct: float
    range_high_pct: float
    profit_probability_pct: float
    loss_probability_pct: float
    risk_label: str
    confidence_pct: float
    recommendation: str
    reasons: list[str]
    suggested_quantity: int
    estimated_investment_inr: float
    stop_loss_price: float
    target_price: float
    scanned_at: Optional[str] = None


class AlertCreate(BaseModel):
    symbol: str
    alert_type: str  # "BUY" or "SELL"
    target_price: float


class PositionSizeRequest(BaseModel):
    symbol: str
    capital: float
    risk_per_trade_pct: float


class PortfolioRequest(BaseModel):
    symbols: list[str]
    capital: float
    max_positions: Optional[int] = 8
    risk_per_trade_pct: Optional[float] = None
