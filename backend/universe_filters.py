"""
Universe filters (request #4): only surface stocks that are

  1. priced under MAX_STOCK_PRICE_INR (default 2000) - retail-affordable,
  2. listed at least MIN_LISTING_YEARS (default 5) - filters out newly
     listed/IPO stocks with no track record,
  3. not flagged for ongoing litigation/regulatory action.

Honesty note on (3): there is no free, comprehensive, machine-readable feed
of "which NSE companies currently have open legal cases" - SEBI orders,
NCLT/NCLAT proceedings, and court cases are scattered across many sources
and would need a paid legal-data provider (e.g. Watchout Investors, Screener
Pro, or a law-database API) for real coverage. What this module does
instead, and is upfront about the limits of:
  - maintains a small, manually-curated exclusion list you control
    (KNOWN_LITIGATION_SYMBOLS below - add to it yourself as you learn of cases)
  - treats a stock as "possibly flagged" if the Sentiment Agent's own news
    scan turned up litigation/regulatory keywords in recent headlines
    (fraud, SEBI probe, CBI, ED raid, insolvency, NCLT, penalty, scam)
This is a best-effort screen, not a legal clearance - always verify
independently (e.g. on the exchange's own disclosures) before trading.
"""
from config import settings

# You control this list. Add NSE symbols here as you become aware of
# material ongoing litigation/regulatory action you want auto-excluded.
KNOWN_LITIGATION_SYMBOLS: set[str] = set()

LITIGATION_KEYWORDS = (
    "fraud", "sebi probe", "sebi penalty", "cbi", "ed raid", "enforcement directorate",
    "insolvency", "nclt", "nclat", "scam", "money laundering", "chargesheet",
    "court case", "lawsuit", "litigation", "regulatory action", "ban", "debarred",
)


def is_litigation_flagged(symbol: str, flagged_risk_events: list[str]) -> bool:
    if symbol in KNOWN_LITIGATION_SYMBOLS:
        return True
    text = " ".join(flagged_risk_events).lower()
    return any(kw in text for kw in LITIGATION_KEYWORDS)


def listing_age_years(history_len_trading_days: int) -> float:
    """~252 trading days/year. yfinance 'max' period history length is our
    best free proxy for how long a symbol has traded - if we can pull N
    years of daily bars, it's been listed at least N years."""
    return round(history_len_trading_days / 252, 1)


def passes_universe_filters(last_price: float, history_len_trading_days: int,
                             flagged_risk_events: list[str], symbol: str) -> tuple[bool, str]:
    """Returns (passes, reason_if_excluded)."""
    if last_price is None:
        return False, "no price data"
    if last_price > settings.MAX_STOCK_PRICE_INR:
        return False, f"price {last_price} exceeds {settings.MAX_STOCK_PRICE_INR} cap"

    age = listing_age_years(history_len_trading_days)
    if age < settings.MIN_LISTING_YEARS:
        return False, f"only ~{age}y of history (<{settings.MIN_LISTING_YEARS}y required)"

    if settings.EXCLUDE_LITIGATION_FLAGGED and is_litigation_flagged(symbol, flagged_risk_events):
        return False, "litigation/regulatory keyword flagged in recent news"

    return True, ""
