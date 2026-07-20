"""
Market-hours awareness (requests #2 and #6).

NSE cash market trades Mon-Fri, 09:15-15:30 IST (with occasional special
holidays this doesn't track - plug in an NSE holiday calendar file if you
need holiday-accuracy; as-is this is "weekday + time window", which covers
the vast majority of cases).
"""
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from config import settings

IST = ZoneInfo(settings.MARKET_TIMEZONE)


def now_ist() -> datetime:
    return datetime.now(IST)


def market_session(dt: datetime = None) -> str:
    """Returns 'pre_market', 'open', or 'post_market'."""
    dt = dt or now_ist()
    if dt.weekday() >= 5:  # Sat/Sun
        return "post_market"
    t = dt.time()
    open_t = dtime(settings.MARKET_OPEN_HOUR, settings.MARKET_OPEN_MINUTE)
    close_t = dtime(settings.MARKET_CLOSE_HOUR, settings.MARKET_CLOSE_MINUTE)
    if t < open_t:
        return "pre_market"
    if t > close_t:
        return "post_market"
    return "open"


def is_market_open(dt: datetime = None) -> bool:
    return market_session(dt) == "open"
