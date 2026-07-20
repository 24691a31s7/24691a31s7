"""
Thin wrapper around the official `growwapi` Python SDK.

Docs used to build this (verified July 2026):
  https://groww.in/trade-api/docs/python-sdk
  https://groww.in/trade-api/docs/python-sdk/live-data

Auth: Groww supports two flows -
  1. API Key + Secret  -> requires re-approval once every 24h on the
     Groww Cloud API Keys page.
  2. API Key + TOTP secret -> no expiry, generates a fresh TOTP each call.
This client defaults to the TOTP flow (GROWW_AUTH_MODE=totp) since it's
better suited to an unattended 9:15 AM scheduled job.

If no Groww credentials are configured, every method transparently falls
back to `yfinance` (delayed / free data) so the rest of the app keeps
working in a "demo mode" - this makes local development possible without
paying for the Groww Trading API subscription.

RATE-LIMIT FIX (this file previously caused the "Groww auth failed: rate
limit exceeded" spam on every single stock): once `_groww_client` is None
after a failed auth attempt, the OLD code would try to re-authenticate on
every subsequent call - so a single rate-limit response cascaded into
dozens of repeat auth attempts within the same scan, which kept the
account rate-limited indefinitely. Now a failed auth starts a cooldown
(GROWW_AUTH_COOLDOWN_SECONDS) during which every call skips Groww entirely
and goes straight to the yfinance fallback - one log line, not fifty.
"""
import threading
import time
from typing import Iterable

from config import settings
from utils.logger import get_logger
from utils.rate_limit import groww_limiter

logger = get_logger("stocks.groww")

_groww_client = None
_groww_available = False
_auth_cooldown_until = 0.0
_auth_lock = threading.Lock()
_no_key_logged = False


def _init_groww():
    global _groww_client, _groww_available, _auth_cooldown_until, _no_key_logged

    if _groww_client is not None:
        return _groww_client

    if not settings.GROWW_API_KEY:
        if not _no_key_logged:
            logger.warning("No GROWW_API_KEY configured - running in yfinance fallback mode.")
            _no_key_logged = True  # log this once, not on every call (same fix as the auth-cooldown below)
        return None

    now = time.time()
    if now < _auth_cooldown_until:
        return None  # still cooling down from a rate-limited auth attempt - stay silent, use yfinance

    with _auth_lock:
        # Re-check inside the lock: another thread may have just succeeded
        # (or just started a new cooldown) while we were waiting.
        if _groww_client is not None or time.time() < _auth_cooldown_until:
            return _groww_client

        try:
            from growwapi import GrowwAPI

            if settings.GROWW_AUTH_MODE == "totp":
                import pyotp

                totp = pyotp.TOTP(settings.GROWW_TOTP_SECRET).now()
                access_token = GrowwAPI.get_access_token(api_key=settings.GROWW_API_KEY, totp=totp)
            else:
                access_token = GrowwAPI.get_access_token(
                    api_key=settings.GROWW_API_KEY, secret=settings.GROWW_API_SECRET
                )

            _groww_client = GrowwAPI(access_token)
            _groww_available = True
            logger.info("Groww API session established.")
        except Exception as exc:  # noqa: BLE001
            is_rate_limit = "rate limit" in str(exc).lower()
            _auth_cooldown_until = time.time() + settings.GROWW_AUTH_COOLDOWN_SECONDS
            logger.error(
                "Groww auth failed (%s) - falling back to yfinance for the next %ds%s",
                exc, settings.GROWW_AUTH_COOLDOWN_SECONDS,
                " (rate limited)" if is_rate_limit else "",
            )
            _groww_client = None

    return _groww_client


def is_live() -> bool:
    """True if we are actually talking to Groww, False if in yfinance demo mode."""
    _init_groww()
    return _groww_available


def get_ltp(symbols: Iterable[str]) -> dict:
    """
    Last traded price for up to 50 NSE cash symbols.
    Returns {symbol: price}. Falls back to yfinance if Groww isn't configured.
    """
    symbols = list(symbols)
    client = _init_groww()

    if client is not None:
        try:
            with groww_limiter:
                exch_symbols = tuple(f"NSE_{s}" for s in symbols)
                raw = client.get_ltp(segment=client.SEGMENT_CASH, exchange_trading_symbols=exch_symbols)
            return {k.replace("NSE_", ""): v for k, v in raw.items()}
        except Exception as exc:  # noqa: BLE001
            logger.error("Groww get_ltp failed (%s), falling back to yfinance", exc)

    return _yfinance_ltp(symbols)


def get_quote(symbol: str) -> dict:
    """Full quote (OHLC, depth, day change, 52w range, etc.) for one symbol."""
    client = _init_groww()
    if client is not None:
        try:
            with groww_limiter:
                return client.get_quote(
                    exchange=client.EXCHANGE_NSE, segment=client.SEGMENT_CASH, trading_symbol=symbol
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Groww get_quote failed for %s (%s), falling back", symbol, exc)

    return _yfinance_quote(symbol)


def get_ohlc(symbols: Iterable[str]) -> dict:
    symbols = list(symbols)
    client = _init_groww()
    if client is not None:
        try:
            with groww_limiter:
                exch_symbols = tuple(f"NSE_{s}" for s in symbols)
                raw = client.get_ohlc(segment=client.SEGMENT_CASH, exchange_trading_symbols=exch_symbols)
            return {k.replace("NSE_", ""): v for k, v in raw.items()}
        except Exception as exc:  # noqa: BLE001
            logger.error("Groww get_ohlc failed (%s), falling back", exc)

    return {s: _yfinance_quote(s).get("ohlc", {}) for s in symbols}


# ---------------------------------------------------------------------------
# yfinance fallback (used automatically when Groww isn't configured, and any
# time a Groww call errors out, so the demo/dev experience never breaks).
# Goes through the shared, larger-pooled HTTP session and a per-provider
# concurrency limiter - see utils/http_session.py and utils/rate_limit.py
# for why (fixes "Connection pool is full" / repeated 429s).
# ---------------------------------------------------------------------------
def _yfinance_ltp(symbols: list[str]) -> dict:
    import yfinance as yf

    from stock_universe import yf_symbol
    from utils.http_session import SHARED_SESSION, with_retry
    from utils.rate_limit import yfinance_limiter

    out = {}
    for s in symbols:
        try:
            with yfinance_limiter:
                def _fetch():
                    t = yf.Ticker(yf_symbol(s), session=SHARED_SESSION)
                    return t.fast_info.get("lastPrice") or t.fast_info.get("last_price")

                price = with_retry(_fetch, what=f"yfinance LTP {s}")
            out[s] = round(float(price), 2) if price else None
        except Exception:  # noqa: BLE001
            out[s] = None
    return out


_EMPTY_QUOTE = {
    "last_price": None, "day_change": 0, "day_change_perc": 0,
    "ohlc": {"open": None, "high": None, "low": None, "close": None},
    "week_52_high": None, "week_52_low": None, "volume": 0, "market_cap": None,
    "_source": "unavailable",
}


def _yfinance_quote(symbol: str) -> dict:
    """Best-effort quote from yfinance. Wrapped end-to-end: yfinance's
    internal scraper can raise KeyError/TypeError (not just network
    exceptions) when Yahoo changes its response shape or blocks the
    request, so a narrow try/except around individual fields isn't enough
    - any failure here degrades to an empty-but-valid quote instead of
    crashing the whole analysis pipeline."""
    from utils.http_session import SHARED_SESSION, with_retry
    from utils.rate_limit import yfinance_limiter

    try:
        import yfinance as yf

        from stock_universe import yf_symbol

        def _fetch():
            t = yf.Ticker(yf_symbol(symbol), session=SHARED_SESSION)
            info = t.fast_info
            hist = t.history(period="5d")
            return info, hist

        with yfinance_limiter:
            info, hist = with_retry(_fetch, what=f"yfinance quote {symbol}")

        last_close = float(hist["Close"].iloc[-1]) if not hist.empty else info.get("lastPrice")
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close
        day_change = (last_close - prev_close) if last_close and prev_close else 0
        return {
            "last_price": round(last_close, 2) if last_close else None,
            "day_change": round(day_change, 2),
            "day_change_perc": round((day_change / prev_close) * 100, 2) if prev_close else 0,
            "ohlc": {
                "open": round(float(info.get("open", 0) or 0), 2),
                "high": round(float(info.get("dayHigh", 0) or 0), 2),
                "low": round(float(info.get("dayLow", 0) or 0), 2),
                "close": round(prev_close, 2) if prev_close else None,
            },
            "week_52_high": round(float(info.get("yearHigh", 0) or 0), 2),
            "week_52_low": round(float(info.get("yearLow", 0) or 0), 2),
            "volume": int(info.get("lastVolume", 0) or 0),
            "market_cap": info.get("marketCap"),
            "_source": "yfinance_fallback",
        }
    except Exception as exc:  # noqa: BLE001
        # Downgraded to warning + concise message: with the fixes above
        # this should now be rare, and when it does happen it no longer
        # needs to be screamed about at ERROR level per stock per scan.
        logger.warning("yfinance quote unavailable for %s (%s) - using empty quote.", symbol, exc)
        return dict(_EMPTY_QUOTE)
