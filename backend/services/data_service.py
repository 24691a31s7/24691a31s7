"""
Data Service: the ONLY module that talks to external APIs (Groww, yfinance,
NewsAPI). Every agent goes through this instead of calling those APIs
directly - one service, many agents, and every outbound call goes through
the shared HTTP session + per-provider concurrency limiter (fixes the
"Connection pool is full" / repeated 429 errors - see utils/http_session.py
and utils/rate_limit.py for the root-cause explanation).

Caching strategy per data type:
  - Live quotes/LTP   -> cache_manager, QUOTE_CACHE_TTL_SECONDS (default 5s)
  - 5Y OHLC history    -> Postgres/SQLite (PriceHistoryCache table),
                           refreshed at most once every 24h
  - Fundamentals       -> cache_manager, FUNDAMENTALS_CACHE_TTL_SECONDS (6h)
  - News articles      -> cache_manager, NEWS_CACHE_TTL_SECONDS (45 min)
"""
import json
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import groww_client
from cache.cache_manager import cache
from config import settings
from database import PriceHistoryCache, SessionLocal
from stock_universe import yf_symbol
from utils.http_session import SHARED_SESSION, with_retry
from utils.logger import get_logger
from utils.rate_limit import newsapi_limiter, yfinance_limiter

log = get_logger("stocks.data_service")

HISTORY_STALE_AFTER = timedelta(hours=24)


# ---------------------------------------------------------------------------
# Live prices / quotes (Groww, with yfinance fallback baked into groww_client)
# ---------------------------------------------------------------------------
def get_live_quote(symbol: str) -> dict:
    key = f"quote:{symbol}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    quote = groww_client.get_quote(symbol)
    cache.set(key, quote, ttl=settings.QUOTE_CACHE_TTL_SECONDS)
    return quote


def get_live_prices_bulk(symbols: list[str]) -> dict:
    prices, missing = {}, []
    for s in symbols:
        hit = cache.get(f"ltp:{s}")
        if hit is not None:
            prices[s] = hit
        else:
            missing.append(s)

    if missing:
        for i in range(0, len(missing), 50):
            batch = missing[i : i + 50]
            fetched = groww_client.get_ltp(batch)
            for sym, price in fetched.items():
                prices[sym] = price
                cache.set(f"ltp:{sym}", price, ttl=settings.QUOTE_CACHE_TTL_SECONDS)
    return prices


# ---------------------------------------------------------------------------
# Historical OHLC - DB-persisted, refreshed at most once/day. This is the
# single biggest rate-limit saver: a stock's 5-year daily history barely
# changes intraday, so once it's cached, a full day of scans re-reads it
# from SQLite/Postgres instead of re-hitting Yahoo at all.
# ---------------------------------------------------------------------------
def get_price_history(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    db = SessionLocal()
    try:
        row = (
            db.query(PriceHistoryCache)
            .filter_by(symbol=symbol, period=period, interval=interval)
            .first()
        )
        if row and (datetime.utcnow() - row.updated_at) < HISTORY_STALE_AFTER:
            try:
                records = json.loads(row.data_json)
                df = pd.DataFrame(records)
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                    return df
            except Exception as exc:  # noqa: BLE001
                log.warning("Cached history for %s unreadable (%s), refetching.", symbol, exc)

        # Stale or missing -> fetch fresh from yfinance and persist
        df = _fetch_history_fresh(symbol, period, interval)
        if not df.empty:
            payload = df.reset_index().rename(columns={"index": "date", "Date": "date"})
            payload["date"] = payload["date"].astype(str)
            data_json = payload.to_json(orient="records")
            if row:
                row.data_json = data_json
                row.updated_at = datetime.utcnow()
            else:
                row = PriceHistoryCache(
                    symbol=symbol, period=period, interval=interval,
                    data_json=data_json, updated_at=datetime.utcnow(),
                )
                db.add(row)
            db.commit()
        return df
    finally:
        db.close()


def _fetch_history_fresh(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        def _fetch():
            df = yf.Ticker(yf_symbol(symbol), session=SHARED_SESSION).history(period=period, interval=interval)
            return df.rename(columns=str.lower).dropna()

        with yfinance_limiter:
            return with_retry(_fetch, what=f"yfinance history {symbol}")
    except Exception as exc:  # noqa: BLE001
        log.warning("History fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Fundamentals - cached FUNDAMENTALS_CACHE_TTL_SECONDS (default 6h - P/E
# etc. don't move intraday, so there's no reason to ever re-fetch them
# within the same trading day).
# ---------------------------------------------------------------------------
def get_fundamentals(symbol: str) -> dict:
    key = f"fundamentals:{symbol}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        def _fetch():
            return yf.Ticker(yf_symbol(symbol), session=SHARED_SESSION).info

        with yfinance_limiter:
            info = with_retry(_fetch, what=f"yfinance fundamentals {symbol}")
    except Exception as exc:  # noqa: BLE001
        log.warning("Fundamentals fetch failed for %s: %s", symbol, exc)
        info = {}
    cache.set(key, info, ttl=settings.FUNDAMENTALS_CACHE_TTL_SECONDS)
    return info


# ---------------------------------------------------------------------------
# News - cached NEWS_CACHE_TTL_SECONDS (default 45 min). Also gated by
# newsapi_limiter (max 2 concurrent by default) since NewsAPI's free tier
# rate-limits hard on burst traffic - a 50-stock scan used to fire ~50
# near-simultaneous requests on first run before the cache was warm; now
# at most NEWSAPI_MAX_CONCURRENT run at once, and every subsequent scan
# within the TTL window reads from cache instead of calling NewsAPI at all.
# ---------------------------------------------------------------------------
def get_news_articles(company_name: str, max_articles: int = 8) -> list[dict]:
    if not settings.NEWSAPI_KEY:
        return []

    key = f"news:{company_name}"
    hit = cache.get(key)
    if hit is not None:
        return hit

    def _fetch():
        resp = SHARED_SESSION.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": f'"{company_name}"',
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_articles,
                "apiKey": settings.NEWSAPI_KEY,
            },
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json().get("articles", [])

    try:
        with newsapi_limiter:
            articles = with_retry(_fetch, what=f"NewsAPI {company_name}")
    except Exception as exc:  # noqa: BLE001
        log.warning("NewsAPI fetch failed for %s: %s", company_name, exc)
        articles = []

    # Cache the (possibly empty) result either way - an empty result from a
    # rate-limited call is still worth caching briefly so we don't retry
    # the same failing call again next scan cycle.
    cache.set(key, articles, ttl=settings.NEWS_CACHE_TTL_SECONDS if articles else 300)
    return articles
