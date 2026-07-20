"""
Shared HTTP infrastructure (fixes the "Connection pool is full" and
repeated-429 errors from your logs).

Root cause of what you saw: every concurrent stock analysis was opening
its own ad-hoc HTTP connection to Yahoo Finance / NewsAPI, all racing at
once. urllib3's default connection pool per host is only 10 connections,
so with SCAN_CONCURRENCY=100 the other 90 requests got "pool is full,
discarding connection" - and Yahoo/NewsAPI started returning 429 Too Many
Requests because ~50-100 requests were landing on them within the same
second.

Fix, in one place, used everywhere a call goes out to yfinance or NewsAPI:
  1. ONE shared requests.Session with a much bigger connection pool
     (SHARED_SESSION below), so concurrent requests reuse connections
     instead of each opening a new one.
  2. A urllib3 Retry policy baked into that session's HTTPAdapter, so
     transient 429/5xx responses are retried with exponential backoff
     automatically at the transport layer.
  3. `with_retry()` - an extra application-level retry wrapper for the
     yfinance calls, which don't always raise a catchable HTTP error for
     429s (yfinance sometimes just returns an empty body, which then fails
     JSON parsing one level up - see groww_client.py / data_service.py).
"""
import random
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.logger import get_logger

log = get_logger("stocks.http")

_retry_policy = Retry(
    total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]), respect_retry_after_header=True,
)
_adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=_retry_policy)

SHARED_SESSION = requests.Session()
SHARED_SESSION.mount("https://", _adapter)
SHARED_SESSION.mount("http://", _adapter)
SHARED_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; StocksApp/1.0)"})


def with_retry(fn, *args, attempts: int = 3, base_delay: float = 1.0, what: str = "request", **kwargs):
    """Application-level retry for calls where a rate limit doesn't come
    back as a clean HTTPError (yfinance is notorious for this - a 429 often
    surfaces as 'Expecting value: line 1 column 1' from a failed JSON parse
    on an empty body). Exponential backoff with jitter; logs once per
    attempt, not once per stock per second."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            is_rate_limit = any(s in str(exc).lower() for s in ("429", "too many requests", "rate limit"))
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            if is_rate_limit:
                delay *= 2  # back off harder specifically for rate limits
            log.debug("%s failed (attempt %d/%d): %s - retrying in %.1fs", what, attempt, attempts, exc, delay)
            time.sleep(delay)
    raise last_exc
