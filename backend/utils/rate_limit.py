"""
Per-provider concurrency limits (fixes "Connection pool is full" / 429s at
the root instead of just retrying around them).

SCAN_CONCURRENCY controls how many STOCKS are analyzed at once, but each
stock analysis fans out to 3+ external calls (quote, history, fundamentals,
news). At SCAN_CONCURRENCY=100, that's potentially 300-400 simultaneous
outbound requests to a handful of providers who rate-limit far below that
- Yahoo Finance and NewsAPI's free tier in particular.

These semaphores cap concurrent in-flight requests PER PROVIDER, separate
from and much lower than SCAN_CONCURRENCY, so raising scan concurrency for
speed never re-introduces the rate-limit storm - the provider limiter is
the actual bottleneck, not the scan loop. They're plain `threading.Semaphore`
(not asyncio) because data_service.py's functions are sync and run inside
`asyncio.to_thread`, i.e. real OS threads.
"""
import threading

from config import settings

yfinance_limiter = threading.Semaphore(settings.YFINANCE_MAX_CONCURRENT)
newsapi_limiter = threading.Semaphore(settings.NEWSAPI_MAX_CONCURRENT)
groww_limiter = threading.Semaphore(settings.GROWW_MAX_CONCURRENT)
