"""
Stocks - central configuration.
All secrets are read from environment variables / a local .env file.
NEVER hardcode API keys in source. See .env.example for the full list.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ---- Groww Trading API (live prices, LTP, OHLC, quotes) ----
    GROWW_API_KEY: str = os.getenv("GROWW_API_KEY", "")
    GROWW_API_SECRET: str = os.getenv("GROWW_API_SECRET", "")
    GROWW_TOTP_SECRET: str = os.getenv("GROWW_TOTP_SECRET", "")
    GROWW_AUTH_MODE: str = os.getenv("GROWW_AUTH_MODE", "totp")  # "totp" or "secret"

    # ---- Optional third-party data / AI keys (system degrades gracefully
    # if these are missing - see agents/*.py fallbacks) ----
    NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # ---- App behaviour ----
    MARKET_TIMEZONE: str = "Asia/Kolkata"
    DAILY_SCAN_HOUR: int = int(os.getenv("DAILY_SCAN_HOUR", "9"))
    DAILY_SCAN_MINUTE: int = int(os.getenv("DAILY_SCAN_MINUTE", "15"))
    TOP_N_RECOMMENDATIONS: int = int(os.getenv("TOP_N_RECOMMENDATIONS", "10"))

    DEFAULT_CAPITAL_INR: float = float(os.getenv("DEFAULT_CAPITAL_INR", "100000"))
    DEFAULT_RISK_PER_TRADE_PCT: float = float(os.getenv("DEFAULT_RISK_PER_TRADE_PCT", "1.0"))
    MAX_ALLOCATION_PER_STOCK_PCT: float = float(os.getenv("MAX_ALLOCATION_PER_STOCK_PCT", "10.0"))

    # ---- Persistence (item #5: swap for Postgres/TimescaleDB in prod by
    # just changing DATABASE_URL - the ORM layer doesn't change) ----
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./stocks.db")

    # ---- Caching (item #3) ----
    REDIS_URL: str = os.getenv("REDIS_URL", "")

    # ---- Background workers (item #6) ----
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", "redis://localhost:6379/1"))
    USE_CELERY: bool = os.getenv("USE_CELERY", "false").lower() == "true"

    # ---- Continuous market intelligence layer (item #24) ----
    # How often the background scanner refreshes precomputed rankings for
    # the whole tracked universe during market hours. Outside market hours
    # the scheduler switches to the slower pre/post-market modes below.
    CONTINUOUS_SCAN_INTERVAL_MINUTES: int = int(os.getenv("CONTINUOUS_SCAN_INTERVAL_MINUTES", "5"))
    # Caps concurrent in-flight stock analyses so a full-universe scan
    # doesn't fire hundreds of simultaneous upstream requests at once.
    # "100 parallel agents" (request #3) = 100 concurrent analyses in-flight.
    # Real ceiling is your upstream data provider's rate limit, not this
    # number - raise it, watch your logs for 429s, and back off if you see them.
    # Safe default per your logs: 100 triggered simultaneous Groww
    # rate-limiting, Yahoo Finance 429s, and "connection pool is full"
    # warnings. 8 is a conservative starting point for the FREE data tier
    # (yfinance + NewsAPI free plan + Groww's own limits); raise gradually
    # and watch the logs for 429s, or move to a paid data feed / Celery
    # worker fan-out (worker.py) to safely go higher (request #3's "100
    # parallel agents" is achievable, just not against free-tier providers
    # from a single process - see README "On rate limits").
    SCAN_CONCURRENCY: int = int(os.getenv("SCAN_CONCURRENCY", "8"))
    # Per-provider caps (utils/rate_limit.py) - the actual fix for
    # "Connection pool is full" and repeated 429s. These stay low
    # regardless of SCAN_CONCURRENCY, since a handful of free-tier
    # providers can't take more than a few concurrent requests each.
    YFINANCE_MAX_CONCURRENT: int = int(os.getenv("YFINANCE_MAX_CONCURRENT", "5"))
    NEWSAPI_MAX_CONCURRENT: int = int(os.getenv("NEWSAPI_MAX_CONCURRENT", "2"))
    GROWW_MAX_CONCURRENT: int = int(os.getenv("GROWW_MAX_CONCURRENT", "3"))
    # After a Groww auth call comes back rate-limited, stop retrying
    # authentication for this many seconds (previously it re-authenticated
    # on EVERY single stock, which is what caused the repeated
    # "rate limit exceeded" spam in your logs - see groww_client.py).
    GROWW_AUTH_COOLDOWN_SECONDS: int = int(os.getenv("GROWW_AUTH_COOLDOWN_SECONDS", "300"))
    # Cap on how many symbols get pulled into a "full universe" scan.
    MAX_UNIVERSE_SIZE: int = int(os.getenv("MAX_UNIVERSE_SIZE", "250"))
    # Use the full ~2000-symbol NSE list (fetched from the NSE archive,
    # cached to disk) instead of the curated ~90-stock list.
    USE_FULL_NSE_UNIVERSE: bool = os.getenv("USE_FULL_NSE_UNIVERSE", "false").lower() == "true"

    # ---- Universe filters (request #4) ----
    MAX_STOCK_PRICE_INR: float = float(os.getenv("MAX_STOCK_PRICE_INR", "2000"))
    MIN_LISTING_YEARS: int = int(os.getenv("MIN_LISTING_YEARS", "5"))
    EXCLUDE_LITIGATION_FLAGGED: bool = os.getenv("EXCLUDE_LITIGATION_FLAGGED", "true").lower() == "true"

    # ---- Market hours (IST) + scan-mode switching (request #2 / #6) ----
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 15
    MARKET_CLOSE_HOUR: int = 15
    MARKET_CLOSE_MINUTE: int = 30


    # ---- Security (item #20 - baseline; see README for what's still
    # your responsibility before going to production) ----
    API_KEY: str = os.getenv("STOCKS_API_KEY", "")  # empty = auth disabled (local dev)
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

    LIVE_PRICE_POLL_SECONDS: int = int(os.getenv("LIVE_PRICE_POLL_SECONDS", "7"))  # request #2: 5-10s during market hours

    # ---- Caching TTLs (tune these up if you're on free-tier providers -
    # longer TTLs directly reduce how often the rate limits in the section
    # above get hit at all) ----
    NEWS_CACHE_TTL_SECONDS: int = int(os.getenv("NEWS_CACHE_TTL_SECONDS", "2700"))  # 45 min (was 30)
    FUNDAMENTALS_CACHE_TTL_SECONDS: int = int(os.getenv("FUNDAMENTALS_CACHE_TTL_SECONDS", "21600"))  # 6h
    QUOTE_CACHE_TTL_SECONDS: int = int(os.getenv("QUOTE_CACHE_TTL_SECONDS", "5"))


settings = Settings()
