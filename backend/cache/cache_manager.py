"""
Cache layer used to avoid hammering Yahoo/Groww/NewsAPI on every request
(review item #3: "If someone requests SBIN 100 times, only one API request
should be made").

Backed by Redis if REDIS_URL is configured and reachable, otherwise falls
back to an in-process TTL dict automatically - same behavior pattern as
groww_client's yfinance fallback, so local dev never needs Redis installed.

Usage:
    from cache.cache_manager import cache

    cache.set("quote:RELIANCE", data, ttl=5)
    data = cache.get("quote:RELIANCE")

    @cached(ttl=86400, key_fn=lambda symbol, **kw: f"history:{symbol}")
    def get_price_history(symbol): ...
"""
import functools
import json
import threading
import time
from typing import Any, Callable, Optional

from utils.logger import get_logger

log = get_logger("stocks.cache")


class _InMemoryBackend:
    """Thread-safe: SCAN_CONCURRENCY concurrent stock analyses run as real
    OS threads (via asyncio.to_thread), all reading/writing this cache
    simultaneously - a bare dict is GIL-safe for single ops but a lock
    keeps get/check-expiry/pop atomic as a group, avoiding rare double-fetch
    races on cache expiry."""
    def __init__(self):
        self._store: dict[str, tuple[float, str]] = {}  # key -> (expires_at, json_value)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at < time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl: int):
        with self._lock:
            self._store[key] = (time.time() + ttl, value)

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def clear_prefix(self, prefix: str):
        with self._lock:
            for k in [k for k in self._store if k.startswith(prefix)]:
                self._store.pop(k, None)


class _RedisBackend:
    def __init__(self, url: str):
        import redis  # local import so `redis` package is optional

        self._client = redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        self._client.ping()  # fail fast if unreachable

    def get(self, key: str) -> Optional[str]:
        return self._client.get(key)

    def set(self, key: str, value: str, ttl: int):
        self._client.setex(key, ttl, value)

    def delete(self, key: str):
        self._client.delete(key)

    def clear_prefix(self, prefix: str):
        for k in self._client.scan_iter(f"{prefix}*"):
            self._client.delete(k)


class CacheManager:
    def __init__(self):
        self.backend = None
        self.mode = "memory"
        self._try_init_redis()
        if self.backend is None:
            self.backend = _InMemoryBackend()
            self.mode = "memory"

    def _try_init_redis(self):
        import os

        url = os.getenv("REDIS_URL", "")
        if not url:
            return
        try:
            self.backend = _RedisBackend(url)
            self.mode = "redis"
            log.info("Cache backend: Redis (%s)", url)
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis unavailable (%s) - falling back to in-memory cache.", exc)
            self.backend = None

    def get(self, key: str) -> Optional[Any]:
        raw = self.backend.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set(self, key: str, value: Any, ttl: int = 60):
        try:
            payload = json.dumps(value, default=str)
        except TypeError:
            payload = str(value)
        self.backend.set(key, payload, ttl)

    def delete(self, key: str):
        self.backend.delete(key)

    def clear_prefix(self, prefix: str):
        self.backend.clear_prefix(prefix)


cache = CacheManager()


def cached(ttl: int, key_fn: Callable[..., str]):
    """
    Decorator for sync functions. key_fn receives the same args/kwargs as
    the wrapped function and must return a cache key string.
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator
