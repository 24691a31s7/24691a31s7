"""
Background workers (item #6): Celery scaffold so full-universe scans can
run on separate worker processes/machines instead of blocking the API
server or its in-process scheduler thread.

This is OPTIONAL and off by default (USE_CELERY=false) - the single-process
APScheduler path in scheduler.py works with zero extra infrastructure.
Turn this on once you actually have a Redis broker and want to scale scans
across multiple worker machines:

    # start a broker (or use the one from docker-compose.yml)
    redis-server

    # start N worker processes (each can be its own machine)
    celery -A worker worker --loglevel=info --concurrency=8

    # trigger a distributed scan
    python -c "from worker import scan_universe_task; scan_universe_task.delay()"

Chunking strategy: instead of one giant task, the universe is split into
chunks of CHUNK_SIZE symbols and dispatched as a Celery group, so 2,000
stocks can be spread across as many workers as you have (item #24: "200
Workers -> Worker 1: 20 stocks, Worker 2: 20 stocks...").
"""
from celery import Celery, group

from config import settings
from stock_universe import STOCK_UNIVERSE

CHUNK_SIZE = 20

celery_app = Celery(
    "stocks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"])


@celery_app.task(name="stocks.analyze_symbol")
def analyze_symbol_task(symbol: str) -> dict:
    from orchestrator import analyze_symbol  # local import: workers don't need FastAPI

    return analyze_symbol(symbol)


@celery_app.task(name="stocks.scan_chunk")
def scan_chunk_task(symbols: list[str]) -> list[dict]:
    from orchestrator import analyze_symbol

    results = []
    for sym in symbols:
        try:
            results.append(analyze_symbol(sym))
        except Exception:  # noqa: BLE001
            continue
    return results


def dispatch_full_universe_scan():
    """Fan the whole tracked universe out across workers in CHUNK_SIZE
    batches and return the Celery GroupResult (call `.get()` to block for
    results, or poll `.ready()` for a non-blocking status check)."""
    symbols = [s["symbol"] for s in STOCK_UNIVERSE]
    chunks = [symbols[i:i + CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
    job = group(scan_chunk_task.s(chunk) for chunk in chunks)
    return job.apply_async()
