"""
Monitoring (item #15). Exposes a /metrics endpoint in Prometheus text
format via prometheus_client. Wire it into Grafana with the bundled
monitoring/prometheus.yml + docker-compose.yml service.

If prometheus_client isn't installed, every call here becomes a no-op so
the app still runs without the monitoring stack.
"""
from utils.logger import get_logger

log = get_logger("stocks.metrics")

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    ANALYSIS_REQUESTS = Counter("stocks_analysis_requests_total", "Total single-stock analysis requests")
    SCAN_REQUESTS = Counter("stocks_scan_requests_total", "Total full-universe scan requests")
    ANALYSIS_LATENCY = Histogram("stocks_analysis_latency_seconds", "Single-stock analysis latency")
    SCAN_LATENCY = Histogram("stocks_scan_latency_seconds", "Full-universe scan latency")
    AGENT_ERRORS = Counter("stocks_agent_errors_total", "Agent execution errors", ["agent"])

    _ENABLED = True
except ImportError:  # pragma: no cover
    log.warning("prometheus_client not installed - /metrics will return empty output.")
    _ENABLED = False

    class _NoOp:
        def labels(self, *a, **kw):
            return self

        def inc(self, *a, **kw):
            pass

        def time(self):
            import contextlib

            return contextlib.nullcontext()

    ANALYSIS_REQUESTS = SCAN_REQUESTS = ANALYSIS_LATENCY = SCAN_LATENCY = AGENT_ERRORS = _NoOp()


def render() -> tuple[bytes, str]:
    if not _ENABLED:
        return b"", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST
