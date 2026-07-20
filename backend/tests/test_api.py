"""
Integration tests for the FastAPI app surface. Network-dependent endpoints
(/api/stock/{symbol}, /api/top10 without cache) will hit real Yahoo/Groww
APIs - keep those minimal/optional in CI environments without outbound
network access.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from database import init_db  # noqa: E402

init_db()
client = TestClient(main.app)
client.__enter__()  # trigger the lifespan startup (init_db + scheduler)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["cache_backend"] in ("memory", "redis")


def test_universe():
    r = client.get("/api/universe")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0


def test_unknown_symbol_returns_404():
    r = client.get("/api/stock/NOTASTOCK")
    assert r.status_code == 404


def test_unknown_agent_router_symbol_returns_404():
    r = client.get("/api/agent/technical/NOTASTOCK")
    assert r.status_code == 404


def test_alerts_crud():
    r = client.post("/api/alerts", json={"symbol": "TCS", "alert_type": "BUY", "target_price": 3500})
    assert r.status_code == 200
    alert_id = r.json()["id"]

    r = client.get("/api/alerts")
    assert any(a["id"] == alert_id for a in r.json()["alerts"])

    r = client.delete(f"/api/alerts/{alert_id}")
    assert r.status_code == 200


def test_invalid_alert_type_rejected():
    r = client.post("/api/alerts", json={"symbol": "TCS", "alert_type": "HOLD", "target_price": 100})
    assert r.status_code == 400


def test_accuracy_report_empty_state():
    r = client.get("/api/accuracy")
    assert r.status_code == 200
    assert "overall_accuracy_pct" in r.json()


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
