"""
Stocks - FastAPI backend entrypoint / API Gateway (item #23).

Run locally:  uvicorn main:app --reload --port 8000
Docker:       docker compose up   (see ../docker-compose.yml)
"""
from pathlib import Path
import asyncio
import json
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc
from sqlalchemy.orm import Session

import groww_client
from agents.learning_agent import learning_agent
from agents.market_data_agent import get_live_prices_bulk
from cache.cache_manager import cache
from config import settings
from database import MarketIntelligence, PriceAlert, ScanResult, SessionLocal, get_session, init_db
from models import AlertCreate, PortfolioRequest, PositionSizeRequest
from market_hours import is_market_open, market_session, now_ist
from orchestrator import analyze_symbol_async, build_portfolio_async, run_full_scan_async, run_single_agent
from scheduler import scheduler, start_scheduler
from stock_universe import STOCK_UNIVERSE, SYMBOL_SET
from utils import metrics
from utils.logger import get_logger

log = get_logger("stocks.main")


# ---------------------------------------------------------------------------
# Rate limiting (item #20 baseline) - simple fixed-window in-memory limiter.
# Good enough for a single-instance deployment; swap for a Redis-backed
# limiter (e.g. slowapi + Redis) once you're running multiple API replicas
# behind a load balancer, since in-memory state doesn't share across them.
# ---------------------------------------------------------------------------
_request_log: dict[str, list[float]] = defaultdict(list)


async def rate_limiter(request: Request):
    client_id = request.client.host if request.client else "unknown"
    now = time.time()
    window = _request_log[client_id]
    window[:] = [t for t in window if now - t < 60]
    if len(window) >= settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    window.append(now)


async def require_api_key(x_api_key: str = Header(default="")):
    """No-op when STOCKS_API_KEY is unset (local dev). Set it in production
    and require it on any endpoint that can trigger a full scan or
    place-order-adjacent action (item #20)."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_scheduler()
    scheduler.add_job(
        check_price_alerts, IntervalTrigger(seconds=settings.LIVE_PRICE_POLL_SECONDS),
        id="alert_checker", replace_existing=True,
    )
    log.info("Stocks backend started. Live mode: %s | Universe size: %d | Redis cache: %s",
             groww_client.is_live(), len(STOCK_UNIVERSE), cache.mode)
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="Stocks", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Universe / search
# ---------------------------------------------------------------------------
@app.get("/api/universe")
def get_universe():
    return {"stocks": STOCK_UNIVERSE, "count": len(STOCK_UNIVERSE), "live_mode": groww_client.is_live()}


# ---------------------------------------------------------------------------
# Top-N recommendations - served from the precomputed MarketIntelligence
# table first (item #24), falling back to a live scan only if nothing has
# been computed yet (e.g. first-ever boot before the scheduler's first run).
# ---------------------------------------------------------------------------
@app.get("/api/top10")
async def get_top10(force_refresh: bool = False, db: Session = Depends(get_session)):
    """Auto mode-aware (request #6): before 09:15 IST this effectively
    reflects "today's" high-probability/low-risk picks; after 15:30 IST the
    background scheduler has already re-ranked toward "tomorrow's" picks
    weighted by fresh news. See /api/market-status for the current session."""
    session = market_session()
    if not force_refresh:
        rows = (
            db.query(MarketIntelligence)
            .order_by(desc(MarketIntelligence.rank_score))
            .limit(settings.TOP_N_RECOMMENDATIONS)
            .all()
        )
        if rows:
            latest_ts = max(r.updated_at for r in rows)
            return {"scanned_at": latest_ts.isoformat(), "source": "precomputed_market_intelligence",
                     "market_session": session, "results": [json.loads(r.payload_json) for r in rows]}

    with metrics.SCAN_LATENCY.time():
        results = await run_full_scan_async(mode=session)
    metrics.SCAN_REQUESTS.inc()
    return {"scanned_at": datetime.utcnow().isoformat(), "source": "live_scan", "market_session": session,
             "results": results}


@app.get("/api/market-status")
def get_market_status():
    """Powers the frontend's live countdown / session badge (request #2 & #10)."""
    session = market_session()
    now = now_ist()
    return {
        "session": session, "is_open": is_market_open(), "server_time_ist": now.isoformat(),
        "market_open_time": f"{settings.MARKET_OPEN_HOUR:02d}:{settings.MARKET_OPEN_MINUTE:02d}",
        "market_close_time": f"{settings.MARKET_CLOSE_HOUR:02d}:{settings.MARKET_CLOSE_MINUTE:02d}",
        "live_refresh_seconds": settings.LIVE_PRICE_POLL_SECONDS,
    }


@app.post("/api/scan", dependencies=[Depends(require_api_key)])
async def trigger_manual_scan(full_universe: bool = False):
    """Manually re-run a scan. full_universe=true scans everything tracked
    (can take a while on the curated ~90 list; considerably longer on the
    full NSE universe - see README 'How long does a full scan take')."""
    session = market_session()
    with metrics.SCAN_LATENCY.time():
        results = await run_full_scan_async(top_n=None if full_universe else settings.TOP_N_RECOMMENDATIONS,
                                             mode=session)
    metrics.SCAN_REQUESTS.inc()
    return {"scanned_at": datetime.utcnow().isoformat(), "count": len(results), "market_session": session,
             "results": results}


# ---------------------------------------------------------------------------
# Single-stock analysis + agent router (item #18: run only what's needed)
# ---------------------------------------------------------------------------
@app.get("/api/stock/{symbol}")
async def get_stock_analysis(symbol: str, capital: float = None, risk_per_trade_pct: float = None,
                              _rl=Depends(rate_limiter)):
    symbol = symbol.upper()
    if symbol not in SYMBOL_SET:
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not in the tracked NSE universe.")
    try:
        with metrics.ANALYSIS_LATENCY.time():
            result = await analyze_symbol_async(symbol, capital=capital, risk_per_trade_pct=risk_per_trade_pct)
        metrics.ANALYSIS_REQUESTS.inc()
        return result
    except Exception as exc:  # noqa: BLE001
        metrics.AGENT_ERRORS.labels(agent="orchestrator").inc()
        log.exception("Analysis failed for %s", symbol)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/agent/{agent_name}/{symbol}")
async def run_agent(agent_name: str, symbol: str, _rl=Depends(rate_limiter)):
    """Agent router (item #18): call a single agent directly instead of the
    full pipeline, e.g. /api/agent/technical/RELIANCE for just the
    technical score - cheaper than a full analysis when that's all the
    caller needs."""
    symbol = symbol.upper()
    if symbol not in SYMBOL_SET:
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not in the tracked NSE universe.")
    try:
        return await run_single_agent(agent_name, symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/position-size")
async def recompute_position_size(req: PositionSizeRequest):
    return await analyze_symbol_async(req.symbol, capital=req.capital, risk_per_trade_pct=req.risk_per_trade_pct)


# ---------------------------------------------------------------------------
# Portfolio Agent (item #11)
# ---------------------------------------------------------------------------
@app.post("/api/portfolio")
async def build_portfolio(req: PortfolioRequest, _rl=Depends(rate_limiter)):
    unknown = [s for s in req.symbols if s.upper() not in SYMBOL_SET]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Not in tracked universe: {unknown}")
    return await build_portfolio_async(
        [s.upper() for s in req.symbols], capital=req.capital,
        max_positions=req.max_positions or 8, risk_per_trade_pct=req.risk_per_trade_pct,
    )


# ---------------------------------------------------------------------------
# Learning Agent (items #13-14) - prediction accuracy over time
# ---------------------------------------------------------------------------
@app.get("/api/accuracy")
def get_accuracy_report():
    return learning_agent.accuracy_report()


# ---------------------------------------------------------------------------
# Live prices - REST (polling) and WebSocket (streaming, item #7)
# ---------------------------------------------------------------------------
@app.get("/api/live-prices")
def get_live_prices(symbols: str):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    prices = get_live_prices_bulk(symbol_list)
    return {"prices": prices, "live_mode": groww_client.is_live(), "timestamp": datetime.utcnow().isoformat()}


@app.websocket("/ws/live-prices")
async def ws_live_prices(websocket: WebSocket):
    """Streams live prices for client-supplied symbols every
    LIVE_PRICE_POLL_SECONDS, replacing the old poll-and-refresh loop
    (item #7). Client sends a JSON message once to subscribe:
        {"symbols": ["RELIANCE", "TCS"]}
    and then just listens for price updates.
    """
    await websocket.accept()
    symbols: list[str] = []
    try:
        subscribe_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        symbols = [s.strip().upper() for s in subscribe_msg.get("symbols", []) if s.strip()]
    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
        await websocket.send_json({"error": "Expected {'symbols': [...]} within 10s of connecting."})
        await websocket.close()
        return

    try:
        while True:
            prices = get_live_prices_bulk(symbols)
            triggered = _check_alerts_for_symbols(symbols, prices)
            await websocket.send_json({
                "prices": prices, "timestamp": datetime.utcnow().isoformat(), "live_mode": groww_client.is_live(),
                "market_session": market_session(), "triggered_alerts": triggered,
            })
            # Fast tick while the market's open (request #2: 5-10s); back off
            # outside market hours so the tab doesn't hammer the API for no reason.
            interval = settings.LIVE_PRICE_POLL_SECONDS if is_market_open() else max(settings.LIVE_PRICE_POLL_SECONDS, 30)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        log.debug("WebSocket client disconnected (%d symbols)", len(symbols))


def _check_alerts_for_symbols(symbols: list[str], prices: dict) -> list[dict]:
    """Checks active alerts for the currently-subscribed symbols against
    the just-fetched prices and returns any that just triggered, WITHOUT
    mutating alert state (that's check_price_alerts()'s job, which runs on
    its own schedule) - this just gives the live frontend an immediate
    push instead of waiting for the next alert_checker cycle (request #7)."""
    if not symbols:
        return []
    db = SessionLocal()
    try:
        alerts = db.query(PriceAlert).filter(PriceAlert.symbol.in_(symbols), PriceAlert.active == 1).all()
        hits = []
        for a in alerts:
            price = prices.get(a.symbol)
            if price is None:
                continue
            hit = (
                (a.alert_type == "BUY" and price <= a.target_price)
                or (a.alert_type == "SELL" and price >= a.target_price)
                or (a.alert_type == "STOP" and price <= a.target_price)
            )
            if hit:
                hits.append({"symbol": a.symbol, "alert_type": a.alert_type, "target_price": a.target_price,
                              "current_price": price, "auto_generated": bool(a.auto_generated)})
        return hits
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Price alerts
# ---------------------------------------------------------------------------
@app.post("/api/alerts")
def create_alert(alert: AlertCreate, db: Session = Depends(get_session)):
    if alert.alert_type not in ("BUY", "SELL", "STOP"):
        raise HTTPException(status_code=400, detail="alert_type must be 'BUY', 'SELL', or 'STOP'")
    row = PriceAlert(symbol=alert.symbol.upper(), alert_type=alert.alert_type, target_price=alert.target_price)
    db.add(row)
    db.commit()
    db.refresh(row)
    return _alert_to_dict(row)


@app.get("/api/alerts")
def list_alerts(db: Session = Depends(get_session)):
    rows = db.query(PriceAlert).order_by(desc(PriceAlert.created_at)).all()
    return {"alerts": [_alert_to_dict(r) for r in rows]}


@app.delete("/api/alerts/{alert_id}")
def delete_alert(alert_id: int, db: Session = Depends(get_session)):
    row = db.query(PriceAlert).filter(PriceAlert.id == alert_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    db.delete(row)
    db.commit()
    return {"deleted": alert_id}


def check_price_alerts():
    db = SessionLocal()
    try:
        active = db.query(PriceAlert).filter(PriceAlert.active == 1).all()
        if not active:
            return
        symbols = list({a.symbol for a in active})
        prices = get_live_prices_bulk(symbols)
        for a in active:
            price = prices.get(a.symbol)
            if price is None:
                continue
            hit = (
                (a.alert_type == "BUY" and price <= a.target_price)
                or (a.alert_type == "SELL" and price >= a.target_price)
                or (a.alert_type == "STOP" and price <= a.target_price)
            )
            if hit:
                a.active = 0
                a.triggered_at = datetime.utcnow()
                log.info("ALERT TRIGGERED: %s %s at target %.2f (LTP %.2f)", a.symbol, a.alert_type,
                          a.target_price, price)
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Ops / monitoring (item #15)
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok", "cache_backend": cache.mode, "live_mode": groww_client.is_live(),
        "universe_size": len(STOCK_UNIVERSE), "database": settings.DATABASE_URL.split("://")[0],
        "celery_enabled": settings.USE_CELERY,
    }


@app.get("/metrics")
def prometheus_metrics():
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _alert_to_dict(a: PriceAlert) -> dict:
    return {
        "id": a.id, "symbol": a.symbol, "alert_type": a.alert_type, "target_price": a.target_price,
        "active": bool(a.active), "auto_generated": bool(a.auto_generated),
        "created_at": a.created_at.isoformat(),
        "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
    }


# Serve the single-file frontend at "/" (mount AFTER all /api routes)
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend"
)
