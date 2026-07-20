"""
Background jobs (items #6 / #7 / #14 / #24):

  1. daily_915_scan            09:15 IST market open - full scan, stored to
                                ScanResult (historical record).
  2. continuous_intelligence   runs on a short interval WHILE THE MARKET IS
                                OPEN (request #2/#6), a slower interval
                                pre-market ("today's picks"), and a slower
                                interval post-market ("tomorrow's picks",
                                weighted toward fresh news - see
                                orchestrator.run_full_scan_async mode="post_market").
                                Rescans the tracked universe and *overwrites*
                                MarketIntelligence per symbol, so
                                /api/top10 is always served from precomputed
                                data instead of a cold start.
  3. learning_evaluation       daily - evaluates predictions whose horizon
                                has elapsed (see agents/learning_agent.py).
  4. auto_alert_sync           after every continuous_intelligence run -
                                auto-creates/refreshes BUY/TARGET/STOP alerts
                                for the current Top 10 (request #7).

Uses APScheduler's BackgroundScheduler (thread-based), calling the sync
`run_full_scan()` wrapper (which internally does asyncio.run) - this keeps
the scheduler dependency-light for local dev. For a true multi-process
production deployment, promote these to Celery Beat + worker tasks (see
worker.py) so scans run on separate processes/machines instead of the
API server's own thread pool.
"""
import json
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from agents.learning_agent import learning_agent
from config import settings
from database import MarketIntelligence, PriceAlert, ScanResult, SessionLocal
from market_hours import market_session
from orchestrator import run_full_scan
from utils import metrics
from utils.logger import get_logger

logger = get_logger("stocks.scheduler")

scheduler = BackgroundScheduler(timezone=settings.MARKET_TIMEZONE)

# Interval (minutes) for the continuous_intelligence job, switched based on
# market session so the system runs "as fast as possible" (request #2)
# during trading hours without hammering upstream APIs 24/7 for no reason.
SCAN_INTERVAL_BY_SESSION = {
    "open": max(1, settings.CONTINUOUS_SCAN_INTERVAL_MINUTES),  # e.g. every 5 min intraday
    "pre_market": 15,
    "post_market": 30,
}


def run_daily_scan_job():
    logger.info("Running scheduled 09:15 market scan...")
    try:
        with metrics.SCAN_LATENCY.time():
            results = run_full_scan()
        metrics.SCAN_REQUESTS.inc()
    except Exception as exc:  # noqa: BLE001
        logger.error("Scheduled daily scan failed: %s", exc)
        return

    db = SessionLocal()
    try:
        for r in results:
            db.add(ScanResult(
                symbol=r["symbol"], name=r["name"], sector=r["sector"], last_price=r["last_price"],
                expected_return_pct=r["expected_return_pct"], profit_probability_pct=r["profit_probability_pct"],
                loss_probability_pct=r["loss_probability_pct"], risk_label=r["risk_label"],
                confidence_pct=r["confidence_pct"], recommendation=r["recommendation"],
                reasons_json=json.dumps(r["reasons"]), suggested_quantity=r["suggested_quantity"],
                stop_loss_price=r["stop_loss_price"], target_price=r["target_price"],
            ))
        db.commit()
        logger.info("Daily scan complete: %d stocks stored.", len(results))
    finally:
        db.close()


def run_continuous_intelligence_job():
    """The 'precomputed market intelligence' layer (item #24), mode-aware
    per request #6: pre-market -> today's high-probability/low-risk picks;
    intraday -> continuously refreshed live rankings; post-market ->
    tomorrow's picks, weighted toward the latest news."""
    session = market_session()
    logger.info("Running continuous market-intelligence scan (session=%s)...", session)
    try:
        with metrics.SCAN_LATENCY.time():
            results = run_full_scan(top_n=None, mode=session)
        metrics.SCAN_REQUESTS.inc()
    except Exception as exc:  # noqa: BLE001
        logger.error("Continuous intelligence scan failed: %s", exc)
        return

    db = SessionLocal()
    try:
        for r in results:
            row = db.query(MarketIntelligence).filter_by(symbol=r["symbol"]).first()
            fields = dict(
                name=r["name"], sector=r["sector"], last_price=r["last_price"],
                day_change_pct=r.get("day_change_pct"), expected_return_pct=r["expected_return_pct"],
                profit_probability_pct=r["profit_probability_pct"], risk_label=r["risk_label"],
                confidence_pct=r["confidence_pct"], recommendation=r["recommendation"],
                rank_score=r["rank_score"], payload_json=json.dumps(r), updated_at=datetime.utcnow(),
            )
            if row:
                for k, v in fields.items():
                    setattr(row, k, v)
            else:
                db.add(MarketIntelligence(symbol=r["symbol"], **fields))
        db.commit()
        logger.info("Continuous scan complete: %d symbols refreshed (session=%s).", len(results), session)
    finally:
        db.close()

    _sync_top10_alerts(results[: settings.TOP_N_RECOMMENDATIONS])
    _reschedule_continuous_job_if_session_changed(session)


def _sync_top10_alerts(top_results: list[dict]):
    """Auto-creates/refreshes BUY, TARGET (sell), and STOP alerts for the
    current Top 10 (request #7). Existing auto-generated alerts for symbols
    that fell OUT of the Top 10 are deactivated so old alerts don't keep
    firing for stocks no longer being tracked as a recommendation."""
    db = SessionLocal()
    try:
        current_symbols = set()
        for r in top_results:
            if r["recommendation"] != "BUY" or not r.get("last_price"):
                continue
            current_symbols.add(r["symbol"])
            _upsert_auto_alert(db, r["symbol"], "BUY", r["last_price"])
            _upsert_auto_alert(db, r["symbol"], "SELL", r["target_price"])  # target = take-profit alert
            _upsert_auto_alert(db, r["symbol"], "STOP", r["stop_loss_price"])

        stale = (
            db.query(PriceAlert)
            .filter(PriceAlert.active == 1, PriceAlert.symbol.notin_(current_symbols) if current_symbols else True,
                     PriceAlert.auto_generated == 1)
            .all()
        )
        for a in stale:
            a.active = 0
        db.commit()
    finally:
        db.close()


def _upsert_auto_alert(db, symbol: str, alert_type: str, target_price: float):
    if not target_price:
        return
    existing = (
        db.query(PriceAlert)
        .filter_by(symbol=symbol, alert_type=alert_type, auto_generated=1, active=1)
        .first()
    )
    if existing:
        existing.target_price = target_price
    else:
        db.add(PriceAlert(symbol=symbol, alert_type=alert_type, target_price=target_price,
                           active=1, auto_generated=1))


def run_learning_evaluation_job():
    try:
        n = learning_agent.evaluate_due_predictions()
        logger.info("Learning agent evaluated %d matured predictions.", n)
    except Exception as exc:  # noqa: BLE001
        logger.error("Learning evaluation failed: %s", exc)


_last_scheduled_session = {"value": None}


def _reschedule_continuous_job_if_session_changed(session: str):
    """Bumps the continuous_intelligence job's own interval when the market
    session changes (pre_market -> open -> post_market), so refresh speed
    actually follows the 9:15-3:30 window instead of running at one fixed
    cadence all day (request #2 / #6)."""
    if _last_scheduled_session["value"] == session:
        return
    _last_scheduled_session["value"] = session
    minutes = SCAN_INTERVAL_BY_SESSION.get(session, 10)
    scheduler.reschedule_job("continuous_intelligence", trigger=IntervalTrigger(minutes=minutes))
    logger.info("Market session changed to '%s' - continuous scan interval now every %d minute(s).",
                session, minutes)


def start_scheduler():
    scheduler.add_job(
        run_daily_scan_job,
        CronTrigger(day_of_week="mon-fri", hour=settings.DAILY_SCAN_HOUR, minute=settings.DAILY_SCAN_MINUTE,
                    timezone=settings.MARKET_TIMEZONE),
        id="daily_915_scan", replace_existing=True,
    )
    scheduler.add_job(
        run_continuous_intelligence_job,
        IntervalTrigger(minutes=SCAN_INTERVAL_BY_SESSION[market_session()]),
        id="continuous_intelligence", replace_existing=True,
    )
    scheduler.add_job(
        run_learning_evaluation_job,
        CronTrigger(hour=18, minute=0, timezone=settings.MARKET_TIMEZONE),  # once daily, after market close
        id="learning_evaluation", replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started: daily scan %02d:%02d %s | continuous scan session-aware (5-10s live tick via "
        "WebSocket, %dm full re-rank intraday) | learning eval 18:00 %s",
        settings.DAILY_SCAN_HOUR, settings.DAILY_SCAN_MINUTE, settings.MARKET_TIMEZONE,
        settings.CONTINUOUS_SCAN_INTERVAL_MINUTES, settings.MARKET_TIMEZONE,
    )
