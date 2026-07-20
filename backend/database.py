"""
Persistence layer. SQLite by default for zero-setup local dev; point
DATABASE_URL at Postgres (or Postgres + TimescaleDB extension for the
market-data hypertables) in production - nothing else in this file changes,
since SQLAlchemy abstracts the dialect (item #5).

    DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/stocks

Tables:
  - ScanResult          per-symbol result from a scan (daily or continuous)
  - PriceHistoryCache   persisted OHLC, refreshed at most once/24h
  - PriceAlert          user-configured BUY/SELL trigger levels
  - MarketIntelligence  ONE row per symbol, always overwritten with the
                         latest continuous-scan result (item #24: "results
                         come from precomputed intelligence, not a cold
                         start"). This is what /api/top10 and /api/stock/*
                         read from first.
  - PredictionLog       every prediction the system has ever made, plus
                         (once known) the actual realized return - this is
                         the substrate for the Memory/Learning agents
                         (items #13-14).
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

_connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ScanResult(Base):
    __tablename__ = "scan_results"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    name = Column(String)
    sector = Column(String)
    last_price = Column(Float)
    expected_return_pct = Column(Float)
    profit_probability_pct = Column(Float)
    loss_probability_pct = Column(Float)
    risk_label = Column(String)
    confidence_pct = Column(Float)
    recommendation = Column(String)
    reasons_json = Column(String)
    suggested_quantity = Column(Integer)
    stop_loss_price = Column(Float)
    target_price = Column(Float)
    scanned_at = Column(DateTime, default=datetime.utcnow, index=True)


class PriceHistoryCache(Base):
    __tablename__ = "price_history_cache"
    symbol = Column(String, primary_key=True)
    period = Column(String, primary_key=True)
    interval = Column(String, primary_key=True)
    data_json = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)


class PriceAlert(Base):
    __tablename__ = "price_alerts"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    alert_type = Column(String)  # "BUY", "SELL", or "STOP"
    target_price = Column(Float)
    active = Column(Integer, default=1)
    auto_generated = Column(Integer, default=0)  # 1 = created by the Top-10 auto-alert sync (request #7)
    created_at = Column(DateTime, default=datetime.utcnow)
    triggered_at = Column(DateTime, nullable=True)


class MarketIntelligence(Base):
    """One row per symbol, continuously overwritten by the background
    scanner - the 'precomputed intelligence layer' (item #24)."""
    __tablename__ = "market_intelligence"
    symbol = Column(String, primary_key=True)
    name = Column(String)
    sector = Column(String)
    last_price = Column(Float)
    day_change_pct = Column(Float)
    expected_return_pct = Column(Float)
    profit_probability_pct = Column(Float)
    risk_label = Column(String)
    confidence_pct = Column(Float)
    recommendation = Column(String)
    rank_score = Column(Float, index=True)
    payload_json = Column(String)  # full analysis payload, JSON-encoded
    updated_at = Column(DateTime, default=datetime.utcnow, index=True)


class PredictionLog(Base):
    """Every prediction ever made, plus (once available) the realized
    outcome N days later. Feeds the Learning Agent's accuracy metrics."""
    __tablename__ = "prediction_log"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    predicted_at = Column(DateTime, default=datetime.utcnow, index=True)
    price_at_prediction = Column(Float)
    predicted_return_pct = Column(Float)
    predicted_direction = Column(String)  # "BUY" / "SELL" / "HOLD"
    horizon_days = Column(Integer, default=5)
    actual_return_pct = Column(Float, nullable=True)
    actual_direction_correct = Column(Integer, nullable=True)  # 1/0, null = not yet evaluated
    evaluated_at = Column(DateTime, nullable=True)


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
