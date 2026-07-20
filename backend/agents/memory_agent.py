"""
Memory Agent (item #13): persists every prediction the system makes so
the Learning Agent can later compare it to what actually happened.
This is the cross-run, cross-process "long-term memory" referenced in
base_agent.py - backed by the PredictionLog table, so it survives
restarts and works the same whether DATABASE_URL is SQLite or Postgres.
"""
from datetime import datetime

from agents.base_agent import BaseAgent
from database import PredictionLog, SessionLocal


class MemoryAgent(BaseAgent):
    name = "memory_agent"
    goal = "Persist predictions and retrieve prior predictions/outcomes for a symbol."
    output_schema = {}

    def reason(self, action: str = "record", **kwargs) -> dict:
        if action == "record":
            return self.record_prediction(**kwargs)
        if action == "history":
            return {"history": self.get_history(kwargs.get("symbol"))}
        raise ValueError(f"Unknown memory action: {action}")

    def record_prediction(self, symbol: str, price_at_prediction: float, predicted_return_pct: float,
                           predicted_direction: str, horizon_days: int = 5) -> dict:
        db = SessionLocal()
        try:
            row = PredictionLog(
                symbol=symbol, price_at_prediction=price_at_prediction,
                predicted_return_pct=predicted_return_pct, predicted_direction=predicted_direction,
                horizon_days=horizon_days, predicted_at=datetime.utcnow(),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return {"id": row.id}
        finally:
            db.close()

    def get_history(self, symbol: str, limit: int = 20) -> list[dict]:
        db = SessionLocal()
        try:
            rows = (
                db.query(PredictionLog)
                .filter(PredictionLog.symbol == symbol)
                .order_by(PredictionLog.predicted_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "predicted_at": r.predicted_at.isoformat(),
                    "predicted_return_pct": r.predicted_return_pct,
                    "predicted_direction": r.predicted_direction,
                    "actual_return_pct": r.actual_return_pct,
                    "actual_direction_correct": bool(r.actual_direction_correct) if r.actual_direction_correct is not None else None,
                }
                for r in rows
            ]
        finally:
            db.close()


memory_agent = MemoryAgent()
