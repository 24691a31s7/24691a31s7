"""
Learning Agent (item #14): evaluates predictions once their horizon has
elapsed (predicted_at + horizon_days <= today) by fetching the actual
realized price move, then reports accuracy metrics overall / per symbol.

`evaluate_due_predictions()` is meant to run on a daily schedule (see
scheduler.py). This intentionally does NOT auto-retrain a model on every
run in this codebase (a real retraining loop needs a labelled training
pipeline + validation gate, not a cron job blindly overwriting weights) -
it computes the accuracy numbers a human (or a training script triggered
from CI) would use to decide whether/when to retrain. That keeps
"self-improving" honest instead of silently degrading the model.
"""
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from database import PredictionLog, SessionLocal
from services import data_service
from utils.logger import get_logger

log = get_logger("stocks.agent.learning_agent")


class LearningAgent(BaseAgent):
    name = "learning_agent"
    goal = "Evaluate past predictions against realized outcomes and report accuracy."
    output_schema = {}

    def reason(self, **_) -> dict:
        return self.accuracy_report()

    def evaluate_due_predictions(self) -> int:
        db = SessionLocal()
        evaluated = 0
        try:
            due = (
                db.query(PredictionLog)
                .filter(PredictionLog.actual_return_pct.is_(None))
                .all()
            )
            for row in due:
                due_date = row.predicted_at + timedelta(days=row.horizon_days)
                if due_date > datetime.utcnow():
                    continue
                try:
                    quote = data_service.get_live_quote(row.symbol)
                    current_price = quote.get("last_price")
                except Exception as exc:  # noqa: BLE001
                    log.warning("Could not fetch price to evaluate %s: %s", row.symbol, exc)
                    continue
                if not current_price or not row.price_at_prediction:
                    continue

                actual_return = (current_price - row.price_at_prediction) / row.price_at_prediction * 100
                predicted_up = row.predicted_direction == "BUY" or row.predicted_return_pct > 0
                actual_up = actual_return > 0
                row.actual_return_pct = round(actual_return, 3)
                row.actual_direction_correct = int(predicted_up == actual_up)
                row.evaluated_at = datetime.utcnow()
                evaluated += 1
            db.commit()
        finally:
            db.close()
        return evaluated

    def accuracy_report(self) -> dict:
        db = SessionLocal()
        try:
            evaluated = db.query(PredictionLog).filter(PredictionLog.actual_return_pct.isnot(None)).all()
            if not evaluated:
                return {"evaluated_count": 0, "overall_accuracy_pct": None, "per_symbol": {}}

            correct = sum(r.actual_direction_correct for r in evaluated)
            overall = round(correct / len(evaluated) * 100, 1)

            per_symbol: dict[str, dict] = {}
            for r in evaluated:
                s = per_symbol.setdefault(r.symbol, {"count": 0, "correct": 0})
                s["count"] += 1
                s["correct"] += r.actual_direction_correct
            for s, v in per_symbol.items():
                v["accuracy_pct"] = round(v["correct"] / v["count"] * 100, 1)

            return {"evaluated_count": len(evaluated), "overall_accuracy_pct": overall, "per_symbol": per_symbol}
        finally:
            db.close()


learning_agent = LearningAgent()
