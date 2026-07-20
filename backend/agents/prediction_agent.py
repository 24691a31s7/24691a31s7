"""
Prediction Agent: expected-return / profit-probability forecast.

v1 stays an explainable weighted-score model (see _prediction_logic.py) so
every number is traceable to a reason, matching the "Explainability Agent"
requirement (item #12). A trained gradient-boosted model (item #8) is a
drop-in replacement: point PREDICTION_MODEL_PATH at a saved model file
implementing `.predict_feature_vector(dict) -> dict` with the same output
shape, and this agent will use it instead - see agents/ml/train_model.py
for the training script scaffold. Without a trained model (the default,
since training needs your own historical dataset) it uses the weighted
formula, which remains fully functional.
"""
import os

import pandas as pd

from agents import _prediction_logic as logic
from agents.base_agent import BaseAgent


class PredictionAgent(BaseAgent):
    name = "prediction_agent"
    goal = "Forecast expected return %, a confidence range, and profit/loss probability for a stock."
    output_schema = {"expected_return_pct": float, "profit_probability_pct": float, "loss_probability_pct": float}

    def __init__(self):
        super().__init__()
        self._ml_model = self._try_load_ml_model()

    def _try_load_ml_model(self):
        path = os.getenv("PREDICTION_MODEL_PATH", "")
        if not path or not os.path.exists(path):
            return None
        try:
            import joblib

            model = joblib.load(path)
            self.log.info("Loaded trained prediction model from %s", path)
            return model
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Could not load PREDICTION_MODEL_PATH (%s): %s", path, exc)
            return None

    def plan(self, **inputs) -> list[str]:
        return ["ml_predict"] if self._ml_model is not None else ["weighted_formula"]

    def reason(self, technical_score: float = 0, fundamental_score: float = 0, sentiment_score: float = 0,
               price_history: pd.DataFrame = None, feature_vector: dict = None, pattern_score: float = 0.0,
               **_) -> dict:
        if self._ml_model is not None and feature_vector is not None:
            try:
                return self._ml_model.predict_feature_vector(feature_vector)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("ML model prediction failed, falling back to formula: %s", exc)
        return logic.forecast_return(technical_score, fundamental_score, sentiment_score, price_history,
                                      pattern_score=pattern_score)


prediction_agent = PredictionAgent()


def forecast_return(technical_score, fundamental_score, sentiment_score, price_history, pattern_score=0.0) -> dict:
    return prediction_agent.reason(
        technical_score=technical_score, fundamental_score=fundamental_score,
        sentiment_score=sentiment_score, price_history=price_history, pattern_score=pattern_score,
    )
