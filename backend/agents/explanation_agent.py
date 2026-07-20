"""Explainability Agent (item #12): turns structured upstream outputs into
a human-readable reason list AND a structured breakdown (confidence,
technical/fundamental strength, stop loss, target, probability) so the
frontend can render a professional "BUY - 88% confidence, here's why" card
instead of a bare label."""
from agents import _explanation_logic as logic
from agents.base_agent import BaseAgent


class ExplanationAgent(BaseAgent):
    name = "explanation_agent"
    goal = "Explain a recommendation in plain English, citing the specific signals that drove it."
    output_schema = {}

    def reason(self, decision: str = None, technical: dict = None, fundamental: dict = None,
               sentiment: dict = None, risk: dict = None, prediction: dict = None, pattern: dict = None,
               **_) -> dict:
        reasons = logic.explain(decision, technical, fundamental, sentiment, risk, prediction, pattern=pattern)
        return {"reasons": reasons, "headline": logic.headline(decision, "")}


explanation_agent = ExplanationAgent()


def explain(recommendation, technical, fundamental, sentiment, risk, prediction, pattern=None) -> list[str]:
    return logic.explain(recommendation, technical, fundamental, sentiment, risk, prediction, pattern=pattern)


def headline(recommendation: str, symbol: str) -> str:
    return logic.headline(recommendation, symbol)
