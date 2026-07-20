"""
Recommendation Agent: combines every upstream signal into a final
BUY / SELL / HOLD / WATCHLIST decision, plus fixed-fractional position
sizing. Plain, auditable arithmetic parameterized by the user's own
capital/risk inputs - not personalized advice, and it never places an
order (alerts only). See README "Disclaimer".
"""
from agents import _recommendation_logic as logic
from agents.base_agent import BaseAgent


class RecommendationAgent(BaseAgent):
    name = "recommendation_agent"
    goal = "Turn upstream agent signals into a BUY/SELL/HOLD/WATCHLIST call and a position size."
    output_schema = {}

    def plan(self, **inputs) -> list[str]:
        if inputs.get("conflict_detected"):
            return ["flag_watchlist"]
        return ["score_and_decide"]

    def reason(self, expected_return_pct=None, profit_probability_pct=None, risk_label=None,
               confidence_pct=None, conflict_detected=None, **_) -> dict:
        decision = logic.decide(expected_return_pct, profit_probability_pct, risk_label,
                                 confidence_pct, conflict_detected)
        return {"decision": decision}

    def size(self, entry_price, risk_label, capital=None, risk_per_trade_pct=None) -> dict:
        return logic.size_position(entry_price, risk_label, capital, risk_per_trade_pct)


recommendation_agent = RecommendationAgent()


def decide(expected_return_pct, profit_probability_pct, risk_label, confidence_pct, conflict_detected) -> str:
    return recommendation_agent.reason(
        expected_return_pct=expected_return_pct, profit_probability_pct=profit_probability_pct,
        risk_label=risk_label, confidence_pct=confidence_pct, conflict_detected=conflict_detected,
    )["decision"]


def size_position(entry_price, risk_label, capital=None, risk_per_trade_pct=None) -> dict:
    return recommendation_agent.size(entry_price, risk_label, capital, risk_per_trade_pct)
