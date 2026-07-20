"""Fundamental Agent: P/E, P/B, ROE, debt/equity -> a 0-100 fundamental score."""
from agents import _fundamental_logic as logic
from agents.base_agent import BaseAgent


class FundamentalAgent(BaseAgent):
    name = "fundamental_agent"
    goal = "Score a stock's fundamentals (0-100) from valuation and quality ratios."
    output_schema = {"score": float, "details": dict}

    def reason(self, symbol: str = None, **_) -> dict:
        return logic.analyze(symbol)


fundamental_agent = FundamentalAgent()


def analyze(symbol: str) -> dict:
    return fundamental_agent.reason(symbol=symbol)
