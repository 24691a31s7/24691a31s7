"""Risk Agent: volatility + liquidity + event risk -> Low/Medium/High label."""
import pandas as pd

from agents import _risk_logic as logic
from agents.base_agent import BaseAgent


class RiskAgent(BaseAgent):
    name = "risk_agent"
    goal = "Classify a stock's risk (Low/Medium/High) from volatility, liquidity, and flagged news events."
    output_schema = {"risk_label": str, "details": dict}

    def reason(self, price_history: pd.DataFrame = None, flagged_risk_events: list = None, **_) -> dict:
        return logic.assess(price_history, flagged_risk_events or [])


risk_agent = RiskAgent()


def assess(price_history: pd.DataFrame, flagged_risk_events: list) -> dict:
    return risk_agent.reason(price_history=price_history, flagged_risk_events=flagged_risk_events)
