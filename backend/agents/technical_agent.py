"""
Technical Agent: RSI, MACD, EMA crossovers, Bollinger Bands -> a 0-100
technical strength score. Deterministic math -> stays deterministic
(no LLM in the hot path; see base_agent.py docstring for why).
"""
import pandas as pd

from agents import _technical_logic as logic
from agents.base_agent import BaseAgent


class TechnicalAgent(BaseAgent):
    name = "technical_agent"
    goal = "Score a stock's technical setup (0-100) from RSI/MACD/EMA/Bollinger signals."
    output_schema = {"score": float, "details": dict}
    tools = [logic._rsi, logic._macd, logic._ema, logic._bollinger]

    def reason(self, price_history: pd.DataFrame = None, **_) -> dict:
        return logic.analyze(price_history)


technical_agent = TechnicalAgent()


def analyze(df: pd.DataFrame) -> dict:
    """Simple functional entrypoint used by the orchestrator's hot path."""
    return technical_agent.reason(price_history=df)
