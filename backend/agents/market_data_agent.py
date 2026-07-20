"""
Market Data Agent: real-time + historical data collection.
Thin, typed wrapper over services/data_service.py, which owns all caching
and external API calls (one data service, many agents).
"""
import pandas as pd

from agents.base_agent import BaseAgent
from services import data_service


class MarketDataAgent(BaseAgent):
    name = "market_data_agent"
    goal = "Provide live quotes and historical OHLCV data for a symbol, cached to minimize upstream API load."
    output_schema = {"last_price": float, "day_change_perc": float}
    tools = [data_service.get_live_quote, data_service.get_price_history, data_service.get_live_prices_bulk]

    def reason(self, symbol: str = None, **_) -> dict:
        return data_service.get_live_quote(symbol)

    # Convenience passthroughs used directly by the orchestrator/services
    def get_live_quote(self, symbol: str) -> dict:
        return data_service.get_live_quote(symbol)

    def get_live_prices_bulk(self, symbols: list[str]) -> dict:
        return data_service.get_live_prices_bulk(symbols)

    def get_price_history(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        return data_service.get_price_history(symbol, period, interval)


market_data_agent = MarketDataAgent()

# module-level function passthroughs so existing call sites keep working
get_live_quote = market_data_agent.get_live_quote
get_live_prices_bulk = market_data_agent.get_live_prices_bulk
get_price_history = market_data_agent.get_price_history
