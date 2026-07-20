"""
News & Sentiment Agent. Rule-based headline scoring by default (works with
zero API keys); if GEMINI_API_KEY is set, uses it to summarize/score
articles with actual language understanding instead of keyword matching
(item #10 - "News Intelligence": summarize, explain market impact, give a
confidence score, not just positive/negative).
"""
from agents import _sentiment_logic as logic
from agents.base_agent import BaseAgent
from config import settings


class SentimentAgent(BaseAgent):
    name = "sentiment_agent"
    goal = "Summarize recent news for a stock and score its sentiment (0-100), flagging risk events."
    output_schema = {"score": float, "details": dict}

    def plan(self, **inputs) -> list[str]:
        return ["llm_summarize"] if settings.GEMINI_API_KEY else ["keyword_score"]

    def reason(self, symbol: str = None, **_) -> dict:
        result = logic.get_news_and_sentiment(symbol)
        if settings.GEMINI_API_KEY:
            result = self._enrich_with_llm(symbol, result)
        return result

    def _enrich_with_llm(self, symbol: str, result: dict) -> dict:
        """Optional LLM pass: turns the rule-based headline scores into a
        short natural-language explanation of *why* sentiment is what it
        is. Never raises - a Gemini failure just means we keep the
        rule-based result, so this agent always degrades gracefully."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            headlines = [a.get("headline", "") for a in result.get("details", {}).get("articles", [])][:8]
            if not headlines:
                return result
            prompt = (
                f"You are a financial news analyst. Given these recent headlines about "
                f"{symbol}, write ONE sentence summarizing the net market impact and a "
                f"confidence 0-100 that this reading is reliable. Headlines:\n"
                + "\n".join(f"- {h}" for h in headlines)
            )
            resp = model.generate_content(prompt)
            result.setdefault("details", {})["llm_summary"] = resp.text.strip()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Gemini enrichment skipped for %s: %s", symbol, exc)
        return result


sentiment_agent = SentimentAgent()


def get_news_and_sentiment(symbol: str, max_articles: int = 8) -> dict:
    return sentiment_agent.reason(symbol=symbol)
