"""
News Agent + Sentiment Analysis. Uses services/data_service for the
actual NewsAPI call (cached 30 min). Sentiment scoring is a transparent
lexicon-based scorer - swap in FinBERT/Gemini for a real NLP classifier
by replacing `_score_headline`.
"""
from schemas import AgentResult
from services import data_service
from stock_universe import SYMBOL_TO_NAME

POSITIVE_WORDS = {
    "beats", "surge", "rally", "growth", "profit", "upgrade", "record",
    "wins", "expansion", "strong", "outperform", "buy", "bullish", "jump",
    "gains", "boost", "positive", "raises", "approval", "partnership",
}
NEGATIVE_WORDS = {
    "fraud", "probe", "lawsuit", "downgrade", "miss", "slump", "plunge",
    "crash", "loss", "resign", "penalty", "scam", "fine", "layoff",
    "negative", "sell-off", "bearish", "default", "investigation", "ban",
}


def _score_headline(headline: str) -> int:
    text = headline.lower()
    score = sum(1 for w in POSITIVE_WORDS if w in text)
    score -= sum(2 for w in NEGATIVE_WORDS if w in text)
    return score


def get_news_and_sentiment(symbol: str, max_articles: int = 8) -> dict:
    company_name = SYMBOL_TO_NAME.get(symbol, symbol)
    articles = data_service.get_news_articles(company_name, max_articles)

    if not articles:
        return AgentResult(
            agent="sentiment", score=0.0, confidence=15.0,
            reason="No recent news found - neutral default",
            details={"headlines": [], "flagged_risk_events": []},
        ).to_dict()

    headlines = [a["title"] for a in articles if a.get("title")]
    raw_scores = [_score_headline(h) for h in headlines]
    flagged = [h for h, s in zip(headlines, raw_scores) if s < 0]

    avg = sum(raw_scores) / len(raw_scores) if raw_scores else 0
    norm = max(-1.0, min(1.0, avg / 3))

    reason = "Positive news sentiment" if norm > 0.15 else ("Negative news sentiment" if norm < -0.15 else "Neutral news sentiment")
    if flagged:
        reason = f"Flagged risk headline: {flagged[0]}"

    # Confidence scales with how many headlines we actually found (more signal = more confidence)
    confidence = round(min(90.0, 30 + len(headlines) * 8), 1)

    return AgentResult(
        agent="sentiment",
        score=round(norm, 3),
        confidence=confidence,
        reason=reason,
        details={"headlines": headlines[:5], "flagged_risk_events": flagged[:3]},
    ).to_dict()
