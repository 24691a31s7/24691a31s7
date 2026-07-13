from agents.market_agent import get_stock_data
from agents.history_agent import get_history
from agents.technical_agent import calculate_rsi
from agents.recommendation_agent import recommend
from agents.news_agent import get_stock_news
from agents.sentiment_agent import analyze_sentiment

def run_analysis(symbol, company):

    market = get_stock_data(symbol)

    history = get_history(symbol)

    rsi = calculate_rsi(history)

    headlines = get_stock_news(company)

    sentiment = analyze_sentiment(headlines)

    recommendation = recommend(
        rsi,
        sentiment["score"]
    )

    return {
        "market": market,
        "rsi": rsi,
        "sentiment": sentiment,
        "headlines": headlines[:5],
        "recommendation": recommendation
    }