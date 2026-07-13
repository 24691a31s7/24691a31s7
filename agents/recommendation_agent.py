def recommend(rsi, sentiment_score):

    if rsi < 35 and sentiment_score > 60:
        return "BUY"

    elif rsi > 70 and sentiment_score < 40:
        return "SELL"

    else:
        return "HOLD"