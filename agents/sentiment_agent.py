from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis"
)

def analyze_sentiment(headlines):

    if not headlines:
        return {
            "sentiment":"neutral",
            "score":50
        }

    scores = []

    for headline in headlines:

        result = sentiment_model(headline)

        label = result[0]["label"]

        confidence = result[0]["score"]

        if label == "POSITIVE":
            scores.append(confidence)

        else:
            scores.append(-confidence)

    avg = sum(scores) / len(scores)

    if avg > 0:
        sentiment = "positive"
    else:
        sentiment = "negative"

    return {
        "sentiment": sentiment,
        "score": round(abs(avg)*100,2)
    }