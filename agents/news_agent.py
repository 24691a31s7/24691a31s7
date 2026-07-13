from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

load_dotenv()

newsapi = NewsApiClient(
    api_key=os.getenv("NEWS_API_KEY")
)

def get_stock_news(company):

    news = newsapi.get_everything(
        q=company,
        language="en",
        sort_by="publishedAt",
        page_size=10
    )

    headlines = []

    for article in news["articles"]:
        headlines.append(article["title"])

    return headlines