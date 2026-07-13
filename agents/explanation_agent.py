import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def generate_explanation(data):

    prompt = f"""
    Analyze stock recommendation.

    Stock: {data['symbol']}

    Price: {data['price']}

    RSI: {data['rsi']}

    Sentiment:
    {data['sentiment']}

    Recommendation:
    {data['recommendation']}

    Explain in simple investor language.
    """

    response = model.generate_content(prompt)

    return response.text