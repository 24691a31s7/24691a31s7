from agents.orchestrator import run_analysis

symbol = input("Stock Symbol: ")
company = input("Company Name: ")

result = run_analysis(
    symbol,
    company
)

print("\n===== AlphaFlow AI =====")

print("Price:",
      result["market"]["current_price"])

print("RSI:",
      result["rsi"])

print("Sentiment:",
      result["sentiment"])

print("Recommendation:",
      result["recommendation"])

print("\nTop News:")

for news in result["headlines"]:
    print("-", news)