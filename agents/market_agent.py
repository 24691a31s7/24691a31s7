import yfinance as yf

def get_stock_data(symbol):

    stock = yf.Ticker(symbol)

    info = stock.info

    return {
        "symbol": symbol,
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "volume": info.get("volume"),
        "high_52w": info.get("fiftyTwoWeekHigh"),
        "low_52w": info.get("fiftyTwoWeekLow")
    }