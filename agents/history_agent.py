import yfinance as yf

def get_history(symbol):

    stock = yf.Ticker(symbol)

    df = stock.history(period="5y")

    return df