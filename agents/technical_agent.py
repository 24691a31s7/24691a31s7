from ta.momentum import RSIIndicator

def calculate_rsi(df):

    rsi = RSIIndicator(df["Close"])

    return round(rsi.rsi().iloc[-1], 2)