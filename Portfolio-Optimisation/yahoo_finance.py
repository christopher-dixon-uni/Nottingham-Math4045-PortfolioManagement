import yfinance as yf

def get_historical_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1y")
    return data

get_historical_data('AAPL')