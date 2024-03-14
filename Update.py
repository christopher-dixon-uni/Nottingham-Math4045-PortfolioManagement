
import yfinance as yf
import pandas as pd

def return_stocks(max_stocks=500):
    # Fetch the current list of S&P 500 companies
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500_symbols = sp500['Symbol'].tolist()[:max_stocks]

    data = pd.DataFrame()

    for ticker in sp500_symbols:
        try:
            stock_obj = yf.Ticker(ticker)
            stock_info = stock_obj.info

            df = pd.DataFrame([{
                'ticker': ticker,
                'marketCap': stock_info.get('marketCap'),
                'forwardPE': stock_info.get('forwardPE'),
                'priceToBook': stock_info.get('priceToBook'),
                'dividendYield': stock_info.get('dividendYield') if stock_info.get('dividendYield') is not None else stock_info.get('dividendRate') / stock_info.get('previousClose'),
                'roe': stock_info.get('returnOnEquity') * 100 if stock_info.get('returnOnEquity') is not None else None,  # Convert to percentage if needed
                'earningsGrowth': stock_info.get('earningsGrowth')
            }])

            data = pd.concat([data, df], ignore_index=True)
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

    # Filter data based on your criteria
    filtered_data = data[
        (data['marketCap'] > 2e10) &
        (data['forwardPE'] < 20) &
        (data['priceToBook'] < 1.5) &
        (data['dividendYield'] > 0.03) &
        (data['roe'] > 15) &
        (data['earningsGrowth'] > 0)
    ]

    return 


filtered_stocks = return_stocks(max_stocks=500)
print(filtered_stocks)
