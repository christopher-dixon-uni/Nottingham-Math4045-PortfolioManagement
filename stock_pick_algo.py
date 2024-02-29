import yfinance as yf
import pandas as pd
import datetime as dt

def return_stocks(max=500):
    """
    selecting from the S&p500 according to the following criteria:
    - Market Cap: >$20 Billion
    - Price to Earnings ratio (PE): <20
    - Price to book ratio (PB) > 1
    - Dividend  > 2.5%

    #forwardPE
    #marketCap
    #priceToBook
    #dividendYield
    """


    today = dt.date.today().strftime('%Y-%m-%d')
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d')

    # Download the S&P 500 list
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500.to_csv('sp500.csv')
    sp500 = pd.read_csv('sp500.csv')
    sp500 = sp500['Symbol'].tolist()[max]

    stock_temp = yf.Ticker('AAPL')
    df_temp = pd.DataFrame(stock_temp.info).head(1)
    colnames = list(df_temp.columns)

    # Download the data for each stock
    data = pd.DataFrame(columns = colnames)
    for ticker in sp500:
        try:
            stock_obj = yf.Ticker(ticker)

            df = pd.DataFrame(stock_obj.info).head(1)
            df.insert(loc = 0,column='ticker', value=ticker)
            data = pd.concat([data, df.iloc[[0]]], ignore_index=True)        
        except:
            pass

    data.insert(0, 'ticker', data.pop('ticker'))

    #filter data
    filtered_data = data[(data['marketCap'] > 20000000000) & (data['forwardPE'] < 20) & (data['priceToBook'] > 1) & (data['dividendYield'] > 0.025)]

    #save filtered data df to csv
    filtered_data.to_csv('./OliverTurner_ProjectWork/Stock picking/filtered_data.csv')

    return filtered_data
