import yfinance as yf
import pandas as pd
import datetime as dt
import json
import os

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
    yesterday = (dt.date.today() + dt.timedelta(days=-1)).strftime('%Y-%m-%d')

    #check if data has been updated today
    try:
        with open('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/stock_info_last_updated.json', 'r') as f:
            last_updated = json.load(f)
            if last_updated['last_updated'] == today:
                print('Data already updated today, returning...')
                updated = True
            else:
                updated = False
    except:
        updated = False
        #if no json exists, create one
        with open('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/stock_info_last_updated.json', 'w') as f:
            json.dump({'last_updated': today}, f)
            print('Created last_updated.json')
            
    if updated == False:
        print('Data not updated today, updating...')
        print('Updating may take some time.')
    
        # Download the S&P 500 list
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500.to_csv('sp500.csv')
        sp500 = pd.read_csv('sp500.csv')
        sp500 = sp500['Symbol'].tolist()[0:max]

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
        filtered_data = data[(data['marketCap'] > 2e10) & (data['forwardPE'] < 20) & (data['priceToBook'] > 1) & (data['dividendYield'] > 0.025)]
            
        #save filtered data df to csv
        filtered_data.to_csv('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/stock_info.csv')

        #update last_updated.json
        with open('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/stock_info_last_updated.json', 'w') as f:
            json.dump({'last_updated': today}, f)
        
        print('Data updated and saved to stock_info.csv')
    
    
    filtered_data = pd.read_csv('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/stock_info.csv')

    return filtered_data

if __name__ == "__main__":
    print(list(return_stocks(max=500)['ticker']))