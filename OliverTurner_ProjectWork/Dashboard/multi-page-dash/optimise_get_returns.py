import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
import json
import os
import pickle


'''


run functions as follows
investible_universe = return_stocks()

expected_returns = get_expected_returns(investible_universe, start_date, end_date)

weights = get_weights(expected_returns, mu)

returns = calculate_returns(weights, start_date, end_date)


'''


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
        sp500 = sorted(sp500['Symbol'].tolist()[0:max])

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



#get expected returns
def get_expected_returns(end_date):
    """
    ticker: list of stock tickers
    start_date: start date for historical data
    end_date: end date for historical data
    mu: expected return for each stock

    returns: dataframe of weights for each stock
    """
    #load predicted_returns.json
    try:
        with open('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/predicted_returns.json', 'r') as f:
            predicted_returns = json.load(f)

    except:
        print('predicted_returns.json not found')

    expected_returns = predicted_returns[end_date]

    expected_returns = dict(sorted(expected_returns.items()))

    #create dataframe of expected returns
    return expected_returns



def get_weights(expected_returns, mu):
    
    tickers = sorted(list(expected_returns.keys()))

    # get historical returns of each stock

    temp_end_date = pd.to_datetime('today')
    temp_start_date = temp_end_date - pd.DateOffset(years=1)

    #download dailyReturns as csv 
    ##check if file exists
    if os.path.exists('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/dailyReturns.pkl'):
        dailyReturns = pd.read_pickle('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/dailyReturns.pkl')


        # Depending on the structure of your CSV, you may need to adjust this.
    else:
        dailyReturns = yf.download(sorted(tickers), start=temp_start_date, end=temp_end_date)
        dailyReturns.to_pickle('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/dailyReturns.pkl')



        

    returns = dailyReturns['Adj Close'].pct_change().dropna()

    # Compute covariance matrix
    V = returns.cov()
    invV = np.linalg.inv(V)

    n_assets = invV.shape[0]

    ones = np.ones((n_assets, 1))

    alpha = np.dot(np.dot(ones.T, invV), ones).item()


    beta = np.dot(np.dot(ones.T, invV), np.array(list(expected_returns.values())).T).item()

    gamma = np.dot(np.dot(np.array(list(expected_returns.values())), invV), np.array(list(expected_returns.values())).T).item()

    delta = alpha*gamma - beta**2

    lambda1 = (gamma - beta*mu)/delta

    lambda2 = (alpha*mu - beta)/delta

    epsilonG = (1/alpha)*np.dot(ones.T, invV).flatten()

    epsilonD = (1/beta)*np.dot(invV, np.array(list(expected_returns.values()))).flatten()


    epsilonMVE = lambda1*alpha*epsilonG + lambda2*beta*epsilonD

    weights = {ticker: weight for ticker, weight in zip(tickers, epsilonMVE)}


    return weights, V, invV, alpha, beta, gamma, delta, lambda1, lambda2, epsilonG, epsilonD, epsilonMVE



def calculate_returns(asset_weights, start_date, end_date):
    #calculate returns from start date to end date using weights from
    #get_expected_returns

    weights = list(asset_weights.values())
    tickers = sorted(list(asset_weights.keys()))
    #get historical returns of each stock
    df = yf.download(sorted(list(asset_weights.keys())), start=start_date, end=end_date)

    #flatten headers
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    daily_returns_list = []

    for ticker in tickers:
        ticker_daily_returns = df[f'Adj Close_{ticker}'].pct_change().dropna()
        daily_returns_list.append(ticker_daily_returns.rename(f'Daily Returns_{ticker}'))

    # Concatenate all the daily returns into a single DataFrame
    daily_returns_df = pd.concat(daily_returns_list, axis=1)


    df=df.copy()
    df = pd.concat([df, daily_returns_df], axis=1)

    portfolio_daily_returns = [weights[i]*df[f'Daily Returns_{tickers[i]}'] for i in range(len(tickers))]
    df['Portfolio Daily Returns'] = sum(portfolio_daily_returns)
    

    #Portfolio cumulative returns
    #df['Portfolio Cumulative Returns'] = (1 + df['Portfolio Daily Returns']).cumprod() - 1

    return df



def plot_returns(returns_df):

    #plot cumulative returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns_df['Date'], y=returns_df['Portfolio Cumulative Returns'], mode='lines', name='Portfolio Cumulative Returns'))
    fig.update_layout(title='Portfolio Cumulative Returns',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Returns',
                      template="plotly_dark",
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      font_color="white")
    return fig


def get_2023_returns(mu):
    #quarterly start and end dates 2023
    '''
    start_dates = ['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01']
    end_dates = ['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31']

    #loop through each quarter and get expected returns
    for end_date in end_dates:
        expected_returns = get_expected_returns(end_date)

        #get weights
        weights, V, InvV, alpha, beta, gamma, delta, lambda1, lambda2, epsilonG, epsilonD, epsilonMVE = get_weights(expected_returns, mu)

        #calculate returns
        returns = calculate_returns(weights, start_dates[end_dates.index(end_date)], end_date)

        #join returns to one dataframe
        if end_date == end_dates[0]:
            all_returns = returns
        else:
            all_returns = pd.concat([all_returns, returns])
    
    #save all_returns to csv
    all_returns.to_csv('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/2023_returns.csv')
    return all_returns, weights, V, InvV, alpha, beta, gamma, delta, lambda1, lambda2, epsilonG, epsilonD, epsilonMVE
'''
    # Quarterly start and end dates for 2023
    start_dates = ['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01']
    end_dates = ['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31']

    all_returns = pd.DataFrame()  # Initialize an empty dataframe

    # Loop through each quarter to calculate returns
    for start_date, end_date in zip(start_dates, end_dates):
        expected_returns = get_expected_returns(end_date)
        weights, V, InvV, alpha, beta, gamma, delta, lambda1, lambda2, epsilonG, epsilonD, epsilonMVE = get_weights(expected_returns, mu)
        quarter_returns = calculate_returns(weights, start_date, end_date)
        
        # Concatenate the quarter's returns with the cumulative dataframe
        all_returns = pd.concat([all_returns, quarter_returns])

    # Ensure the index is sorted (if it's a datetime index)
    all_returns.sort_index(inplace=True)

    all_returns['Portfolio Daily Returns'] = all_returns['Portfolio Daily Returns'].fillna(0)


    # Calculate cumulative returns on the concatenated dataframe
    all_returns['Portfolio Cumulative Returns'] = (1 + all_returns['Portfolio Daily Returns']).cumprod() - 1

    # Save all_returns to CSV
    all_returns.to_csv('OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/2023_returns.csv')

    return all_returns, weights, V, InvV, alpha, beta, gamma, delta, lambda1, lambda2, epsilonG, epsilonD, epsilonMVE

def plot_efficient_frontier():

    expected_returns = get_expected_returns('2023-12-31')
    
    (weights, V, invV, alpha, beta, gamma, delta, lambda1,
     lambda2, epsilonG, epsilonD,
     epsilonMVE)= get_weights(expected_returns, mu=0.10)


    mu_range = np.linspace(-0.3, 0.3, 200)
    
    

    ones = np.ones((len(list(expected_returns.keys())), 1))

    # Arrays to hold standard deviations and expected returns
    std_devs = []
    exp_returns = []
    
    

    for mu in mu_range:
        lambda1 = (gamma - beta*mu) / delta
        lambda2 = (alpha*mu - beta) / delta
        epsilonG = (1 / alpha) * np.dot(ones.T, invV).flatten()
        epsilonD = (1 / beta) * np.dot(invV, np.array(list(expected_returns.values()))).flatten()
        epsilonMVE = lambda1*alpha*epsilonG + lambda2*beta*epsilonD    
        portfolio_variance = np.dot(np.dot(epsilonMVE, V), epsilonMVE.T)
        portfolio_std_dev = np.sqrt(portfolio_variance)
        std_devs.append(portfolio_std_dev)
        exp_returns.append(mu)


    # Plotting the efficient frontier
    efficient_frontier_fig = go.Figure()
    efficient_frontier_fig.add_trace(go.Scatter(x=std_devs, y=exp_returns, mode='lines', name='Efficient Frontier'))

    efficient_frontier_fig.update_layout(title='Efficient Frontier',
                    xaxis_title='Standard Deviation (Risk)',
                    yaxis_title='Expected Return',
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color="white",
                    showlegend=True)
    
    return efficient_frontier_fig


def get_recent_weights_sectors():

    weights, V, invV, alpha, beta, gamma, delta, lambda1, lambda2, epsilonG, epsilonD, epsilonMVE = get_weights(get_expected_returns('2023-12-31'), 0.10)

    ticker_info = {}
    for ticker in weights.keys():
        stock = yf.Ticker(ticker)
        info = stock.info
        # Attempt to fetch the sector and longName, default to 'Unknown' if not found
        ticker_info[ticker] = {
            'sector': info.get('sector', 'Unknown'),
            'longName': info.get('longName', 'Unknown')
        }

    # Combine the weights, sector, and longName information
    combined_data = []
    for ticker, weight in weights.items():
        info = ticker_info[ticker]
        combined_data.append({
            'Ticker': ticker,
            'Weight': weight,
            'Sector': info['sector'],
            'Long Name': info['longName']
        })

    # Convert combined data to a DataFrame
    df_combined = pd.DataFrame(combined_data)

    return df_combined

        

'''    #get expected returns
    expected_returns = get_expected_returns('2023-01-01', '2024-01-01')

    #get weights
    weights = get_weights(expected_returns, 0.1)

    #calculate returns
    returns = calculate_returns(weights, '2023-01-01', '2024-01-01')

    #plot returns
    plot_returns(returns)'''




if __name__ == "__main__":
    None