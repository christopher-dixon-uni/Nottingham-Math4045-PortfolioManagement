import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
from stock_pick_algo import return_stocks
import datetime as dt
import json

today = dt.date.today().strftime('%Y-%m-%d')


def return_assets_weights():
    # Section 1: Define Tickers and Time Range
    stock_info = return_stocks(max=500)

    stock_list = list(stock_info['ticker'])
    tickers = stock_list
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)

    # Section 2: Download Adjusted Close Prices
        #check if data has been updated today
    try:
        with open('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/adj_close_df.json', 'r') as f:
            last_updated = json.load(f)
            if last_updated['last_updated'] == today:
                print('Data already updated today, returning...')
                updated = True
            else:
                updated = False
    except:
        updated = False
        #if no json exists, create one
        with open('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/adj_close_df.json', 'w') as f:
            json.dump({'last_updated': today}, f)
            print('Created last_updated.json')
        
    if updated == False:
        print('Data not updated today, updating...')
        print('Updating may take some time.')
        adj_close_df = pd.concat([yf.download(ticker, start=start_date, end=end_date)['Adj Close'] for ticker in tickers], axis=1, keys=tickers)
        #save to csv
        adj_close_df.to_csv('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/adj_close_df.csv')

    adj_close_df = pd.read_csv('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/adj_close_df.csv', index_col=0)
    

    # Section 3: Calculate Lognormal Returns
    log_returns = np.log(adj_close_df / adj_close_df.shift()).dropna()

    # Section 4: Calculate Covariance Matrix
    cov_matrix = log_returns.cov() * 252

    # Section 5: Define Portfolio Performance Metrics
    def standard_deviation(weights): return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    def expected_return(weights): return np.sum(log_returns.mean() * weights) * 252
    def sharpe_ratio(weights, risk_free_rate=0.02): return (expected_return(weights) - risk_free_rate) / standard_deviation(weights)

    # Section 6: Portfolio Optimization with Minimum Weight Constraint
    bounds = [(0.0025, 0.05)] * len(tickers)
    initial_weights = np.full(len(tickers), 1/len(tickers))
    optimized_results = minimize(lambda weights: -sharpe_ratio(weights), initial_weights, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Section 7: Analyze the Optimal Portfolio
    optimal_weights = optimized_results.x

    
    ticker_weight_df = pd.DataFrame({'ticker': tickers, 'optimal_weights': optimal_weights})

    stock_info_weights_df = stock_info.merge(ticker_weight_df, on='ticker')

     

    return stock_info_weights_df[['ticker', 'longName', 'sector', 'optimal_weights']]

if __name__ == "__main__":
    print(return_assets_weights())