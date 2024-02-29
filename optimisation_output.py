import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
from stock_pick_algo import return_stocks

def return_assets_weights():
    # Section 1: Define Tickers and Time Range
    stock_info = return_stocks(max=10)

    stock_list = list(stock_info['ticker'])
    tickers = stock_list
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)

    # Section 2: Download Adjusted Close Prices
    adj_close_df = pd.concat([yf.download(ticker, start=start_date, end=end_date)['Adj Close'] for ticker in tickers], axis=1, keys=tickers)

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

    stock_info_weights_df = pd.merge(ticker_weight_df, stock_info, on='ticker')

     

    return stock_info_weights_df['ticker', 'sector', 'optimal_weights']

print(return_assets_weights().head())