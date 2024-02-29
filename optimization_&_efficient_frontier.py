import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
from stock_pick_algo import return_stocks

# Section 1: Define Tickers and Time Range
stock_list = return_stocks()
stock_list = list(stock_list['ticker'])
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
print("Optimal Weights:", dict(zip(tickers, optimal_weights)))
print(f"Expected Annual Return: {expected_return(optimal_weights):.4f}")
print(f"Expected Volatility: {standard_deviation(optimal_weights):.4f}")
print(f"Sharpe Ratio: {sharpe_ratio(optimal_weights):.4f}")

# Display the Final Portfolio in a Plot
plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights, color='blue')
plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')
plt.xticks(rotation=45)
plt.show()

# Section 8: Simulate Random Portfolios
num_portfolios = 10000
all_weights = np.zeros((num_portfolios, len(tickers)))
ret_arr, vol_arr, sharpe_arr = np.zeros(num_portfolios), np.zeros(num_portfolios), np.zeros(num_portfolios)

for ind in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    all_weights[ind] = weights
    ret_arr[ind], vol_arr[ind] = expected_return(weights), standard_deviation(weights)
    sharpe_arr[ind] = sharpe_ratio(weights)

# Plotting the Efficient Frontier
plt.figure(figsize=(12, 8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier with Optimal Portfolio')
plt.scatter(vol_arr[sharpe_arr.argmax()], ret_arr[sharpe_arr.argmax()], color='r', s=50, edgecolors='black') # Optimal Portfolio
plt.plot([0, vol_arr[sharpe_arr.argmax()]], [0.02, ret_arr[sharpe_arr.argmax()]], 'k--', linewidth=2, label='Capital Market Line')
plt.legend()
plt.show()
