import dash
from dash import dcc, html
import plotly.graph_objs as go
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Dash app initialization
app = dash.Dash(__name__)

# Data preparation and optimization (adapted from your code)
tickers = ['MMM', 'ABBV', 'APD', 'AEP', 'AMGN', 'ADM', 'T', 'BKR', 'BAC', 'BK', 'BAX', 'BMY', 'CVX', 'CSCO', 'KO', 'CMCSA', 'ED', 'GLW', 'CTRA', 'CVS', 'DRI', 'DVN', 'FANG', 'D', 'DOW', 'DTE', 'DUK', 'EIX', 'ETR', 'EOG', 'ES', 'EXC', 'XOM', 'FIS', 'FITB', 'FE', 'F', 'GIS', 'GPC', 'GILD', 'GS', 'HSY', 'HPE', 'IBM', 'JNJ', 'KVUE', 'KDP', 'KMB', 'KMI', 'LMT', 'LYB', 'MDT', 'MET', 'MS', 'NEM', 'NEE', 'OKE', 'PEP', 'PFE', 'PSX', 'PXD', 'PNC', 'PRU', 'PEG', 'RTX', 'SRE', 'SO', 'STT', 'TROW', 'TGT', 'USB', 'UPS', 'VLO', 'VZ', 'VICI', 'WEC', 'WFC', 'WMB', 'XEL']
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10 + 2 * 365)  # Adding an extra 2 years buffer
adj_close_df = pd.concat([yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close'] for ticker in tickers], axis=1, keys=tickers)
log_returns = np.log(adj_close_df / adj_close_df.shift()).dropna()
cov_matrix = log_returns.cov() * 252

# Portfolio Optimization adapted for Dash (your optimization logic)
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
#print("Optimal Weights:", dict(zip(tickers, optimal_weights)))
#print(f"Expected Annual Return: {expected_return(optimal_weights):.4f}")
#print(f"Expected Volatility: {standard_deviation(optimal_weights):.4f}")
#print(f"Sharpe Ratio: {sharpe_ratio(optimal_weights):.4f}")

# Efficient Frontier Plot (converted to Plotly)
num_portfolios = 10000
all_weights = np.zeros((num_portfolios, len(tickers)))
ret_arr, vol_arr, sharpe_arr = np.zeros(num_portfolios), np.zeros(num_portfolios), np.zeros(num_portfolios)

for ind in range(num_portfolios):
   weights = np.random.random(len(tickers))
   weights /= np.sum(weights)
   all_weights[ind] = weights
   ret_arr[ind] = expected_return(weights)
   vol_arr[ind] = standard_deviation(weights)
   sharpe_arr[ind] = sharpe_ratio(weights)


# Plot data for the efficient frontier
efficient_frontier = go.Scatter(
   x=vol_arr, y=ret_arr,
   mode='markers',
   marker=dict(size=5, color=sharpe_arr, colorscale='Viridis', showscale=True),
   name='Efficient Frontier'
)
efficient_frontier_fig = go.Figure(data=[efficient_frontier])


efficient_frontier_fig.update_layout(xaxis={'title': 'Volatility'},
                                 yaxis={'title': 'Return'},
                                 title='Efficient Frontier',
                                 hovermode='closest',
                                 template="plotly_dark",
                                 plot_bgcolor='rgba(0,0,0,0)',
                                 paper_bgcolor='rgba(0,0,0,0)',
                                 font_color="white",)


# Assume the following are the outputs from your optimization
# These will be used in the simulation
optimized_annual_return = expected_return(optimal_weights)
optimized_volatility = standard_deviation(optimal_weights)

# Function to simulate quarterly rebalanced portfolio returns
def simulate_rebalanced_portfolio( adj_close_df, optimized_annual_return, optimized_volatility, rebalance_frequency='Q'):
   daily_returns = np.log(adj_close_df / adj_close_df.shift())
   rebalance_dates = adj_close_df.resample(rebalance_frequency).last().index
   
   simulated_portfolio_values = [10_000_000]  # Starting portfolio value at $10 million
   current_value = simulated_portfolio_values[0]
   
   for i in range(1, len(daily_returns)):
       date = daily_returns.index[i]
       
       # Simulate daily return based on annualized optimized return and volatility
       daily_simulated_return = np.random.normal(optimized_annual_return / 252, optimized_volatility / np.sqrt(252))
       
       # Update portfolio value
       current_value *= (1 + daily_simulated_return)
       simulated_portfolio_values.append(current_value)
       
       # Quarterly rebalancing (adjust portfolio value back to initial if it's a rebalance date)
       if date in rebalance_dates:
           # For simplicity, this assumes taking profits ever quarter
           current_value = simulated_portfolio_values[0]  
       
   return pd.DataFrame(simulated_portfolio_values, index=daily_returns.index, columns=['Portfolio Value'])

# Simulate the rebalanced portfolio
cumulative_capital = simulate_rebalanced_portfolio(adj_close_df, optimized_annual_return, optimized_volatility)

# Existing function to plot stock prices
def plot_stock_prices(adj_close_df):
   fig = go.Figure()
   for ticker in adj_close_df.columns:
       fig.add_trace(go.Scatter(x=adj_close_df.index, y=adj_close_df[ticker], mode='lines', name=ticker))
   fig.update_layout(title='Stock Price Trends Over Time',
                     xaxis_title='Date',
                     yaxis_title='Adjusted Close Price' ,
                     template="plotly_dark",
                     plot_bgcolor='rgba(0,0,0,0)',
                     paper_bgcolor='rgba(0,0,0,0)',
                     font_color="white",)
   return fig

# Function to plot the Cumulative Capital
def plot_cumulative_capital(cumulative_capital):
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=cumulative_capital.index, y=cumulative_capital['Portfolio Value'], mode='lines', name='Portfolio Value'))
   fig.update_layout(title='Cumulative Capital Over Time with Quarterly Rebalancing and Profit Taking',
                     xaxis_title='Date',
                     yaxis_title='Portfolio Value in $',
                     template="plotly_dark",
                     plot_bgcolor='rgba(0,0,0,0)',
                     paper_bgcolor='rgba(0,0,0,0)',
                     font_color="white",)
   return fig

# Update the Dash app layout to include these plots
optimisation_layout = html.Div([
   html.H1('Portfolio Optimization Dashboard'),
   dcc.Graph(id='efficient-frontier', figure=efficient_frontier_fig),
   dcc.Graph(id='stock-price-trends', figure=plot_stock_prices(adj_close_df)),
   dcc.Graph(id='cumulative-capital', figure=plot_cumulative_capital(cumulative_capital))
])


