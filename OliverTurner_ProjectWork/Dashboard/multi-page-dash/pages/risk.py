import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash_table import DataTable
import yfinance as yf
import numpy as np
import plotly.express as px
from scipy import stats
import datetime as dt
import pandas as pd
from scipy.optimize import minimize

# Function to calculate beta
def calculate_beta(stock_symbol, market_index_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)['Adj Close']
    market_data = yf.download(market_index_symbol, start=start_date, end=end_date)['Adj Close']

    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data.pct_change().dropna()

    covariance_matrix = np.cov(stock_returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]

    return beta

# Function to rank stocks by beta
def rank_stocks_by_beta(portfolio, market_index, start_date, end_date):
    beta_values = []

    for stock_symbol in portfolio:
        beta = calculate_beta(stock_symbol, market_index, start_date, end_date)
        beta_values.append((stock_symbol, beta))

    sorted_beta_values = sorted(beta_values, key=lambda x: x[1], reverse=True)
    return np.array(sorted_beta_values)

# Function to calculate Treynor ratio
def treynor_ratio(stocks, start_date, end_date, rfr):
    results = []
    try:
        for i in stocks:
            data = yf.download(i, start=start_date, end=end_date)['Adj Close']
            ret = data.pct_change()

            bench = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
            bench_ret = bench.pct_change()

            cumu_i = (data.iloc[-1] / data.iloc[0]) - 1
            beta_i = stats.linregress(ret.dropna(), bench_ret.dropna())[0]
            treynor_i = (cumu_i - rfr / 100) / beta_i
            results.append({'Stocks': i, 'Cumulative Return': cumu_i, 'Treynor Ratio': treynor_i})

    except Exception as e:
        print(f"Error fetching data: {e}")

    return results

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)['Adj Close']

# Function to simulate Heston model
def heston_model_simulation(params, S0, T, N):
    kappa, theta, v0, rho, sigma = params
    dt = T / N
    prices = np.zeros(N + 1)
    variances = np.zeros(N + 1)

    prices[0] = S0
    variances[0] = v0

    for t in range(1, N + 1):
        dW_s = np.random.normal(0, np.sqrt(dt))
        dW_v = rho * dW_s + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))

        variances[t] = max(variances[t-1] + kappa * (theta - max(variances[t-1], 0)) * dt +
                           sigma * np.sqrt(max(variances[t-1], 0)) * dW_v, 0)
        prices[t] = prices[t-1] * np.exp(-0.5 * variances[t-1] * dt +
                                          np.sqrt(variances[t-1] * dt) * dW_s)

    return prices

# Function to estimate Heston model parameters
def estimate_heston_parameters(observed_prices, dt):
    initial_params = [0.1, 0.04, 0.04, 0.1, 0.3]  # Adjust initial guess
    result = minimize(objective_function, initial_params, args=(observed_prices, dt), method='L-BFGS-B',
                      bounds=[(0, None), (0, None), (0, None), (-1, 1), (0, None)])
    return result.x

# Function to simulate returns for a stock
def simulate_returns_for_stock(ticker, start_date, end_date, T, N):
    historical_data = fetch_stock_data(ticker, start_date, end_date)
    observed_prices = historical_data.values
    S0 = observed_prices[-1]
    dt = T / N

    # Estimate Heston model parameters for this stock
    estimated_params = estimate_heston_parameters(observed_prices, dt)

    # Simulate new price paths with the estimated Heston parameters
    simulated_prices = heston_model_simulation(estimated_params, S0, T, N)

    # Calculate returns from the simulated prices
    simulated_returns = np.diff(simulated_prices) / simulated_prices[:-1]

    return simulated_returns

# Function to calculate ES
def calculate_es(simulated_returns, alpha):
    sorted_returns = np.sort(simulated_returns)
    cutoff_index = int(alpha * len(sorted_returns))
    return -np.mean(sorted_returns[:cutoff_index])

# Function to calculate VaR and C-VaR
def calculate_var_cvar(simulated_returns, alpha):
    var_value = -np.percentile(np.sum(simulated_returns, axis=0), 100 * alpha)
    cvar_value = calculate_es(np.sum(simulated_returns, axis=0), alpha)
    return var_value, cvar_value

# Objective function for optimization
def objective_function(params, observed_prices, dt):
    model_prices = heston_model_simulation(params, observed_prices[0], len(observed_prices) - 1, len(observed_prices) - 1)
    differences = model_prices - observed_prices
    return np.sum(differences ** 2)

# Example usage
tickers = ['MMM', 'ABBV', 'APD', 'AEP', 'AMGN', 'ADM', 'T', 'BKR', 'BAC', 'BK',
           'BAX', 'BMY', 'CVX', 'CSCO', 'KO', 'CMCSA', 'ED', 'GLW', 'CTRA',
           'CVS', 'DRI', 'DVN', 'FANG', 'D', 'DOW', 'DTE', 'DUK', 'EIX', 'ETR',
           'EOG', 'ES', 'EXC', 'XOM', 'FIS', 'FITB', 'FE', 'F', 'GIS', 'GPC',
           'GILD', 'GS', 'HSY', 'HPE', 'IBM', 'JNJ', 'KDP', 'KMB', 'KMI',
           'LMT', 'LYB', 'MDT', 'MET', 'MS', 'NEM', 'NEE', 'OKE', 'PEP', 'PFE',
           'PSX', 'PXD', 'PNC', 'PRU', 'PEG', 'RTX', 'SRE', 'SO', 'STT', 'TROW',
           'TGT', 'USB', 'UPS', 'VLO', 'VZ', 'VICI', 'WEC', 'WFC', 'WMB', 'XEL']
start_date = dt.datetime.now() - dt.timedelta(days=365*3)
end_date = dt.datetime.now()
T = 1/4
N = 252

all_returns = []
for ticker in tickers:
    simulated_returns = simulate_returns_for_stock(ticker, start_date, end_date, T, N)
    all_returns.append(simulated_returns)

# Calculate the covariance matrix of the simulated returns
cov_matrix = np.cov(all_returns)

# Calculate Expected Shortfall (ES) at different confidence levels
alphas = [0.1, 0.05, 0.01]
for alpha in alphas:
    es_value = calculate_es(np.sum(all_returns, axis=0), alpha)
    print(f"ES at {(1-alpha) * 100}% confidence level: {es_value * 1000}%")

# Calculate Value at Risk (VaR) at different confidence levels
for alpha in alphas:
    var_value = -np.percentile(np.sum(all_returns, axis=0), 100 * alpha)
    print(f"VaR at {(1-alpha) * 100}% confidence level: {var_value * 1000}%")

import dash
from dash import dcc, html
import dash_table
from dash.dependencies import Input, Output
import numpy as np
import yfinance as yf
from scipy import stats
import datetime as dt
import pandas as pd

app = dash.Dash(__name__)

# Stock Beta Section
portfolio = ['MMM', 'ABBV', 'APD', 'AEP', 'AMGN', 'ADM', 'T', 'BKR', 'BAC', 'BK',
           'BAX', 'BMY', 'CVX', 'CSCO', 'KO', 'CMCSA', 'ED', 'GLW', 'CTRA',
           'CVS', 'DRI', 'DVN', 'FANG', 'D', 'DOW', 'DTE', 'DUK', 'EIX', 'ETR',
           'EOG', 'ES', 'EXC', 'XOM', 'FIS', 'FITB', 'FE', 'F', 'GIS', 'GPC',
           'GILD', 'GS', 'HSY', 'HPE', 'IBM', 'JNJ', 'KDP', 'KMB', 'KMI',
           'LMT', 'LYB', 'MDT', 'MET', 'MS', 'NEM', 'NEE', 'OKE', 'PEP', 'PFE',
           'PSX', 'PXD', 'PNC', 'PRU', 'PEG', 'RTX', 'SRE', 'SO', 'STT', 'TROW',
           'TGT', 'USB', 'UPS', 'VLO', 'VZ', 'VICI', 'WEC', 'WFC', 'WMB', 'XEL']
market_index = '^GSPC'  # S&P 500 as an example
start_date = dt.datetime.now() - dt.timedelta(days = 365*3)
end_date = dt.datetime.now()

ranked_portfolio = rank_stocks_by_beta(portfolio, market_index, start_date, end_date)

# Heston Model Section
start_date_heston = dt.datetime.now() - dt.timedelta(days=365*3)
end_date_heston = dt.datetime.now()
T_heston = 1/4
N_heston = 252

all_returns = []
for ticker in tickers:
    simulated_returns = simulate_returns_for_stock(ticker, start_date_heston, end_date_heston, T_heston, N_heston)
    all_returns.append(simulated_returns)

cov_matrix = np.cov(all_returns)

alphas_heston = [0.1, 0.05, 0.01]
for alpha in alphas_heston:
    es_value = calculate_es(np.sum(all_returns, axis=0), alpha)
    print(f"ES at {(1-alpha) * 100}% confidence level: {es_value * 1000}%")

for alpha in alphas_heston:
    var_value = -np.percentile(np.sum(all_returns, axis=0), 100 * alpha)
    print(f"VaR at {(1-alpha) * 100}% confidence level: {var_value * 1000}%")

# Treynor Ratio Section
app.layout = html.Div(children=[
    html.H1("Portfolio Risk Management"),

    # Stock Beta Section
    html.Div([
        html.H2("Stock Beta Analysis"),
        html.Label("Select Stock:"),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': stock, 'value': stock} for stock in portfolio],
            value=portfolio[0]
        ),
        dcc.Graph(
            id='beta-histogram'
        ),
        html.Div([
            html.H3("Selected Stock Beta:"),
            html.Div(id='selected-stock-beta')
        ])
    ]),

    # Heston Model Section
    html.Div([
        html.H2("Heston Model Analysis"),
        DataTable(
            id='var-cvar-table',
            columns=[
                {'name': 'Confidence Interval', 'id': 'Alpha', 'type': 'text', 'presentation': 'dropdown'},
                {'name': 'VaR', 'id': 'VaR', 'type': 'text'},
                {'name': 'C-VaR', 'id': 'C-VaR', 'type': 'text'},
            ],
            style_cell={
                'textAlign': 'center',
                'minWidth': '100px',
                'width': '100px',
                'maxWidth': '100px',
                'whiteSpace': 'normal',
            },
            style_table={
                'overflowX': 'auto',
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Alpha'},
                    'fontWeight': 'bold',
                },
            ],
        )
    ]),

    # Treynor Ratio Section
    html.Div([
        html.H2("Treynor Ratio Analysis"),
        dash_table.DataTable(
            id='treynor-table',
            columns=[
                {'name': 'Stocks', 'id': 'Stocks'},
                {'name': 'Cumulative Return', 'id': 'Cumulative Return'},
                {'name': 'Treynor Ratio', 'id': 'Treynor Ratio'},
            ],
            style_table={'height': '500px', 'overflowY': 'auto'},
        ),
    ]),
])

# Callback to update the histogram and display the specific beta value
@app.callback(
    [Output('beta-histogram', 'figure'),
     Output('selected-stock-beta', 'children')],
    [Input('stock-dropdown', 'value')]
)
def update_histogram(selected_stock):
    beta_values = [beta for _, beta in ranked_portfolio]

    # Create color scale based on beta values
    color_scale = px.colors.diverging.RdBu[::-1]  # You can choose a different color scale

    figure = {
        'data': [
            {
                'x': [stock for stock, _ in ranked_portfolio],
                'y': beta_values,
                'type': 'bar',
                'name': 'Beta Values',
                'marker': {'color': beta_values, 'colorscale': color_scale, 'showscale': True}
            }
        ],
        'layout': {
            'title': 'Beta Values for Stocks',
            'xaxis': {'title': 'Stocks'},
            'yaxis': {'title': 'Beta Value'}
        }
    }

    selected_stock_beta = [f"Beta value for {selected_stock}: {beta}" for stock, beta in ranked_portfolio if stock == selected_stock][0]

    return figure, selected_stock_beta

# Callback to update VaR and C-VaR table
@app.callback(Output('var-cvar-table', 'data'), [Input('var-cvar-table', 'id')])
def update_var_cvar_table(dummy):
    var_cvar_data = {'Alpha': [], 'VaR': [], 'C-VaR': []}
    for alpha in alphas_heston:
        var_value, cvar_value = calculate_var_cvar(all_returns, alpha)
        var_cvar_data['Alpha'].append(f"{(1 - alpha) * 100:.2f}%")
        var_cvar_data['VaR'].append(f"{var_value * 1000:.2f}%")
        var_cvar_data['C-VaR'].append(f"{cvar_value * 1000:.2f}%")

    # Create DataFrame for the table
    var_cvar_df = pd.DataFrame(var_cvar_data)

    return var_cvar_df.to_dict('records')

# Callback to update Treynor Ratio table
@app.callback(Output('treynor-table', 'data'), [Input('treynor-table', 'id')])
def update_treynor_table(dummy):
    stocks_treynor = ['MMM', 'ABBV', 'APD', 'AEP', 'AMGN', 'ADM', 'T', 'BKR', 'BAC', 'BK', 
           'BAX', 'BMY', 'CVX', 'CSCO', 'KO', 'CMCSA', 'ED', 'GLW', 'CTRA', 
           'CVS', 'DRI', 'DVN', 'FANG', 'D', 'DOW', 'DTE', 'DUK', 'EIX', 'ETR',
           'EOG', 'ES', 'EXC', 'XOM', 'FIS', 'FITB', 'FE', 'F', 'GIS', 'GPC', 
           'GILD', 'GS', 'HSY', 'HPE', 'IBM', 'JNJ', 'KDP', 'KMB', 'KMI',
           'LMT', 'LYB', 'MDT', 'MET', 'MS', 'NEM', 'NEE', 'OKE', 'PEP', 'PFE', 
           'PSX', 'PXD', 'PNC', 'PRU', 'PEG', 'RTX', 'SRE', 'SO', 'STT', 'TROW',
           'TGT', 'USB', 'UPS', 'VLO', 'VZ', 'VICI', 'WEC', 'WFC', 'WMB', 'XEL']
    start_date_treynor = dt.datetime.now() - dt.timedelta(days=365*2)
    end_date_treynor = dt.datetime.now()
    
    results_treynor = treynor_ratio(stocks_treynor, start_date_treynor, end_date_treynor, 2)
    return results_treynor

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)