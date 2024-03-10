import optimise_get_returns as opt
import pandas as pd
import yfinance as yf
import plotly.express as px
mu=0.10

#fetch the portfolio assets using fundamentals from bloomberg
investible_universe = opt.return_stocks()

#fetch expected returns and other parameters
(returns_2023, weights, V, invV,
  alpha, beta, gamma, delta, 
  lambda1, lambda2, epsilonG, 
 epsilonD, epsilonMVE) = opt.get_2023_returns(mu)

#Change date index to column
returns_2023.reset_index(inplace=True)

efficient_frontier_fig = opt.plot_efficient_frontier()

weights, V, invV, alpha, beta, gamma, delta, lambda1, lambda2, epsilonG, epsilonD, epsilonMVE = opt.get_weights(opt.get_expected_returns('2023-12-31'), mu)



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

print(df_combined)

treemap_fig = px.treemap(df_combined,
                         path=[px.Constant("Portfolio"), 'Sector', 'Ticker'],
                         values='Weight',
                         title='Current Portfolio Weightings by Sector and Stock')
treemap_fig.show()


returns_fig = opt.plot_returns(returns_2023)
returns_fig.show()
efficient_frontier_fig.show()


