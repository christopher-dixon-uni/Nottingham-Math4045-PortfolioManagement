import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Define a portfolio of stocks
portfolio_symbols = portfolio_symbols = ['MMM', 'ABBV', 'APD', 'AEP', 'AMGN', 'ADM', 'T', 'BKR', 'BAC', 'BK', 'BAX', 'BMY', 'CVX', 'CSCO', 'KO', 'CMCSA', 'ED', 'GLW', 'CTRA', 'CVS', 'DRI', 'DVN', 'FANG', 'D', 'DOW', 'DTE', 'DUK', 'EIX', 'ETR', 'EOG', 'ES', 'EXC', 'XOM', 'FIS', 'FITB', 'FE', 'F', 'GIS', 'GPC', 'GILD', 'GS', 'HSY', 'HPE', 'IBM', 'JNJ', 'KVUE', 'KDP', 'KMB', 'KMI', 'LMT', 'LYB', 'MDT', 'MET', 'MS', 'NEM', 'NEE', 'OKE', 'PEP', 'PFE', 'PSX', 'PXD', 'PNC', 'PRU', 'PEG', 'RTX', 'SRE', 'SO', 'STT', 'TROW', 'TGT', 'USB', 'UPS', 'VLO', 'VZ', 'VICI', 'WEC', 'WFC', 'WMB', 'XEL']

# Prepare the GDP growth data
gdp_growth_extended = pd.DataFrame({
    'Date': pd.to_datetime([
        '2023-10-01', '2023-07-01', '2023-04-01', '2023-01-01',
        '2022-10-01', '2022-07-01', '2022-04-01', '2022-01-01',
        '2021-10-01', '2021-07-01', '2021-04-01', '2021-01-01',
        '2020-10-01', '2020-07-01', '2020-04-01', '2020-01-01',
        '2019-10-01', '2019-07-01', '2019-04-01', '2019-01-01',
        '2018-10-01', '2018-07-01', '2018-04-01', '2018-01-01',
        '2017-10-01', '2017-07-01', '2017-04-01', '2017-01-01',
        '2016-10-01', '2016-07-01', '2016-04-01', '2016-01-01',
        '2015-10-01', '2015-07-01', '2015-04-01', '2015-01-01',
        '2014-10-01', '2014-07-01', '2014-04-01', '2014-01-01',
        '2013-10-01', '2013-07-01', '2013-04-01'
    ]),
    'GDP_Growth': [
        3.3, 4.9, 2.1, 2.2,
        2.6, 3.2, -0.6, -1.6,
        7.0, 2.7, 7.0, 6.3,
        3.9, 35.3, -29.9, -4.6,
        1.8, 3.6, 2.7, 2.2,
        0.7, 2.9, 2.8, 2.8,
        4.1, 3.4, 2.0, 1.7,
        2.0, 2.4, 1.2, 2.4,
        0.6, 1.3, 2.3, 3.3,
        1.8, 4.7, 5.2, -1.4,
        2.9, 3.2, 0.6
    ]
})

# Interest rates and GDP growth data setup
tnx = yf.Ticker("^TNX")
interest_rates = tnx.history(period="10y")['Close'].rename('Interest_Rate')
interest_rates.index = pd.to_datetime(interest_rates.index).tz_localize(None)

# Initialize an empty list to store predicted returns for each stock
predicted_returns_vector = []

for symbol in portfolio_symbols:
    # Fetch historical data for the stock
    stock_data = yf.Ticker(symbol).history(period="10y")
    stock_data.index = pd.to_datetime(stock_data.index).tz_localize(None)
    
    # Merge with interest rates
    stock_data = stock_data.merge(interest_rates, how='left', left_index=True, right_index=True)
    
    # Merge GDP data on the closest date without going over for each stock data point
    stock_data.reset_index(inplace=True)
    stock_data['GDP_Growth'] = stock_data['Date'].apply(
        lambda x: gdp_growth_extended[gdp_growth_extended['Date'] <= x]['GDP_Growth'].iloc[-1])
    
    stock_data.set_index('Date', inplace=True)
    
    # Calculate features
    stock_data['Lagged_Return'] = stock_data['Close'].pct_change().shift()
    stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
    
    # Ensure no NaN or infinite values in the newly created features
    stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    stock_data.dropna(inplace=True)

    # Adjust target variable to predict the 3-month future return
    stock_data['Future_3M_Return'] = stock_data['Close'].pct_change(periods=63).shift(-63)
    
    # Drop NaN values introduced by the shift operation
    stock_data.dropna(inplace=True)

    # Define predictors and target
    X = stock_data[['Lagged_Return', 'Volume_Change', 'Interest_Rate', 'GDP_Growth']]
    y = stock_data['Future_3M_Return']  # 3-month future return
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale features
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Train models
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    gb_reg = GradientBoostingRegressor(random_state=42)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5]}
    grid_search = GridSearchCV(gb_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_gb_reg = grid_search.best_estimator_
    
    # Predict future returns
    y_pred_lin = lin_reg.predict(X_test)
    y_pred_gb = best_gb_reg.predict(X_test)
    y_pred_combined = (y_pred_lin + y_pred_gb) / 2
    
    # Append the last predicted return
    predicted_returns_vector.append(y_pred_combined[-1])

# Convert the list of predicted returns into a numpy array
predicted_returns_vector = np.array(predicted_returns_vector)

print("Predicted returns vector:", predicted_returns_vector)