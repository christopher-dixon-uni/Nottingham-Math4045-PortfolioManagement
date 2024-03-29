import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta


# list of tickers
tickers = ['AAPL', 'MSFT']

def arima_model(ticker):
    # dates
    start = '2021-01-01'
    today = datetime.now().date().strftime("%Y-%m-%d") # end

    # historical price data from yahoo
    data = yf.Ticker(ticker)
    prices = data.history(start=start, end=today).Close
    returns = prices.pct_change().dropna()

    # plot prices
    plt.figure(figsize=(10,4))
    plt.plot(prices)
    plt.ylabel('Prices', fontsize=20)
    plt.title(f'Stock Prices for {ticker}', fontsize=20)
    plt.show()

    # check stationarity using ADF test
    def check_stationarity(data):
        result = adfuller(data)
        adf_statistic = result[0]
        p_value = result[1]
        return adf_statistic, p_value

    prices = pd.Series(prices)

    # check initial stationarity
    adf_statistic, p_value = check_stationarity(prices)
    print("Initial ADF Statistic:", adf_statistic)
    print("Initial p-value:", p_value)

    # differencing until the data is stationary
    d = 0 
    prices_adf = prices

    while p_value > 0.05:
        prices_adf = prices_adf.diff().dropna() 
        d += 1
        adf_statistic, p_value = check_stationarity(prices_adf) 
        print("New ADF Statistic after differencing:", adf_statistic)
        print("New p-value after differencing:", p_value)

    # plot prices again after it is stationary
    plt.figure(figsize=(10,4))
    plt.plot(prices_adf) 
    plt.ylabel('Prices')
    plt.title(f'Stationary Prices for {ticker}', fontsize=20)
    plt.show()

    # parameters to test in ARIMA model
    parameters = [(1,d,0), (2,d,0), (3,d,0), 
                  (1,d,1), (2,d,1), (3,d,1), 
                  (1,d,2), (2,d,2), (3,d,2), 
                  (1,d,3), (2,d,3), (3,d,3),
                  (0,d,1), (0,d,2), (0,d,3)]

    # AIC test to find the best parameters
    def test_arima_models(prices, parameters): 
        best_aic = float("inf")
        best_model = None
        best_parameter = None

        for p, d, q in parameters:
            model = ARIMA(prices, order=(p, d, q)) 
            results = model.fit()
            aic = results.aic
            if aic < best_aic:
                best_aic = aic
                best_model = results
                best_parameter = (p, d, q)    
        return best_model, best_parameter

    # fit ARIMA model with the best parameters
    best_model, best_parameter = test_arima_models(prices, parameters) 
    p, d, q = best_parameter
    model_fit = best_model

    # evaluate the ARIMA model
    print(model_fit.summary())

    # split into train and test data
    train_size = 0.8 # 80%
    split = int(len(prices) * train_size)
    train_data = prices[:split]
    test_data = prices[split:]

    # perform the rolling forecast
    rolling_predictions = pd.Series(index=test_data.index) 

    for i, test_date in enumerate(test_data.index):
        train_end = test_date - timedelta(days=30)  # change days to change training window
        train_data_window = prices[:train_end]
        model = ARIMA(train_data_window, order=(p, d, q))
        model_fit = model.fit()
        pred = float(model_fit.forecast().iloc[0])
        rolling_predictions[test_date] = pred

    # calculate residuals
    rolling_residuals = test_data - rolling_predictions

    # plot residuals
    plt.figure(figsize=(10,4))
    plt.plot(rolling_residuals)
    plt.axhline(0, linestyle='--', color='k')
    plt.title(f'ARIMA Model Residuals for {ticker}', fontsize=20)
    plt.ylabel('Error', fontsize=16)
    plt.show()

    # plot forecast
    plt.figure(figsize=(10,4))
    plt.plot(prices)
    plt.plot(rolling_predictions)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title(f'Forecast for {ticker}', fontsize=20)
    plt.show()

# fit ARIMA model for each ticker
for ticker in tickers:
    arima_model(ticker)
