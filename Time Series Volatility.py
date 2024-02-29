import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime, timedelta

# list of tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

def fit_garch_and_plot(ticker):
    # ticker and dates
    start = '2021-01-01'
    today = datetime.now().date().strftime("%Y-%m-%d")

    # historical price data from yahoo
    stock_data = yf.download(ticker, start=start, end=today)

    returns = 100 * stock_data['Adj Close'].pct_change().dropna()

    # plot returns
    plt.figure(figsize=(10,4))
    plt.plot(returns)
    plt.ylabel('Pct Return')
    plt.title(f'{ticker} Returns')

    # parameters to test in GARCH model
    p_range = range(1, 4)
    q_range = range(1, 4)

    # perform grid search for best parameters
    best_p, best_q = grid_search_garch(returns, p_range, q_range, criterion='aic')
    print(f"Best (p, q) parameters for {ticker}: ({best_p}, {best_q})")

    # fit garch model with best parameters
    model = arch_model(returns, p=best_p, q=best_q)
    model_fit = model.fit(disp='off')

    # evaluate the GARCH model
    print(model_fit.summary())

    # plot ACF and PACF of residuals
    residuals = model_fit.resid
    plot_acf(residuals, lags=20)
    plot_pacf(residuals, lags=20)
    plt.show()

    # forecast horizon
    horizon = 7
    
    # fit GARCH model with best parameters
    model = arch_model(returns, p=best_p, q=best_q)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=horizon)
    future_dates = [returns.index[-1] + timedelta(days=i) for i in range(1, horizon+1)]
    pred = pd.Series(np.sqrt(pred.variance.values[-1,:]), index=future_dates)
    plt.figure(figsize=(10,4))
    plt.plot(pred)
    plt.title(f'Volatility {horizon} Days Prediction for {ticker}', fontsize=20)
    plt.show()

def grid_search_garch(returns, p_range, q_range, criterion='aic'):
    best_criterion = np.inf if criterion == 'aic' else np.inf
    best_params = None
    
    for p in p_range:
        for q in q_range:
            model = arch_model(returns, p=p, q=q)
            results = model.fit(disp='off')
            if criterion == 'aic':
                current_criterion = results.aic
            elif criterion == 'bic':
                current_criterion = results.bic
            
            if current_criterion < best_criterion:
                best_criterion = current_criterion
                best_params = (p, q)
                
    return best_params

# fit GARCH model for each tickers
for ticker in tickers:
    fit_garch_and_plot(ticker)
