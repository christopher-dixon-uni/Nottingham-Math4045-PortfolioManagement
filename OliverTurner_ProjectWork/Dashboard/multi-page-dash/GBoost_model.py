import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from datetime import datetime, timedelta
import pickle
import json

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    return yf.download(ticker, start=start_date, end=end_date)

def calculate_features(df):
    """
    Calculate selected features for a given stock DataFrame.
    """
    df['5-Day MA'] = df['Adj Close'].rolling(window=5).mean()
    df['20-Day MA'] = df['Adj Close'].rolling(window=20).mean()
    delta = df['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['14-Day RSI'] = 100 - (100 / (1 + rs))
    df['26-Day EMA'] = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['12-Day EMA'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
    df['MACD'] = df['12-Day EMA'] - df['26-Day EMA']
    return df.dropna()

def calculate_future_return(df, period=63):
    """
    Calculate future 3-month returns based on adjusted close prices.
    """
    df['Future 3M Return'] = df['Adj Close'].pct_change(periods=-period).shift(periods=period)
    return df.dropna()

def retrain_model(tickers, start_date, end_date, period=63):
    """
    Retrain the model for each ticker.
    """
    models = {}

    for ticker in tickers:
        df = fetch_stock_data(ticker, start_date, end_date)
        df = calculate_features(df)
        df = calculate_future_return(df, period=period)
        df.dropna(inplace=True)

        if not df.empty:
            X = df[['5-Day MA', '20-Day MA', '14-Day RSI', 'MACD']]
            y = df['Future 3M Return']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            
            models[ticker] = model

            #save models to file using pickle
            with open(f'./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/models.pkl', 'wb') as f:
                for model in models:
                    pickle.dump(models, f)



    return models

def predict_latest_return(tickers, models, start_date, end_date):
    """
    Predict the expected return for today (or the closest possible) for the next 3 months for multiple tickers.
    """
    predictions = {}
    for ticker, model in models.items():
        df = fetch_stock_data(ticker, start_date, end_date)
        df = calculate_features(df)

        if not df.empty:
            latest_features = df[['5-Day MA', '20-Day MA', '14-Day RSI', 'MACD']].iloc[-1:].dropna()

            if not latest_features.empty:
                predicted_return = model.predict(latest_features)[0]
                predictions[ticker] = predicted_return
            else:
                predictions[ticker] = float('nan')

    return predictions


def train_and_save():
    # Define your tickers and date range
    tickers = ['MMM', 'ABBV', 'APD', 'AEP', 'AMGN', 'ADM', 'T', 'BKR', 'BAC', 'BK', 'BAX', 'BMY', 'CVX', 'CSCO', 'KO', 'CMCSA', 'ED', 'GLW', 'CTRA', 'CVS', 'DRI', 'DVN', 'FANG', 'D', 'DOW', 'DTE', 'DUK', 'EIX', 'ETR', 'EOG', 'ES', 'EXC', 'XOM', 'FIS', 'FITB', 'FE', 'F', 'GIS', 'GPC', 'GILD', 'GS', 'HSY', 'HPE', 'IBM', 'JNJ', 'KDP', 'KMB', 'KMI', 'LMT', 'LYB', 'MDT', 'MET', 'MS', 'NEM', 'NEE', 'OKE', 'PEP', 'PFE', 'PSX', 'PXD', 'PNC', 'PRU', 'PEG', 'RTX', 'SRE', 'SO', 'STT', 'TROW', 'TGT', 'USB', 'UPS', 'VLO', 'VZ', 'VICI', 'WEC', 'WFC', 'WMB', 'XEL']
    end_date = '2023-01-01'
    start_date = '2013-01-01'

    # Retrain models and predict the latest return
    models = retrain_model(tickers, start_date, end_date)
    print('models trained and saved to file')


def load_and_predict():
    
    tickers = ['MMM', 'ABBV', 'APD', 'AEP', 'AMGN', 'ADM', 'T', 'BKR', 'BAC', 'BK', 'BAX', 'BMY', 'CVX', 'CSCO', 'KO', 'CMCSA', 'ED', 'GLW', 'CTRA', 'CVS', 'DRI', 'DVN', 'FANG', 'D', 'DOW', 'DTE', 'DUK', 'EIX', 'ETR', 'EOG', 'ES', 'EXC', 'XOM', 'FIS', 'FITB', 'FE', 'F', 'GIS', 'GPC', 'GILD', 'GS', 'HSY', 'HPE', 'IBM', 'JNJ', 'KDP', 'KMB', 'KMI', 'LMT', 'LYB', 'MDT', 'MET', 'MS', 'NEM', 'NEE', 'OKE', 'PEP', 'PFE', 'PSX', 'PXD', 'PNC', 'PRU', 'PEG', 'RTX', 'SRE', 'SO', 'STT', 'TROW', 'TGT', 'USB', 'UPS', 'VLO', 'VZ', 'VICI', 'WEC', 'WFC', 'WMB', 'XEL']
    start_dates = ['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01']
    end_dates = ['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31']

    predicted_returns = {}
    for start_date, end_date in zip(start_dates, end_dates):
        models = pickle.load(open('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/models.pkl', 'rb'))
        predicted_returns[f'{end_date}'] = predict_latest_return(tickers, models, start_date, end_date)

    with open('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/predicted_returns.json', 'w') as f:
        json.dump(predicted_returns, f)

#run model and save to json for beggining of each quarter in 2023, predictin 3 months ahead
if __name__ == "__main__":
    load_and_predict()




