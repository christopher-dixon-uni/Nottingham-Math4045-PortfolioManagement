# Creating a python module with functions to retrieve data
# from the Yahoo Finance API
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import os
import re
import json

# creates a json with portfolio details

def create_portfolio(investment: float, start_date: str, end_date: str,  
           assets: list, weights: list):
    """
    Reads most recent JSON to find the most recent date and investment amount.
    Saves the inputs as a JSON object, each json numbered sequentially.

    Args:
        investment (float): The amount of investment.
        start_date (str): The start date of the portfolio.
        end_date (str): The end date of the portfolio.
        assets (list): List of asset names.
        weights (list): List of corresponding asset weights.

    Returns:
        None
        JSON saved as follows:
        {
            "investment": investment,
            "start_date": start_date,
            "end_date": end_date,
            "assets": assets,
            "weights": weights}
    """

    Portfolio = {
                "investment": investment,
                "start_date": start_date,
                "end_date": end_date,
                "assets": assets,
                "weights": weights}



    # Count the number of files in the directory
    path = 'c:/Users/olive/OneDrive - The University of Nottingham/Year 3/Group Project/Portfolio-Optimisation/Live Data/Portfolios'
    files = os.listdir(path)
    json_files = [f for f in files if f.endswith('.json')]

    # Save the portfolio as a JSON file
    with open(f'Portfolios/portfolio_{len(json_files) + 1}.json', 'w') as f:
        json.dump(Portfolio, f)





# reads the most recent portfolio from the json
        
def get_recent_portfolio() -> dict:
    """
    Reads the most recent JSON file in the directory and returns the portfolio.

    Returns:
        dict: The portfolio as a dictionary.
    """
    list_of_files = os.listdir('./Portfolios')

    def extract_number(f):
        s = re.findall("\d+$",f)
        return (int(s[0]) if s else -1,f)

    #most recent portfolio:
    with open(f'./Portfolios/{max(list_of_files,key=extract_number)}', 'r') as file:
        Portfolio = json.load(file)
    
    return Portfolio



# downloads the data from the Yahoo Finance API accounding to the portfolio details

def download_data(Portfolio: dict) -> pd.DataFrame:

    """
    Uses JSON in path to download the data from Yahoo Finance API.

    Args:
        path (str): The path to the JSON file.

    Returns:
        pandas.DataFrame: The data for the assets.
    """

    #getting parameters from the JSON file
    start_date = Portfolio['start_date']
    end_date = Portfolio['end_date']
    assets = Portfolio['assets']
    weights = Portfolio['weights']
    investment = Portfolio['investment']

    # Download data from Yahoo Finance API
    data = yf.download(" ".join(assets), start=start_date, end=end_date)

    # Flatten headers in data
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

    #asset daily returns
    for a in assets:
        data[f'Daily Returns_{a}'] = data[f'Adj Close_{a}'].pct_change()

    # Portfolio Daily Returns
    portfolio_daily_returns = [weights[i]*data[f'Daily Returns_{assets[i]}'] for i in range(len(assets))]
    data['Portfolio Daily Returns'] = sum(portfolio_daily_returns)
    

    #Portfolio cumulative returns
    data['Portfolio Cumulative Returns'] = (1 + data['Portfolio Daily Returns']).cumprod()

    #Portfolio value
    data['Portfolio Value'] = data['Portfolio Cumulative Returns'] * investment

         

    return data


# creates a json with portfolio details, returns the updated portfolio details after running the portfolio

def create_and_test_portfolio(investment: float, start_date: str, end_date: str,  
           assets: list, weights: list):
    """
    Reads most recent JSON to find the most recent date and investment amount.
    Saves the inputs as a JSON object, each json numbered sequentially.

    Args:
        investment (float): The amount of investment.
        start_date (str): The start date of the portfolio.
        end_date (str): The end date of the portfolio.
        assets (list): List of asset names.
        weights (list): List of corresponding asset weights.

    Returns:
        None
        JSON saved as follows:
        {
            "investment": investment,
            "start_date": start_date,
            "end_date": end_date,
            "assets": assets,
            "weights": weights,
            cum_returns: cum_returns,
            investment_end: investment_end}
    """

    Portfolio = {
            "investment": investment,
            "start_date": start_date,
            "end_date": end_date,
            "assets": assets,
            "weights": weights}

    #downloading data and running at current weights
    df = download_data(Portfolio)
    cum_returns = df['Portfolio Cumulative Returns'][-1]
    investment_end = df['Portfolio Value'][-1]

    # Saving cumulative returns and investment_end to the Portfolio dictionary
    Portfolio['cum_returns'] = cum_returns
    Portfolio['investment_end'] = investment_end

    # updating the JSON file

    # Count the number of files in the directory
    path = 'c:/Users/olive/OneDrive - The University of Nottingham/Year 3/Group Project/Portfolio-Optimisation/Live Data/Portfolios'
    files = os.listdir(path)
    json_files = [f for f in files if f.endswith('.json')]

    # Save the portfolio as a JSON file
    with open(f'Portfolios/portfolio_{len(json_files) + 1}.json', 'w') as f:
        json.dump(Portfolio, f)
