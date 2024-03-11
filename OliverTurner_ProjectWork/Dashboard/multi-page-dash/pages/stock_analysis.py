import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.io as pio
import plotly.tools as tls
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from app_instance import app
import dash_bootstrap_components as dbc




# dates
start = '2021-01-01'
today = datetime.now().date().strftime("%Y-%m-%d")


# function to calculate correlation matrix
def calculate_correlation_matrix(tickers, start_date, end_date):
    prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    correlation_matrix = prices.pct_change().corr()
    return correlation_matrix


# tickers and parameters
tickers = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
parameters = [(5,1,2), (4,1,2), (2,1,3), (4,1,2), (3,1,2), (0,1,1), (1,1,1), (1,0,1), (2,1,5), (2,1,2), (1,0,0)]


# calculate correlation matrix
correlation_matrix = calculate_correlation_matrix(tickers, start, today)


# function to fetch YTD data 
def fetch_sector_etf_data():
    sector_tickers = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
    today = datetime.today().strftime('%Y-%m-%d')
    start_date = datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d')
    sector_data = yf.download(sector_tickers, start=start_date, end=today)['Adj Close']
    # calculate daily returns for each sector
    sector_returns = sector_data.pct_change()
    # calculate cumulative growth for each sector
    cumulative_growth = (1 + sector_returns).cumprod()
    
    return cumulative_growth


# function to generate forecast plot
def generate_forecast_plot(ticker, parameters):
    data = yf.Ticker(ticker)
    prices = data.history(start=start, end=today).Close
    #prices = data.history(start=start, end=today, interval="1mo").Close

    model = ARIMA(prices, order=parameters)
    model_fit = model.fit()

    forecast_horizon = 8
    forecast = model_fit.forecast(steps=forecast_horizon)

    # dates for the forecast period
    last_date = prices.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=0), periods=forecast_horizon)
    #forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='MS')
    
    # get confidence intervals for the forecast
    #conf_int = model_fit.get_forecast(steps=forecast_horizon).conf_int()
    #lower_conf_int = conf_int.iloc[:, 0]
    #upper_conf_int = conf_int.iloc[:, 1]

    # plot forecast
    fig, ax = plt.subplots(figsize=(14, 6))
    #ax.plot(prices.loc[today:], label='Historical Prices')
    ax.plot(prices.loc['2024,01,03':], label='Historical Prices')
    ax.plot(forecast_dates, forecast, color='red', label='Forecasted Prices')
    #ax.fill_between(forecast_dates, lower_conf_int, upper_conf_int, color='pink', alpha=0.3, label='Confidence Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prices')
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color('white')


    
    # Convert Matplotlib figure to Plotly figure
    forecast_fig = tls.mpl_to_plotly(fig)
    forecast_fig.update_layout(       
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white")

    return forecast_fig


# Define layout
analysis_layout = dbc.Container([ html.Div([
    html.H1("Stock Analysis Dashboard"),
    
    # Input for stock ticker
    html.Div([
        dcc.Input(id="stock-input", value="SPY", type="text"),
        html.Button(id="submit-button", n_clicks=0, children="Enter Ticker", style={'display': 'inline-block'})
    ]),
    
    # Line and Candlestick Graphs
    html.Div([
        dcc.Graph(id="line-graph", config={'displayModeBar': False}),  # Disable mode bar for better presentation
        dcc.Graph(id="candlestick-graph", config={'displayModeBar': False})
    ], style={'width': '100%'}),
    
    # YTD Cumulative Growth Graph
    html.Div([
        dcc.Graph(id="ytd-cumulative-growth-graph")
    ]),
    
    # Correlation Matrix Heatmap
    html.Div([
        dcc.Graph(
            id='correlation-heatmap',
            figure={
                'data': [go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='Viridis')],
                'layout': go.Layout(
                    title='S&P 500 Sectors Correlation Matrix Heatmap',
                    xaxis=dict(title='Sectors'),
                    yaxis=dict(title='Sectors'),
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color="white")
            }
        )
    ]),
    
    # Forecast Plot
    html.Div(id='forecast-plot'),
    
    # Dropdown for selecting ticker
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in tickers],
        value=tickers[0],  # Default value
        style={'width': '50%', 'color': 'black'}

    ),
    
    # Forecast Plot
    html.Div(id='forecast-plot')
])], fluid=True)


# Callback to update line and candlestick graphs
@app.callback([Output("line-graph", "figure"), Output("candlestick-graph", "figure")],
              [Input("submit-button", "n_clicks")],
              [State("stock-input", "value")])
def update_fig(n_clicks, input_value):
    df = yf.Ticker(input_value)
    prices = df.history(start=start, end=today)
    
    trace_line = go.Scatter(x=prices.index,
                             y=prices.Close,
                             name="Line")
    
    trace_candle = go.Candlestick(x=prices.index,
                                  open=prices.Open,
                                  high=prices.High,
                                  low=prices.Low,
                                  close=prices.Close,
                                  name="Candle")
    
    layout_line = dict(title=f"Line Graph - {input_value}",
                       title_font=dict(color='white'),
                       autosize=True,
                       #margin=dict(l=50, r=50, t=50, b=50),  
                       xaxis=dict(
                           title_font=dict(color='white'),
                           tickfont=dict(color='white'),
                           rangeselector=dict(
                               buttons=list([
                                   dict(count=1,
                                        label="1m",
                                        step="month",
                                        stepmode="backward"),
                                   dict(count=6,
                                        label="6m",
                                        step="month",
                                        stepmode="backward"),
                                   dict(count=1,
                                        label="YTD",
                                        step="year",
                                        stepmode="todate"),
                                   dict(count=1,
                                        label="1y",
                                        step="year",
                                        stepmode="backward"),
                                   dict(step="all")
                               ]),
                                font=dict(size=12, color='white'), # Set font size and color outside buttons
                                bgcolor='rgba(68, 68, 68, 0.1)', # Set background color outside buttons
                                activecolor='rgba(68, 68, 68, 0.5)', # Background color of the active button
                                borderwidth=1,
                                bordercolor='grey',
                           ),
                           rangeslider=dict(
                               visible=True
                           )
                       
                       ),
                        yaxis=dict(
                            title='Price',  # Y-axis label
                            title_font=dict(color='white'),  # Set Y-axis title color
                            tickfont=dict(color='white'),  # Set Y-axis tick color
                        ),
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                       )

    
    layout_candle = dict(title=f"Candlestick Graph - {input_value}",
                         autosize=True,
                         #margin=dict(l=50, r=50, t=50, b=50),  
                         xaxis=dict(
                             rangeselector=dict(
                                 buttons=list([
                                     dict(count=1,
                                          label="1m",
                                          step="month",
                                          stepmode="backward"),
                                     dict(count=6,
                                          label="6m",
                                          step="month",
                                          stepmode="backward"),
                                     dict(count=1,
                                          label="YTD",
                                          step="year",
                                          stepmode="todate"),
                                     dict(count=1,
                                          label="1y",
                                          step="year",
                                          stepmode="backward"),
                                     dict(step="all")
                                 ]),
                                    font=dict(size=12, color='white'), # Set font size and color outside buttons
                                    bgcolor='rgba(68, 68, 68, 0.1)', # Set background color outside buttons
                                    activecolor='rgba(68, 68, 68, 0.5)', # Background color of the active button
                                    borderwidth=1,
                                    bordercolor='grey',

                             ),
                             rangeslider=dict(
                                 visible=True,
                             )
                         ),
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
)
    
    return {"data": [trace_line], "layout": layout_line}, {"data": [trace_candle], "layout": layout_candle}


# Callback to update YTD cumulative growth graph
@app.callback(Output("ytd-cumulative-growth-graph", "figure"),
              [Input("submit-button", "n_clicks")],
              [State("stock-input", "value")])
def update_ytd_cumulative_growth_graph(n_clicks, input_value):
    cumulative_growth = fetch_sector_etf_data()
    # sector names corresponding to tickers
    sector_names = {
        'XLC':'XLC - Communication Services',
        'XLY': 'XLY - Consumer Discretionary',
        'XLP': 'XLP - Consumer Staples',
        'XLE': 'XLE - Energy',
        'XLF': 'XLF - Financials',
        'XLV': 'XLV - Health Care',
        'XLI': 'XLI - Industrials',
        'XLB': 'XLB - Materials',
        'XLRE': 'XLRE - Real Estate',
        'XLK': 'XLK - Technology',
        'XLU': 'XLU - Utilities'
    }
    # Plot cumulative growth for YTD
    fig = go.Figure()
    for column in cumulative_growth.columns:
        sector_name = sector_names.get(column, column)
        fig.add_trace(go.Scatter(x=cumulative_growth.index, y=cumulative_growth[column], mode='lines', name=sector_name))
    fig.update_layout(title='Year-to-Date Cumulative Growth of S&P 500 Sectors', 
                      xaxis_title='Date', 
                      yaxis_title='Cumulative Growth',
                      template="plotly_dark",
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      font_color="white")
    return fig


# Callback to update forecast plot
@app.callback(Output('forecast-plot', 'children'),
              [Input('ticker-dropdown', 'value')])
def update_forecast_plot(selected_ticker):
    index = tickers.index(selected_ticker)
    selected_parameters = parameters[index]
    forecast_fig = generate_forecast_plot(selected_ticker, selected_parameters)
    forecast_fig.update_layout(
                      template="plotly_dark",
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      font_color="white")
    
    forecast_plot = dcc.Graph(figure=forecast_fig)

    return forecast_plot


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=5001)
