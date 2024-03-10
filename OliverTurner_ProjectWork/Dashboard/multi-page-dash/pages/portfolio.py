from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from app_instance import app  # Import the Dash app instance
from datetime import date
import numpy as np
from dash import dash_table
import plotly.graph_objects as go
from optimisation_output import return_assets_weights
import warnings
import yfinance as yf
from datetime import datetime, timedelta
import optimise_get_returns as opt
import dash_daq as daq


#styles
negative_style = {"color": "#ff0000"} #red
positive_style = {"color": "#00ff00"} #green




# Sample data: stocks and their weightings
dates = pd.date_range(start='2022-12-01', periods=13, freq='M', )


#start and end dates
min_date = dates.min()
max_date = dates.max()

def calculate_return(data, days):
    data['cumulative_return'] = (1 + data['Adj Close'].pct_change()).cumprod() - 1

    return data['cumulative_return'].iloc[days]


def get_stock_data(start_date, end_date, data=None):
   #get s&p 500 performance data
    # Define the ticker symbol for the S&P 500 ETF (SPY)

    # Download the data
    if data is None:
        data = yf.download('SPY', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        

    returns_7_days = calculate_return(data, 7)*100
    returns_15_days = calculate_return(data, 15)*100
    returns_30_days = calculate_return(data, 30)*100
    returns_200_days = calculate_return(data, 200)*100

    return returns_7_days, returns_15_days, returns_30_days, returns_200_days, data

spReturns_7_days, spReturns_15_days, spReturns_30_days, spReturns_200_days, sp500_data = get_stock_data(min_date, end_date=max_date)

data = pd.read_csv('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/2023_returns.csv')
#calculate 7, 15, 30, 200 day returns in portfolio using cumulative returns column
#add 4 days on as the first 4 days are NaN
portReturns_7_days = (data['Portfolio Cumulative Returns'].iloc[11])*100
portReturns_15_days = (data['Portfolio Cumulative Returns'].iloc[19])*100
portReturns_30_days = (data['Portfolio Cumulative Returns'].iloc[34])*100
portReturns_200_days = (data['Portfolio Cumulative Returns'].iloc[204])*100


sp500_data['cumulative_return'] = (1 + sp500_data['Close'].pct_change()).cumprod() - 1


def sumamry_stats_columns():
    #Portfolio stats colummn

    indicators_ptf = go.Figure()
    indicators_ptf.add_trace(go.Indicator(
        mode = "number+delta",
        value = portReturns_7_days,
        number = {'suffix': " %"},
        title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
        delta = {'position': "bottom", 'reference': spReturns_7_days, 'relative': False},
        domain = {'row': 0, 'column': 0}))

    indicators_ptf.add_trace(go.Indicator(
        mode = "number+delta",
        value = portReturns_15_days,
        number = {'suffix': " %"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
        delta = {'position': "bottom", 'reference': spReturns_15_days, 'relative': False},
        domain = {'row': 1, 'column': 0}))

    indicators_ptf.add_trace(go.Indicator(
        mode = "number+delta",
        value = portReturns_30_days,
        number = {'suffix': " %"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
        delta = {'position': "bottom", 'reference': spReturns_30_days, 'relative': False},
        domain = {'row': 2, 'column': 0}))

    indicators_ptf.add_trace(go.Indicator(
        mode = "number+delta",
        value = portReturns_200_days,
        number = {'suffix': " %"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
        delta = {'position': "bottom", 'reference': spReturns_200_days, 'relative': False},
        domain = {'row': 3, 'column': 1}))

    indicators_ptf.update_layout(
        grid = {'rows': 4, 'columns': 1, 'pattern': "independent"},
        margin=dict(l=50, r=50, t=30, b=30),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white"
        
    )

    #s&P500 stats column

    indicators_sp500 = go.Figure()
    indicators_sp500.add_trace(go.Indicator(
        mode = "number+delta",
        value = spReturns_7_days,
        number = {'suffix': " %"},
        title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
        domain = {'row': 0, 'column': 0}))

    indicators_sp500.add_trace(go.Indicator(
        mode = "number+delta",
        value = spReturns_15_days,
        number = {'suffix': " %"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
        domain = {'row': 1, 'column': 0}))

    indicators_sp500.add_trace(go.Indicator(
        mode = "number+delta",
        value = spReturns_30_days,
        number = {'suffix': " %"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
        domain = {'row': 2, 'column': 0}))

    indicators_sp500.add_trace(go.Indicator(
        mode = "number+delta",
        value = spReturns_200_days,
        number = {'suffix': " %"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
        domain = {'row': 3, 'column': 1}))

    indicators_sp500.update_layout(
        grid = {'rows': 4, 'columns': 1, 'pattern': "independent"},
        margin=dict(l=50, r=50, t=30, b=30),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white"

    )
    return indicators_ptf, indicators_sp500

indicators_ptf, indicators_sp500 = sumamry_stats_columns()


#treemap
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sector_weights_df = opt.get_recent_weights_sectors()
print('pages/portfolio.py: asset weights returned')

if __name__ == "__main__":
    print(sector_weights_df)

#treemap cannot work for negative weights
positive_sector_weights_df = sector_weights_df[sector_weights_df['Weight'] > 0]

treemap_fig = px.treemap(positive_sector_weights_df,
                         path=[px.Constant("Portfolio"), 'Sector', 'Ticker'],
                         values='Weight',
                         title='Current Portfolio Weightings by Sector and Stock',
                         color='Weight',
                         hover_data={
                            'Ticker': True,
                            'Long Name': True,
                            'Sector': True,
                            'Weight': ':.2%'  
                         }, 
                         color_continuous_scale='Algae')

treemap_fig.update_traces(textinfo='label+percent parent',
                  textfont=dict(color='white', size=26))

treemap_fig.update_traces(hovertemplate = "<b>%{customdata[0]}</b><br>" + \
                "Long Name: %{customdata[1]}<br>" + \
                "Sector: %{customdata[2]}<br>" + \
                "Optimal Weights: %{customdata[3]:.2%}<extra></extra>"
)

treemap_fig.update_layout(
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color="white"
)

treemap_graph_component = dcc.Graph(
    figure=treemap_fig,
    id='sector-treemap'
)



#overall returns graph
overall_returns_graph = opt.plot_returns(pd.read_csv('./OliverTurner_ProjectWork/Dashboard/multi-page-dash/assets/2023_returns.csv'))
#add s&p500 cumulative returns
overall_returns_graph.add_scatter(x=sp500_data.index, y=sp500_data['cumulative_return'], mode='lines', name='S&P500', line=dict(color='green', width=2))


overall_returns_graph.update_layout(xaxis_title='Date',
                    yaxis_title='Cumulative Returns',
                    xaxis=dict(tickformat='%Y-%m-%d'),
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color="white")

efficient_frontier_fig = opt.plot_efficient_frontier()

# Define the layout for this page
layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col(html.H1("Portfolio", className="text-center"), width=12),
        dbc.Col(treemap_graph_component,width=12,)
    ]),
    # Date range
    dbc.Row([
        dbc.Col(html.H2("Overall Returns", className="text-center"), width=12),
        dbc.Col(dcc.Graph(id='overall-returns-graph', figure=overall_returns_graph), width={'size': 8, 'offset': 0, 'order': 1}),
        dbc.Col([  # second column on second row
                html.H5('Portfolio', className='text-center'),
                dcc.Graph(id='indicators-ptf',
                        figure=indicators_ptf,
                        style={'height':550}),
                html.Hr()
                ], width={'size': 2, 'offset': 0, 'order': 2}),  # width second column on second row
                dbc.Col([  # third column on second row
                html.H5('S&P500', className='text-center'),
                dcc.Graph(id='indicators-sp',
                        figure=indicators_sp500,
                        style={'height':550}),
                html.Hr()
                ], width={'size': 2, 'offset': 0, 'order': 3}),
    ]),
    dbc.Row([dcc.Graph(id='efficient-frontier-graph', figure = efficient_frontier_fig),]),
    
], fluid=True)




'''
@app.callback(
    Output('monthly-returns-graph', 'figure'),
    [Input('date-picker-single', 'date')]
)
def update_monthly_statistics(selected_date):
    if not selected_date:
        return px.line(title="Please select a month")
    
    # Convert selected_date to datetime and extract year and month
    selected_month = pd.to_datetime(selected_date).replace(day=1)
    

    filtered_df = df[df['date'].dt.to_period('M') == selected_month.to_period('M')]
    if filtered_df.empty:
        return px.imshow([], title="No data available for selected month")

    heatmap_data = {'Metric': ['% Return']}
    for column in filtered_df.columns:
        if '_returns' in column:
            stock_symbol = column.replace('_returns', '')
            # Assuming the returns are already in decimal form, convert to percentage
            heatmap_data[stock_symbol] = [round(filtered_df.iloc[0][column], 4) * 100]

    heatmap_df = pd.DataFrame(heatmap_data).set_index('Metric')

    fig = px.imshow(
        heatmap_df,
        labels=dict(x="Stock", y="Metric", color="Value"),
        x=heatmap_df.columns,
        y=heatmap_df.index,
        text_auto=True,
        color_continuous_scale='blues',
        aspect="auto")

    fig.update_xaxes(side="top")
    fig.update_layout(
        title=f"Monthly Returns for {selected_month.strftime('%B %Y')}",
        xaxis_title='Stock',
        yaxis_title='Metric',
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white"
    )


    
    return fig

@app.callback(
    Output('monthly-weights-graph', 'figure'),
    [Input('date-picker-single', 'date')]
)
def update_weightings_graph(selected_date):
    if not selected_date:
        return px.bar(title="Please select a month")
    
    selected_month = pd.to_datetime(selected_date).replace(day=1)
    filtered_df = df[df['date'].dt.to_period('M') == selected_month.to_period('M')]
    
    if filtered_df.empty:
        return px.bar([], title="No data available for selected month")

    # Dynamically construct the data for the bar chart
    weightings_data = []
    for column in filtered_df.columns:
        if '_weight' in column:
            stock_symbol = column.replace('_weight', '')
            weight = filtered_df.iloc[0][column]
            weightings_data.append({'Stock': stock_symbol, 'Weighting': weight})

    weightings_df = pd.DataFrame(weightings_data)

    # Create the bar chart
    fig = px.bar(
        weightings_df,
        x='Stock',
        y='Weighting',
        text='Weighting',
        title=f"Stock Weightings for {selected_month.strftime('%B %Y')}"
    )

    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(
    xaxis_title='Stock',
    yaxis_title='Weighting',
    yaxis=dict(tickformat='.0%'),
    autosize=True,
    margin=dict(l=50, r=50, t=50, b=100),  # Adjust margins to ensure labels fit
    xaxis_tickangle=-45,
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color="white"
    
)
    

    
    return fig

'''