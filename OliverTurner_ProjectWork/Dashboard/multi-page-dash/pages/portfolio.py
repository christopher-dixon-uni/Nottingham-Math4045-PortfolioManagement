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

#styles
negative_style = {"color": "#ff0000"} #red
positive_style = {"color": "#00ff00"} #green




# Sample data: stocks and their weightings
dates = pd.date_range(start='2017-01-01', periods=60, freq='ME')

AAPL_weight_initial = 0.5 + 0.05 * np.sin(np.arange(len(dates)))
MSFT_weight_initial = np.random.rand(len(dates))
GOOGL_weight_initial = np.random.rand(len(dates))
AMZN_weight_initial = np.random.rand(len(dates))

# Stack the initial weights
weights_stack = np.vstack((AAPL_weight_initial, MSFT_weight_initial, GOOGL_weight_initial, AMZN_weight_initial))

# Normalize the weights so they sum to 1 for each period
weights_normalized = weights_stack / weights_stack.sum(axis=0)

# Assign the normalized weights to variables
AAPL_weight = weights_normalized[0]
MSFT_weight = weights_normalized[1]
GOOGL_weight = weights_normalized[2]
AMZN_weight = weights_normalized[3]

# Generate returns that fluctuate within a reasonable range
np.random.seed(42)  # For reproducibility
AAPL_returns = 0.02 + 0.05 * np.random.randn(len(dates))  # Mean return of 2% with some volatility
MSFT_returns = 0.03 + 0.04 * np.random.randn(len(dates))  # Mean return of 3% with some volatility
GOOG_returns = 0.04 + 0.03 * np.random.randn(len(dates))  # Mean return of 4% with some volatility
AMZN_returns = 0.05 + 0.02 * np.random.randn(len(dates))  # Mean return of 5% with some volatility
# Create the DataFrame
stocks_data = {
    'date': dates,
    'AAPL_weight': weights_normalized[0],
    'MSFT_weight': weights_normalized[1],
    'GOOG_weight': weights_normalized[2],
    'AMZN_weight': weights_normalized[3],
    'AAPL_returns': AAPL_returns,
    'MSFT_returns': MSFT_returns,
    'GOOG_returns': GOOG_returns,
    'AMZN_returns': AMZN_returns
}

df = pd.DataFrame(stocks_data)

#returns calc
df['monthly_return'] = df['AAPL_returns'] * df['AAPL_weight'] + df['MSFT_returns'] * df['MSFT_weight']
df['cumulative_return'] = (1 + df['monthly_return']).cumprod() - 1



#date formatting
df['date'] = pd.to_datetime(df['date'])

#start and end dates
min_date = df['date'].min()
max_date = df['date'].max()


#summary stats colummn

indicators_ptf = go.Figure()
indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = 0.15,
    number = {'suffix': " %"},
    title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
    delta = {'position': "bottom", 'reference': 0.2, 'relative': False},
    domain = {'row': 0, 'column': 0}))

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = 0.2,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
    delta = {'position': "bottom", 'reference': 0.25, 'relative': False},
    domain = {'row': 1, 'column': 0}))

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = 0.3,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
    delta = {'position': "bottom", 'reference': 0.25, 'relative': False},
    domain = {'row': 2, 'column': 0}))

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = 2,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
    delta = {'position': "bottom", 'reference': 1.5, 'relative': False},
    domain = {'row': 3, 'column': 1}))

indicators_ptf.update_layout(
    grid = {'rows': 4, 'columns': 1, 'pattern': "independent"},
    margin=dict(l=50, r=50, t=30, b=30)
)

indicators_sp500 = go.Figure()
indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = 0.2,
    number = {'suffix': " %"},
    title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
    domain = {'row': 0, 'column': 0}))

indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = 0.25,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
    domain = {'row': 1, 'column': 0}))

indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = 0.25,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
    domain = {'row': 2, 'column': 0}))

indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = 1.5,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
    domain = {'row': 3, 'column': 1}))

indicators_sp500.update_layout(
    grid = {'rows': 4, 'columns': 1, 'pattern': "independent"},
    margin=dict(l=50, r=50, t=30, b=30)
)

# Define the layout for this page
layout = dbc.Container([
    #Title
    dbc.Row([
        dbc.Col(html.H1("Portfolio", className="text-center"), width=12),
    ], className="rounded-box"),

    #date range
    dbc.Row([
        dbc.Col(html.H2("Overall Returns", className="text-center"), width=12),
        dbc.Col([
            html.Div("Select Date Range:", className="mb-2"),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=min_date,
                end_date=max_date,
                display_format='YYYY-MM-DD',  # Display format of the date
                className="mb-3"
            ),
    ], md=6),

    #cumulative return graoh
    dbc.Col(dcc.Graph(id='overall-returns-graph'), width={'size': 8, 'offset': 0, 'order': 1}),


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
            ], width={'size': 2, 'offset': 0, 'order': 3})
,
    
    
    ],className="rounded-box"),



    #Single Month
    dbc.Row([
        dbc.Col(html.H2("Monthly Statistics", className="text-center"), width=12),
        dbc.Col([
            html.Div("Select Month:", className="mb-2"),
            dcc.DatePickerSingle(
                id='date-picker-single',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                initial_visible_month=min_date,
                date=max_date,
                display_format='YYYY-MM-DD'
            ),

    ], md=6),

    dcc.Graph(id="monthly-returns-graph"),
    dcc.Graph(id="monthly-weights-graph"),

    ],className="rounded-box"),

], fluid=True)



# Callback to update page
@app.callback(
    Output('overall-returns-graph', 'figure'),
    [
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_returns_graph(start_date, end_date):
    
    # Filter based on selected date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    filtered_df = df.loc[mask]
    
    # Plotting the cumulative returns against the date
    fig = px.line(
        filtered_df,
        x='date',
        y='cumulative_return',
        title='Cumulative Returns Over Time',
        labels={'cumulative_returns': 'Cumulative Returns', 'date': 'Date'},
        height=700
    )
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative Returns', xaxis=dict(tickformat='%Y-%m-%d'))
    return fig


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
    xaxis_tickangle=-45
)

    
    return fig

