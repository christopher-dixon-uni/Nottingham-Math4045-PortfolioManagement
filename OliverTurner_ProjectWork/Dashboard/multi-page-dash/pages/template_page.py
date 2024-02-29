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




# Define the layout for this page
layout = dbc.Container([
    #Title
    dbc.Row([
        dbc.Col(html.H1("TITLE", className="text-center"), width=12),
    ], className="rounded-box"),

    #date range
    dbc.Row([ #row 1
        
        dbc.Col(html.H2("HEADER", className="text-center"), width=12), # first column within row 1

        #cumulative return graoh
        dbc.Col(dcc.Graph(id='overall-returns-graph')), # second column of row 1, #dcc.Graph refers to a plot

    
    ],className="rounded-box"),


])


# Callback gets data every time the date picker is changed
@app.callback(
    Output('overall-returns-graph', 'figure'),
    [
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_returns_graph(start_date, end_date): #define the graph here
    
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

