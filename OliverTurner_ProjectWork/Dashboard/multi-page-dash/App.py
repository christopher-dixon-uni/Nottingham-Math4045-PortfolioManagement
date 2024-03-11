import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.io as pio
from pages import portfolio
from app_instance import app  
import dash_bootstrap_components as dbc
from dash import html
from pages.portfolio import portfolio_layout
from pages.stock_analysis import analysis_layout

pio.templates.default = "plotly_white"

server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Nav([
        dcc.Link('Portfolio', href='/portfolio'),
        html.Span(' | '),
        dcc.Link('Stock Analysis', href='/stock_analysis'),
        html.Span(' | '),
    ]),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/portfolio':
        return portfolio_layout
    elif pathname == '/stock_analysis':
        return analysis_layout
    else:
        return '404 Page not found'

if __name__ == '__main__':
    print('Running app.py')
    print('Please go to http://localhost:8050/ to view the dashboard.')
    app.run_server(host='localhost', port=8050, debug=True)