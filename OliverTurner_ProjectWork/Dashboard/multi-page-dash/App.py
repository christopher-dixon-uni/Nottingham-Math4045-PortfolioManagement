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


pio.templates.default = "plotly_white"

server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Nav([
        dcc.Link('Portfolio', href='/portfolio')
,  # Corrected href
    ]),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/portfolio':  # Corrected pathname check
        return portfolio_layout
    else:
        return '404 Page not found'

if __name__ == '__main__':
    print('Running app.py')
    print('Please go to http://localhost:8050/ to view the dashboard.')
    app.run_server(host='localhost', port=8050, debug=True)