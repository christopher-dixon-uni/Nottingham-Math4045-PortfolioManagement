import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.io as pio
from pages import portfolio
from app_instance import app  

pio.templates.default = "plotly_white"

server = app.server


app.layout = html.Div([
    portfolio.layout
])

if __name__ == '__main__':
    app.run_server(debug=True)
