import dash
import dash_bootstrap_components as dbc

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
GOOGLE_FONT_URL = "https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap"
DARK_THEME = dbc.themes.CYBORG

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.DARKLY,
                                      GOOGLE_FONT_URL,
                                      FONT_AWESOME],
                suppress_callback_exceptions=True)




#app = dash.Dash(__name__, suppress_callback_exceptions=True)
