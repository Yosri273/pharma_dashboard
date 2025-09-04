import dash
from dash import dcc, html

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.DatePickerRange(
        id='test-date-picker',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
])

if __name__ == '__main__':
    app.run(debug=True)