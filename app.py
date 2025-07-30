import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Part 1: Data Generation and Model Training ---

def generate_and_prepare_data():
    """Generates data in memory, trains the model, and prepares all necessary data structures."""
    logging.info("Generating synthetic data...")
    num_rows = 2000
    locations = ['Maharashtra', 'Uttar Pradesh', 'Tamil Nadu', 'West Bengal', 'Karnataka', 'Rajasthan', 'Gujarat']
    start_date, end_date = '2024-01-01', '2024-12-31'
    start, end = datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d')
    date_range_days = (end - start).days

    data = {
        'date': [start + timedelta(days=random.randint(0, date_range_days)) for _ in range(num_rows)],
        'location': random.choices(locations, k=num_rows),
        'age_group': random.choices(['18-40', '41-60', '61+'], weights=[0.3, 0.4, 0.3], k=num_rows),
        'gender': random.choices(['Male', 'Female'], weights=[0.45, 0.55], k=num_rows),
        'severity': np.random.randint(1, 11, size=num_rows),
        'cases': np.random.randint(20, 750, size=num_rows)
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    logging.info("Data generation complete.")

    logging.info("Training model and preparing data...")
    
    # Feature Engineering & Encoding
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    
    encoders = {}
    for col in ['location', 'age_group', 'gender']:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        encoders[col] = le

    features = ['year', 'month', 'dayofyear', 'location_encoded', 'age_group_encoded', 'gender_encoded', 'severity']
    target = 'cases'
    
    X = df[features]
    y = df[target]

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    logging.info("Model training complete.")
    
    return rf_model, encoders, df

# --- Pre-computation at Startup ---
model, data_encoders, processed_data = generate_and_prepare_data()

LOCATIONS = sorted(processed_data['location'].unique().tolist())
AGE_GROUPS = sorted(processed_data['age_group'].unique().tolist())
GENDERS = sorted(processed_data['gender'].unique().tolist())


# --- Part 2: Dash Application Layout ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server # Expose the server variable for Gunicorn

app.layout = html.Div(style={'padding': '20px', 'backgroundColor': '#f4f6f9'}, children=[
    html.Div(style={'textAlign': 'center', 'marginBottom': '40px'}, children=[
        html.H1("Arthritis Case Prediction & Analysis Dashboard", style={'color': '#1a5276', 'fontWeight': 'bold'}),
        html.P("Explore historical data and use the predictive model for on-demand forecasting.", style={'color': '#5d6d7e'})
    ]),
    html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px #ccc'}, children=[
                html.H4("On-Demand Prediction", style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Label("Location"),
                dcc.Dropdown(id='predict-location', options=LOCATIONS, value=LOCATIONS[0]),
                html.Br(),
                html.Label("Age Group"),
                dcc.Dropdown(id='predict-age', options=AGE_GROUPS, value=AGE_GROUPS[0]),
                html.Br(),
                html.Label("Gender"),
                dcc.Dropdown(id='predict-gender', options=GENDERS, value=GENDERS[0]),
                html.Br(),
                html.Label("Severity (1-10)"),
                dcc.Slider(id='predict-severity', min=1, max=10, step=1, value=5, marks={i: str(i) for i in range(1, 11)}),
                html.Br(),
                html.Button('Predict Cases', id='predict-button', n_clicks=0, style={'width': '100%', 'backgroundColor': '#1a5276', 'color': 'white'}),
                html.Div(id='prediction-output', style={'marginTop': '20px', 'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold'})
            ])
        ]),
        html.Div(style={'flex': '3', 'minWidth': '600px'}, children=[
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px #ccc'}, children=[
                html.H4("Explore Historical Data", style={'textAlign': 'center'}),
                html.Label("Select Location to Filter Charts"),
                dcc.Dropdown(id='filter-location', options=[{'label': 'All Locations', 'value': 'All'}] + [{'label': loc, 'value': loc} for loc in LOCATIONS], value='All', style={'marginBottom': '10px'}),
                html.Div(id='metrics-container', style={'display': 'flex', 'gap': '20px', 'justify-content': 'center', 'marginBottom': '20px'}),
                dcc.Tabs(id='charts-tabs', value='tab-demographics', children=[
                    dcc.Tab(label='Demographics', value='tab-demographics', children=[
                        html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '20px'}, children=[
                            dcc.Graph(id='age-barchart', style={'flex': '1'}),
                            dcc.Graph(id='gender-piechart', style={'flex': '1'})
                        ])
                    ]),
                    dcc.Tab(label='Severity Analysis', value='tab-severity', children=[
                        dcc.Graph(id='severity-scatterplot', style={'marginTop': '20px'})
                    ]),
                ]),
            ])
        ])
    ])
])

# --- Part 3: Callbacks ---
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('predict-location', 'value'), State('predict-age', 'value'), State('predict-gender', 'value'), State('predict-severity', 'value')]
)
def update_prediction(n_clicks, location, age, gender, severity):
    if n_clicks == 0: return ""
    today = datetime.now()
    input_data = {
        'year': [today.year], 'month': [today.month], 'dayofyear': [today.timetuple().tm_yday],
        'location_encoded': [data_encoders['location'].transform([location])[0]],
        'age_group_encoded': [data_encoders['age_group'].transform([age])[0]],
        'gender_encoded': [data_encoders['gender'].transform([gender])[0]],
        'severity': [severity]
    }
    input_df = pd.DataFrame(input_data)
    predicted_cases = model.predict(input_df)[0]
    return f"Predicted Cases: {int(predicted_cases)}"

@app.callback(
    [Output('metrics-container', 'children'), Output('age-barchart', 'figure'), Output('gender-piechart', 'figure'), Output('severity-scatterplot', 'figure')],
    Input('filter-location', 'value')
)
def update_charts_and_metrics(location):
    filtered_df = processed_data if location == 'All' else processed_data[processed_data['location'] == location]
    
    total_cases = filtered_df['cases'].sum()
    avg_cases_day = filtered_df.groupby('date')['cases'].sum().mean()
    location_risk = processed_data.groupby('location')['cases'].sum().idxmax()
    
    metric_style = {'flex': '1', 'textAlign': 'center', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'}
    metrics = html.Div(style={'display': 'flex', 'gap': '20px', 'width': '100%'}, children=[
        html.Div([html.H5("Total Cases"), html.P(f"{total_cases:,}")], style=metric_style),
        html.Div([html.H5("Avg Cases/Day"), html.P(f"{avg_cases_day:,.2f}")], style=metric_style),
        html.Div([html.H5("Highest Risk Location"), html.P(location_risk)], style=metric_style),
    ])

    age_df = filtered_df.groupby('age_group')['cases'].sum().reset_index()
    bar_fig = px.bar(age_df, x='age_group', y='cases', title='Cases by Age Group', labels={'age_group': 'Age Group', 'cases': 'Total Cases'}, color_discrete_sequence=['#2980b9'])
    
    gender_df = filtered_df.groupby('gender')['cases'].sum().reset_index()
    pie_fig = px.pie(gender_df, names='gender', values='cases', title='Cases by Gender', color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'})
    
    scatter_fig = px.scatter(filtered_df, x='severity', y='cases', title='Severity vs. Number of Cases', labels={'severity': 'Severity Score', 'cases': 'Number of Cases'}, trendline='ols', trendline_color_override='red')

    for fig in [bar_fig, pie_fig, scatter_fig]:
        fig.update_layout(title_x=0.5, margin=dict(l=40, r=20, t=40, b=20))

    return metrics, bar_fig, pie_fig, scatter_fig

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=8050)
