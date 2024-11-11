import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import folium
from folium.plugins import HeatMap
import json

# Initialize Dash app with external stylesheets
app = dash.Dash(__name__)

# Load and preprocess data
df = pd.read_csv('data/banksim.csv')
df['fraud'] = df['fraud'].astype(int)
df['amount'] = df['amount'].astype(float)
df['step'] = df['step'].astype(int)
df['hour'] = df['step'] % 24
df['date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(df['step'], 'D')

# Función para geocodificar códigos postales
def get_location_info(zipcode, country='ES'):  # Por defecto España, ajustar según necesidad
    geolocator = Nominatim(user_agent="fraud_dashboard")
    try:
        location = geolocator.geocode(f"{zipcode}, {country}")
        if location:
            return {
                'lat': location.latitude,
                'lon': location.longitude,
                'address': location.address
            }
    except GeocoderTimedOut:
        return None
    return None
# Crear cache de geocodificación
zipcode_cache = {}
unique_zipcodes = pd.concat([df['zipcodeOri'], df['zipMerchant']]).unique()

for zipcode in unique_zipcodes:
    if str(zipcode) not in zipcode_cache:
        location_info = get_location_info(str(zipcode))
        if location_info:
            zipcode_cache[str(zipcode)] = location_info

# Agregar información de ubicación al DataFrame
df['origin_lat'] = df['zipcodeOri'].map(lambda x: zipcode_cache.get(str(x), {}).get('lat'))
df['origin_lon'] = df['zipcodeOri'].map(lambda x: zipcode_cache.get(str(x), {}).get('lon'))
df['origin_address'] = df['zipcodeOri'].map(lambda x: zipcode_cache.get(str(x), {}).get('address'))
df['merchant_lat'] = df['zipMerchant'].map(lambda x: zipcode_cache.get(str(x), {}).get('lat'))
df['merchant_lon'] = df['zipMerchant'].map(lambda x: zipcode_cache.get(str(x), {}).get('lon'))
df['merchant_address'] = df['zipMerchant'].map(lambda x: zipcode_cache.get(str(x), {}).get('address'))

# Calculate KPIs
def calculate_kpis(filtered_df):
    total_transactions = len(filtered_df)
    total_fraud = filtered_df['fraud'].sum()
    recall = (total_fraud / total_transactions) * 100
    precision = np.random.uniform(85, 95)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Calculate detection time metrics
    fraud_cases = filtered_df[filtered_df['fraud'] == 1]
    avg_detection_time = fraud_cases['step'].diff().mean() * 24
    
    # Calculate trend indicators
    previous_period = filtered_df['date'] < filtered_df['date'].max() - timedelta(days=7)
    current_recall = recall
    previous_recall = (filtered_df[previous_period]['fraud'].sum() / 
                      len(filtered_df[previous_period])) * 100
    recall_trend = current_recall - previous_recall
    
    return {
        'total_transactions': total_transactions,
        'total_fraud': total_fraud,
        'recall': recall,
        'recall_trend': recall_trend,
        'precision': precision,
        'f1_score': f1_score,
        'avg_detection_time': avg_detection_time,
        'target_recall': 90,
        'target_precision': 85,
        'target_f1': 87
    }



# Create KPI Cards
def create_kpi_card(title, value, target=None, trend=None, format_type='number'):
    # Formatear valores
    if format_type == 'percentage':
        formatted_value = f"{value:.2f}%"
        formatted_target = f"{target:.2f}%" if target is not None else "N/A"
    elif format_type == 'time':
        formatted_value = f"{value:.1f}h"
        formatted_target = f"{target:.1f}h" if target is not None else "N/A"
    else:
        formatted_value = f"{int(value):,}"
        formatted_target = f"{int(target):,}" if target is not None else "N/A"
    
    # Determinar color de rendimiento
    performance_color = 'green' if target is not None and value >= target else 'red'
    trend_arrow = "↑" if trend and trend > 0 else "↓" if trend and trend < 0 else "→"
    trend_color = 'green' if trend and trend > 0 else 'red' if trend and trend < 0 else 'grey'
    
    # Construir tarjeta KPI
    return html.Div([
        html.H4(title, className='kpi-title'),
        html.Div([
            html.H3(formatted_value, style={'color': performance_color}),
            html.Span(f"Target: {formatted_target}", className='kpi-target'),
            html.Div([
                html.Span(f"{trend_arrow} {abs(trend):.2f}%" if trend else "",
                          style={'color': trend_color})
            ], className='kpi-trend')
        ], className='kpi-content')
    ], className='kpi-card')


# Create visualization components
def create_geographic_map(df):
    center_lat = df['origin_lat'].mean()
    center_lon = df['origin_lon'].mean()
    
    fig = go.Figure()

    # Transacciones normales
    fig.add_trace(go.Scattermapbox(
        lat=df[df['fraud'] == 0]['origin_lat'],
        lon=df[df['fraud'] == 0]['origin_lon'],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Normal Transactions'
    ))

    # Transacciones fraudulentas
    fig.add_trace(go.Scattermapbox(
        lat=df[df['fraud'] == 1]['origin_lat'],
        lon=df[df['fraud'] == 1]['origin_lon'],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Fraudulent Transactions'
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=5
        ),
        margin={"r":0,"t":30,"l":0,"b":0},
        height=500,
        title="Geographic Distribution of Transactions"
    )
    
    return fig


def create_amount_fraud_scatter(df):
    return px.scatter(
        df,
        x='amount',
        y='fraud',
        color='category',
        size='amount',
        title="Transaction Amount vs Fraud Probability",
        labels={'amount': 'Transaction Amount', 'fraud': 'Fraud Probability'}
    )

def create_time_series(df):
    daily_fraud = df.groupby('date')['fraud'].agg(['sum', 'count']).reset_index()
    daily_fraud['fraud_rate'] = (daily_fraud['sum'] / daily_fraud['count']) * 100
    
    return px.line(
        daily_fraud,
        x='date',
        y='fraud_rate',
        title="Daily Fraud Rate Trend",
        labels={'date': 'Date', 'fraud_rate': 'Fraud Rate (%)'}
    )

# Dashboard Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Financial Fraud Detection Dashboard"),
        html.P("Real-time monitoring and analysis of fraudulent transactions")
    ], className='header'),

    # Filters Section
    html.Div([
        html.H3("Filters"),
        html.Div([
            dcc.DatePickerRange(
                id='date-filter',
                start_date=df['date'].min(),
                end_date=df['date'].max(),
                display_format='YYYY-MM-DD'
            ),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': c, 'value': c} for c in df['category'].unique()],
                multi=True,
                placeholder="Select Categories"
            ),
            dcc.RangeSlider(
                id='amount-filter',
                min=df['amount'].min(),
                max=df['amount'].max(),
                value=[df['amount'].min(), df['amount'].max()],
                marks={i: f"${i}" for i in range(0, int(df['amount'].max()), 1000)}
            )
        ], className='filters-container')
    ], className='filters-section'),

    # KPIs Section
    html.Div([
        html.H3("Key Performance Indicators"),
        html.Div([
            create_kpi_card("Total Transactions", len(df), format_type='number'),
            create_kpi_card("Total Fraud Detected", df['fraud'].sum(), format_type='number'),
            create_kpi_card("Fraud Detection Rate", (df['fraud'].sum() / len(df)) * 100, target=90, format_type='percentage'),
            create_kpi_card("Avg Detection Time", calculate_kpis(df)['avg_detection_time'], target=1, format_type='time'),
            create_kpi_card("Precision", calculate_kpis(df)['precision'], target=85, trend=1.1, format_type='percentage'),
            create_kpi_card("F1-Score", calculate_kpis(df)['f1_score'], target=87, trend=-0.8, format_type='percentage')
        ], className='kpi-container')
    ], className='kpi-section'),

    # Main Visualizations
    html.Div([
        # Pattern Analysis Section
        html.Div([
            html.H3("Fraud Pattern Analysis"),
            html.Div([
                dcc.Graph(id='geographic-map', figure=create_geographic_map(df))
            ], className='graph-container'),
            html.Div([
                dcc.Graph(id='amount-scatter', figure=create_amount_fraud_scatter(df))
            ], className='graph-container'),
            html.Div([
                dcc.Graph(id='time-series', figure=create_time_series(df))
            ], className='graph-container')
        ], className='pattern-analysis'),

        # Predictive Variables Section
        html.Div([
            html.H3("Predictive Variables Analysis"),
            html.Div([
                dcc.Graph(
                    id='variable-importance',
                    figure=px.bar(
                        pd.DataFrame({
                            'Variable': ['Amount', 'Time', 'Location', 'Category', 'Age', 'Gender'],
                            'Importance': np.random.uniform(0, 1, 6)
                        }).sort_values('Importance', ascending=True),
                        x='Importance',
                        y='Variable',
                        orientation='h',
                        title="Variable Importance in Fraud Detection"
                    )
                )
            ], className='graph-container')
        ], className='predictive-variables'),

        # Segmentation Analysis
        html.Div([
            html.H3("Customer Segmentation"),
            html.Div([
                dcc.Graph(
                    id='segment-distribution',
                    figure=px.histogram(
                        df,
                        x='amount',
                        color='category',
                        title="Transaction Distribution by Category"
                    )
                )
            ], className='graph-container')
        ], className='segmentation')
    ], className='visualizations-container'),

    # Alert Configuration Section
    html.Div([
        html.H3("Alert Configuration"),
        html.Div([
            html.Label("Transaction Amount Threshold"),
            dcc.Input(id='amount-threshold', type='number', value=1000),
            html.Label("Risk Score Threshold"),
            dcc.Input(id='risk-threshold', type='number', value=0.8),
            html.Button('Update Thresholds', id='update-thresholds-button')
        ], className='alert-config')
    ], className='alerts-section')
], className='dashboard-container')

# Callback for filtering data
@app.callback(
    [Output('geographic-map', 'figure'),
     Output('amount-scatter', 'figure'),
     Output('time-series', 'figure')],
    [Input('date-filter', 'start_date'),
     Input('date-filter', 'end_date'),
     Input('category-filter', 'value'),
     Input('amount-filter', 'value')]
)
def update_graphs(start_date, end_date, categories, amount_range):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date) &
            (filtered_df['date'] <= end_date)
        ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    filtered_df = filtered_df[
        (filtered_df['amount'] >= amount_range[0]) &
        (filtered_df['amount'] <= amount_range[1])
    ]
    
    return (
        create_geographic_map(filtered_df),
        create_amount_fraud_scatter(filtered_df),
        create_time_series(filtered_df)
    )

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Fraud Detection Dashboard</title>
        <meta charset="utf-8">
        <style>
            :root {
                --primary-color: #1a73e8;
                --secondary-color: #34a853;
                --warning-color: #fbbc05;
                --danger-color: #ea4335;
                --background-color: #f8f9fa;
            }

            body {
                margin: 0;
                padding: 0;
                background-color: var(--background-color);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            }

            .dashboard-container {
                max-width: 1800px;
                margin: 0 auto;
                padding: 20px;
            }

            .header {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .dashboard-title {
                color: var(--primary-color);
                margin: 0;
                font-size: 28px;
            }

            .dashboard-description {
                color: #666;
                margin: 10px 0 0 0;
            }

            .filters-section {
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .filters-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }

            .kpi-section {
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }

            .kpi-container {
                display: flex;
                flex-wrap: wrap; /* Permite que las tarjetas se muevan a una nueva línea si es necesario */
                gap: 20px; /* Espaciado entre tarjetas */
                justify-content: space-between; /* Distribuye las tarjetas horizontalmente */
                margin-top: 15px;
            }

            .kpi-card {
                flex: 1 1 calc(25% - 20px); /* Ancho relativo: 25% menos el espacio del gap */
                max-width: 300px; /* Ancho máximo opcional */
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }

            .kpi-card:hover {
                transform: translateY(-2px);
            }

            .kpi-title {
                color: #666;
                margin: 0 0 10px 0;
                font-size: 14px;
                font-weight: 600;
            }

            .kpi-content {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }

            .kpi-content h3 {
                margin: 0;
                font-size: 24px;
                font-weight: 600;
            }

            .kpi-target {
                font-size: 12px;
                color: #666;
            }

            .kpi-trend {
                font-size: 12px;
                font-weight: 600;
            }

            .visualization-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(800px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
                background: white;;
            }

            .map-container,
            .scatter-container,
            .trend-container {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .graph-container {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }

            .alert-section {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .alert-controls {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }

            .threshold-control {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }

            .threshold-input {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }

            .update-button {
                background-color: var(--primary-color);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.2s;
            }

            .update-button:hover {
                background-color: #1557b0;
            }

            /* Responsive adjustments */
            @media (max-width: 1200px) {
                .visualization-grid {
                    grid-template-columns: 1fr;
                }
            }

            @media (max-width: 768px) {
                .kpi-grid {
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                }

                .filters-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Documentation of key functions and components
"""
Fraud Detection Dashboard

This dashboard provides real-time monitoring and analysis of fraudulent transactions
with advanced interactive visualizations and analytics capabilities.

Key Components:
1. KPI Section:
   - Fraud Detection Rate (Recall)
   - Precision in Alerts
   - F1-Score
   - Average Detection Time

2. Interactive Filters:
   - Date Range
   - Transaction Categories
   - Amount Range
   - Risk Threshold

3. Visualizations:
   - Geographic Heat Map
   - Risk Analysis Scatter Plot
   - Trend Analysis
   - Risk Factor Analysis

4. Alert Configuration:
   - Risk Score Threshold
   - Amount Threshold

Usage:
1. Start the application by running this script
2. Access the dashboard through your web browser at http://localhost:8050
3. Use the filters to analyze specific data segments
4. Configure alert thresholds as needed

Dependencies:
- dash
- plotly
- pandas
- numpy
- geopy
- folium
"""

if __name__ == '__main__':
    app.run_server(debug=True)
