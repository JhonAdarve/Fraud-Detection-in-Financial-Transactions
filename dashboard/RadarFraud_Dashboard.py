import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# Cargar los datos desde el archivo CSV (ajusta la ruta)
df = pd.read_csv('data/banksim.csv')

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Convertir algunas columnas a tipos adecuados
df['fraud'] = df['fraud'].astype(int)
df['amount'] = df['amount'].astype(float)
df['step'] = df['step'].astype(int)

# Métricas Clave
total_transactions = len(df)
total_fraud = df['fraud'].sum()
fraud_detection_rate = (total_fraud / total_transactions) * 100
precision_alerts = np.random.uniform(85, 95)  # Simulando valor, debe calcularse según el modelo
f1_score = np.random.uniform(87, 93)  # Simulando valor, debe calcularse según el modelo
avg_detection_time = np.random.uniform(5, 10)  # Simulando valor en horas

# Gráfico 1: Mapa de Calor Geográfico (Código Postal de Origen y Comerciantes)
df['zipcodeOri'] = df['zipcodeOri'].astype(str)
df['zipMerchant'] = df['zipMerchant'].astype(str)
heatmap_data = df.groupby(['zipcodeOri', 'zipMerchant'])['fraud'].sum().reset_index()
heatmap = px.density_heatmap(heatmap_data, x='zipcodeOri', y='zipMerchant', z='fraud', 
                             title="Mapa de Calor de Fraudes por Código Postal",
                             labels={'zipcodeOri': 'Código Postal Origen', 'zipMerchant': 'Código Postal Comerciante'})

# Gráfico 2: Gráfico de dispersión (Monto vs Probabilidad de Fraude)
scatter = px.scatter(df, x='amount', y='fraud', color='category', 
                     title="Monto vs. Probabilidad de Fraude", 
                     labels={'amount': 'Monto de la Transacción', 'fraud': 'Fraude'})

# Gráfico 3: Series Temporales de Transacciones Fraudulentas
df['hour'] = df['step'] % 24  # Convertir el paso a horas para fines ilustrativos
time_series = px.line(df.groupby('hour')['fraud'].sum().reset_index(), x='hour',  y='fraud',
                      title="Tendencia de Fraude por Hora del Día", 
                      labels={'hour': 'Hora del Día', 'fraud': 'Número de Fraudes'})

# Gráfico 4: Matriz de correlación entre variables
# Seleccionar solo las columnas numéricas para calcular la correlación
df_numeric = df.select_dtypes(include=[np.number])

# Calcular la matriz de correlación
corr_matrix = df_numeric.corr()

# Crear la matriz de correlación con px.imshow
correlation_heatmap = px.imshow(corr_matrix, 
                                title="Matriz de Correlación entre Variables",
                                labels={'color': 'Correlación'})

# Gráfico 5: Importancia de Variables (usando simulación de importancia)
importancia_vars = pd.DataFrame({
    'Variable': ['Monto', 'Género', 'Edad', 'Categoría', 'Código Postal', 'Zip Comerciante'],
    'Importancia': np.random.uniform(0, 1, 6)
}).sort_values(by='Importancia', ascending=False)
var_importance_chart = px.bar(importancia_vars, x='Variable', y='Importancia', 
                              title="Importancia de Variables Predictivas")

# Diseño del tablero
app.layout = html.Div([
    html.H1("Tablero de Detección de Fraude en Transacciones Financieras"),

    # Métricas Clave
    html.Div([
        html.H2("Métricas Clave"),
        html.P(f"Total de transacciones: {total_transactions}"),
        html.P(f"Total de fraudes detectados: {total_fraud}"),
        html.P(f"Tasa de detección de fraude (Recall): {fraud_detection_rate:.2f}% (Meta: >90%)"),
        html.P(f"Precisión en alertas: {precision_alerts:.2f}% (Meta: >85%)"),
        html.P(f"F1-Score: {f1_score:.2f}% (Meta: >87%)"),
        html.P(f"Tiempo promedio de detección: {avg_detection_time:.2f} horas"),
    ], className='metrics'),

    # Visualizaciones
    html.Div([
        dcc.Graph(id='heatmap-geografico', figure=heatmap),
        dcc.Graph(id='scatter-amount-fraud', figure=scatter),
        dcc.Graph(id='time-series-fraud', figure=time_series),
        dcc.Graph(id='correlation-heatmap', figure=correlation_heatmap),
        dcc.Graph(id='variable-importance', figure=var_importance_chart),
    ], className='visualizations'),

    # Filtros interactivos
    html.Div([
        html.H2("Filtros Interactivos"),
        dcc.DatePickerRange(
            id='date-range',
            start_date=df['step'].min(),
            end_date=df['step'].max(),
            display_format='Y-MM-DD',
        ),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': c, 'value': c} for c in df['category'].unique()],
            value=df['category'].unique()[0],
            multi=True,
            placeholder="Seleccione categoría(s)"
        ),
        dcc.RangeSlider(
            id='amount-slider',
            min=df['amount'].min(),
            max=df['amount'].max(),
            value=[df['amount'].min(), df['amount'].max()],
            marks={int(i): f"${int(i)}" for i in np.linspace(df['amount'].min(), df['amount'].max(), 5)}
        )
    ], className='filters'),

    # Reportes
    html.Div([
        html.H2("Reportes Automáticos"),
        html.P("Reportes automáticos diarios, semanales y mensuales estarán disponibles según los filtros aplicados.")
    ]),

    # Alertas
    html.Div([
        html.H2("Alertas y Umbrales"),
        html.P("Configuración de alertas según patrones anómalos, rendimiento degradado y concentración inusual de alertas.")
    ]),

])

# Ejecución de la app
if __name__ == '__main__':
    app.run_server(debug=True)
