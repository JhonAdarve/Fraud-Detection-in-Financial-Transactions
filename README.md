# Proyecto de Detección de Fraude en Transacciones Bancarias

## 📊 Descripción del Proyecto
Sistema de detección de fraude basado en machine learning para identificar transacciones bancarias fraudulentas, utilizando datos simulados de BankSim basados en transacciones reales de un banco español. El proyecto incluye análisis exploratorio de datos, desarrollo de modelos predictivos y un dashboard interactivo para monitoreo en tiempo real.

## 🎯 Objetivos
- **Principal**: Desarrollar un modelo de machine learning para identificar transacciones fraudulentas con alta precisión
- **Secundarios**:
  - Analizar patrones y características de transacciones fraudulentas
  - Implementar un sistema de monitoreo en tiempo real
  - Minimizar falsos positivos manteniendo alta tasa de detección

## 📂 Estructura del Proyecto
```
fraud-detection/
│
├── data/                   # Datos del proyecto
│   ├── raw/               # Datos sin procesar
│   └── processed/         # Datos procesados
│
├── notebooks/             # Jupyter notebooks
│   ├── 01_EDA.ipynb      # Análisis exploratorio
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── src/                   # Código fuente
│   ├── data/             # Scripts de procesamiento
│   ├── features/         # Feature engineering
│   ├── models/           # Modelos
│   └── visualization/    # Código del dashboard
│
├── dashboard/            # Archivos del dashboard
│
├── requirements.txt      # Dependencias
└── README.md
```

## 🔍 Datos
- **Fuente**: BankSim Simulator
- **Registros**: 594,643 transacciones
  - 587,443 normales
  - 7,200 fraudulentas (1.2%)
- **Variables**: 10 características incluyendo:
  - Información demográfica
  - Detalles de transacción
  - Ubicación geográfica
  - Categorías de compra

## 🛠️ Tecnologías Utilizadas
- **Lenguaje**: Python 3.8+
- **Principales Librerías**:
  - Pandas: Procesamiento de datos
  - Scikit-learn: Modelado
  - Plotly/Dash: Visualización
  - NumPy: Computación numérica

## 📈 Dashboard
Sistema de monitoreo interactivo que incluye:
- Métricas de rendimiento del modelo
- Visualizaciones geográficas
- Análisis temporal
- Sistema de alertas
- Reportes automáticos

## 🚀 Instalación y Uso

1. Clonar el repositorio:
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecutar el dashboard:
```bash
python src/dashboard/app.py
```

## 📝 Requerimientos del Sistema
- Python 3.8+
- 8GB RAM mínimo
- Espacio en disco: 2GB
- Navegador web moderno

## 👥 Equipo
- [Nombre] - Data Scientist
- [Nombre] - ML Engineer
- [Nombre] - Data Analyst

## 📑 Licencia
Este proyecto está bajo la licencia [INCLUIR LICENCIA]

## 🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor, abra un issue primero para discutir los cambios que le gustaría realizar.

## 📧 Contacto
Para preguntas y soporte, contactar a: [EMAIL]

## 🔄 Estado del Proyecto
En desarrollo activo - Versión 1.0.0

## 📚 Referencias
- [Paper BankSim]
- [Documentación relevante]
- [Otros recursos]

---
⚠️ **Nota**: Este proyecto es para fines de investigación y desarrollo. Los datos utilizados son simulados y no contienen información personal real.