import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos usando una ruta relativa desde notebooks
data = pd.read_csv('../data/banksim.csv')

# Mostrar las primeras filas para verificar la carga
print(data.head())

# Paso 1: Preprocesamiento
## Eliminar columnas con un único valor (zipCodeOrigin y zipMerchant)
data.drop(columns=['zipcodeOri', 'zipMerchant'], inplace=True)

## Convertir variables categóricas en valores numéricos
label_encoders = {}
categorical_columns = ['customer', 'merchant', 'category', 'gender', 'age']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

## Escalar las variables numéricas
scaler = StandardScaler()
data[['amount']] = scaler.fit_transform(data[['amount']])

# Paso 2: Definir variables independientes y dependientes
X = data.drop(columns=['fraud'])  # Variables independientes
y = data['fraud']  # Variable objetivo

# Paso 3: Balancear el conjunto de datos con SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Paso 4: Entrenar varios modelos
modelos = {
    "Regresión Logística": LogisticRegression(),
    "Bosques Aleatorios": RandomForestClassifier(),
    "Máquina de Soporte Vectorial": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
}

resultados = []

for nombre, modelo in modelos.items():
    # Entrenar el modelo
    modelo.fit(X_train, y_train)
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None
    
    # Calcular métricas
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    accuracy = accuracy_score(y_test, y_pred)
    
    resultados.append({
        "Modelo": nombre,
        "Precisión": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC": auc,
        "Exactitud": accuracy
    })

# Convertir resultados en DataFrame y ordenarlos
resultados_df = pd.DataFrame(resultados).sort_values(by="F1-Score", ascending=False)

# Paso 5: Visualización de los resultados
## Tabla comparativa
print(resultados_df)

## Gráfico de barras para las métricas
plt.figure(figsize=(12, 7))
sns.barplot(data=resultados_df.melt(id_vars=["Modelo"], value_vars=["F1-Score", "Precisión", "Recall", "AUC"]),
            x="variable", y="value", hue="Modelo", palette="Blues")
plt.title("Comparación de Modelos por Métricas")
plt.ylabel("Valor")
plt.xlabel("Métricas")
plt.legend(title="Modelo")
plt.show()

## Gráfico de ranking por F1-Score
plt.figure(figsize=(10, 6))
sns.barplot(data=resultados_df, x="F1-Score", y="Modelo", palette="Blues")
plt.title("Ranking de Modelos por F1-Score")
plt.xlabel("F1-Score")
plt.ylabel("Modelo")
plt.show()