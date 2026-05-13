import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# 1. Cargar el conjunto de datos [cite: 9]
df = pd.read_csv('diabetes.csv')

# 2. PREPROCESAMIENTO: Identificar y corregir valores nulos ocultos 
# Reemplazamos los 0 por NaN en columnas donde el 0 no tiene sentido físico
cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_con_ceros] = df[cols_con_ceros].replace(0, np.nan)

# Imputamos los valores faltantes usando la mediana del conjunto de datos
df.fillna(df.median(), inplace=True)

# 3. TRANSFORMACIÓN: Escalado y División 
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Escalado estándar para normalizar las magnitudes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Preprocesamiento y Transformación completados con éxito.")

# 4. CONSTRUCCIÓN DEL MODELO: Bosques Aleatorios [cite: 12, 13]
# Instanciamos el modelo computacional
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenamiento del modelo
modelo_rf.fit(X_train, y_train)

# Predicciones
y_pred = modelo_rf.predict(X_test)

# 5. MÉTRICAS DE EVALUACIÓN [cite: 14]
print("\n--- Métricas de Evaluación ---")
print(f"Exactitud (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 6. INTERPRETACIÓN DE RESULTADOS MEDIANTE GRÁFICOS 
plt.figure(figsize=(12, 5))

# Gráfico 1: Matriz de Confusión
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')

# Gráfico 2: Importancia de las Variables
plt.subplot(1, 2, 2)
importancias = pd.Series(modelo_rf.feature_importances_, index=X.columns)
importancias.nlargest(8).plot(kind='barh', color='skyblue')
plt.title('Importancia de las Variables (Features)')

plt.tight_layout()
plt.show()
