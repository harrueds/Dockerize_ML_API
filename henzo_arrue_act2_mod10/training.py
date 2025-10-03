"""
Script de Entrenamiento de Modelo de Clasificación - Breast Cancer Wisconsin

Este módulo implementa el pipeline completo de entrenamiento y serialización de un modelo
de clasificación binaria para la detección de cáncer de mama utilizando el dataset
Breast Cancer Wisconsin Diagnostic.

El proceso incluye:
    1. Carga del dataset desde scikit-learn
    2. División estratificada de datos en conjuntos de entrenamiento y prueba
    3. Entrenamiento de un modelo de Regresión Logística
    4. Serialización del modelo entrenado para su posterior despliegue

Dataset:
    - Nombre: Breast Cancer Wisconsin Diagnostic
    - Características: 30 variables numéricas (media, error estándar y peor valor de 10 medidas)
    - Clases: Binaria (0 = maligno, 1 = benigno)
    - Muestras: 569 instancias

Modelo:
    - Algoritmo: Regresión Logística
    - Hiperparámetros: max_iter=5000 (para garantizar convergencia)

Output:
    - Archivo: modelo.pkl (modelo serializado con joblib)

Requisitos:
    pip install scikit-learn joblib

"""

from sklearn.datasets import load_breast_cancer  # Dataset de cáncer de mama
from sklearn.model_selection import train_test_split  # División de datos
from sklearn.linear_model import LogisticRegression  # Algoritmo de clasificación
import joblib  # Serialización de modelos

# ============================================================================
# 1. CARGA DEL DATASET
# ============================================================================
# Carga el dataset Breast Cancer Wisconsin Diagnostic directamente desde scikit-learn
# - X: matriz de características (569 muestras × 30 características)
# - y: vector objetivo con las etiquetas de clase (0 = maligno, 1 = benigno)
X, y = load_breast_cancer(return_X_y=True)

# ============================================================================
# 2. DIVISIÓN DE DATOS EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
# ============================================================================
# Separa los datos en conjuntos de entrenamiento (80%) y prueba (20%)
# Parámetros:
#   - test_size=0.2: reserva el 20% de los datos para evaluación
#   - random_state=42: semilla aleatoria para reproducibilidad de resultados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# 3. ENTRENAMIENTO DEL MODELO
# ============================================================================
# Inicialización del modelo de Regresión Logística
# - max_iter=5000: número máximo de iteraciones para el algoritmo de optimización
#   (incrementado desde el valor por defecto para asegurar convergencia)
model = LogisticRegression(max_iter=5000)

# Entrenamiento del modelo con los datos de entrenamiento
# El método fit() ajusta los coeficientes del modelo para minimizar la función de pérdida
model.fit(X_train, y_train)

# ============================================================================
# 4. SERIALIZACIÓN DEL MODELO ENTRENADO
# ============================================================================
# Guarda el modelo entrenado en disco usando joblib
# - Formato: pickle optimizado para objetos NumPy
# - Archivo de salida: modelo.pkl
# - Propósito: permite reutilizar el modelo sin necesidad de reentrenamiento
joblib.dump(model, "modelo.pkl")

# ============================================================================
# 5. CONFIRMACIÓN DE ÉXITO
# ============================================================================
# Mensaje informativo para confirmar que el proceso finalizó correctamente
print("✓ Modelo entrenado y guardado exitosamente como 'modelo.pkl'")
