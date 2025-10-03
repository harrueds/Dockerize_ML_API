"""
API de Predicción con Flask y Modelo de Machine Learning

Este módulo implementa un servicio web RESTful que expone un modelo de machine learning
transformado en un endpoint de predicción. Utiliza Flask como framework web y joblib
para cargar el modelo pre-entrenado.

Requisitos:
    pip install flask joblib scikit-learn numpy
"""

from flask import Flask, request, jsonify  # Framework web y utilidades HTTP
import joblib  # Para cargar el modelo entrenado
import numpy as np  # Para operaciones numéricas

# Inicialización de la aplicación Flask
app = Flask(__name__)

# Carga del modelo de machine learning pre-entrenado
# Nota: El archivo 'modelo.pkl' debe estar en el mismo directorio
model = joblib.load("modelo.pkl")


@app.route("/", methods=["GET"])
def home():
    """
    Endpoint raíz para verificación del estado del servicio.

    Returns:
        JSON: Un objeto con el estado del servicio y código de estado HTTP 200.
    """
    return jsonify({"status": "API lista"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para realizar predicciones utilizando el modelo cargado.

    El cuerpo de la petición debe ser un JSON con el siguiente formato:
        {
            "features": [val1, val2, ..., valN]  # Lista de características numéricas
        }

    Returns:
        JSON: Un objeto con la predicción (entero) en caso de éxito,
              o un mensaje de error en caso de fallo.
    """
    try:
        # a) Extraer y validar el JSON del cuerpo de la petición
        data = request.get_json()
        if not data or "features" not in data:
            return (
                jsonify(
                    {
                        "error": "Bad Request",
                        "message": "El cuerpo de la petición debe ser un JSON con la clave 'features'",
                    }
                ),
                400,
            )

        features = data["features"]

        # b) Validar que 'features' sea una lista
        if not isinstance(features, list):
            return (
                jsonify(
                    {
                        "error": "Bad Request",
                        "message": "'features' debe ser una lista de valores numéricos",
                    }
                ),
                400,
            )

        # c) Convertir a array NumPy y ajustar dimensiones
        try:
            X = np.array(features, dtype=float).reshape(1, -1)
        except (ValueError, TypeError):
            return (
                jsonify(
                    {
                        "error": "Bad Request",
                        "message": "Todos los elementos de 'features' deben ser valores numéricos",
                    }
                ),
                400,
            )

        # d) Generar predicción
        try:
            pred = model.predict(X)
            prediction = int(pred[0])  # Convertir a entero para la respuesta JSON
        except Exception as e:
            return (
                jsonify(
                    {
                        "error": "Prediction Error",
                        "message": f"Error al generar la predicción: {str(e)}",
                    }
                ),
                500,
            )

        # e) Devolver resultado como JSON
        return jsonify({"prediction": prediction, "status": "success"}), 200

    except Exception as e:
        # Manejo de errores inesperados
        app.logger.error(f"Error inesperado: {str(e)}")
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "Ocurrió un error interno en el servidor",
                }
            ),
            500,
        )


if __name__ == "__main__":
    # Configuración del servidor de desarrollo
    # - host="0.0.0.0" hace que el servidor sea accesible desde cualquier dirección
    # - debug=True habilita el modo de depuración (solo para desarrollo)
    # - port=5000 define el puerto donde se ejecutará el servidor
    app.run(host="0.0.0.0", port=5000, debug=True)
