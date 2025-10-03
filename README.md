# Despliegue básico de un modelo de Machine Learning con Flask

### Objetivo: Implementar una API REST con Flask que permita consumir un modelo de clasificación entrenado, incorporando validación de entradas, manejo de errores y pruebas con datos en formato JSON.

---

Este proyecto implementa un flujo completo que comprende el entrenamiento de un modelo de clasificación utilizando scikit-learn, su serialización en un archivo local, la creación de una API REST con Flask para exponer dicho modelo, y finalmente la validación del servicio mediante un script de pruebas. El dataset utilizado es **Breast Cancer Wisconsin Diagnostic**, provisto por la librería `sklearn.datasets`.

## Estructura del proyecto

```bash
/henzo_arrue_actividad2_modulo10
├─ henzo_arrue_act2_mod10/             # Código fuente principal del proyecto
│  ├─ Dockerfile                       # Instrucciones para construir la imagen Docker del proyecto
│  ├─ requirements.txt                 # Dependencias necesarias para ejecutar la aplicación
│  ├─ app.py                           # Aplicación principal (por ejemplo, una API)
│  └─ training.py                      # Script de entrenamiento del modelo
│
├─ informe/                            # Documentación del proyecto
│  ├─ informe.md                       # Informe detallado de la actividad
│  └─ screenshots/                     # Capturas de pantalla utilizadas en el informe
│     ├─ dockerlimpio.png              # Entorno Docker limpio
│     ├─ dockerbuild.png               # Proceso de construcción (build) de la imagen Docker
│     ├─ dockerrun1.png                # Primera parte de la ejecución del contenedor
│     ├─ dockerrun2.png                # Segunda parte de la ejecución del contenedor
│     ├─ dockerrun3.png                # Tercera parte de la ejecución del contenedor
│     ├─ informacionip.png             # Información de la IP del contenedor o servicio
│     └─ prueba_externa.png            # Prueba de acceso o uso externo de la aplicación
│
├─ README.md                           # Documentación principal del repositorio

```

## Requisitos

Se requiere instalar las siguientes dependencias:

* flask  
* joblib  
* numpy  
* numpy-base  
* requests  
* scikit-learn  

---

## Descripción de los scripts

### Script `training.py`

Este script realiza el proceso de entrenamiento del modelo. Las principales etapas son:  
1. Carga del dataset Breast Cancer desde `sklearn.datasets`.  
2. División de los datos en conjuntos de entrenamiento y prueba (80%/20%).  
3. Entrenamiento de un modelo de regresión logística (`LogisticRegression`).  
4. Serialización y guardado del modelo en el archivo `modelo.pkl` utilizando la librería `joblib`.  

Al ejecutar este script con `python training.py`, se genera el archivo `modelo.pkl` que contiene el modelo entrenado.

### Script `app.py`

Este archivo implementa una API REST utilizando Flask. Sus componentes principales son:  
- Carga del modelo previamente entrenado desde `modelo.pkl`.  
- Definición de dos rutas:  
  - `GET /` : responde con un mensaje de estado confirmando que la API está activa.  
  - `POST /predict` : recibe un JSON con la clave `features`, que debe contener una lista numérica de 30 valores correspondientes a las características del dataset. Tras la validación de los datos, la entrada se transforma en un arreglo de NumPy y se genera una predicción. La respuesta es un JSON con la clase predicha.  
- Manejo de errores: ante entradas inválidas o fallos internos, la API devuelve un mensaje descriptivo y un código HTTP adecuado (400 o 500).  

---

## Despliegue con Docker

### 1. Construcción de la imagen
Desde el directorio raíz del proyecto (donde se encuentra el `Dockerfile`), ejecutar:

```bash
docker build -t henzo_arrue_act2_mod10 .
````

Esto creará una imagen llamada `henzo_arrue_act2_mod10` que contiene la API lista para ejecutarse.

### 2. Ejecución del contenedor

Para levantar la API en el puerto `5000`:

```bash
docker run -it -p 5000:5000 henzo_arrue_act2_mod10
```

El contenedor quedará corriendo en segundo plano y la API será accesible en `http://127.0.0.1:5000/`.

### 3. Verificación del estado

Se puede comprobar que la API está activa ejecutando:

```bash
curl http://127.0.0.1:5000/
```

La respuesta esperada es un mensaje JSON confirmando que el servicio está en funcionamiento.

### 4. Prueba de predicción

Para probar la ruta `/predict`, se envía un JSON con una lista de **30 valores numéricos** correspondientes a las características del dataset:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @correcto.json
```

La respuesta será un JSON con la predicción del modelo, por ejemplo:

```json
{
  "prediction": 1
  "status": "success"
}
```

### 5. Prueba de errores

Se puede probar que la API maneja correctamente entradas inválidas, estas pruebas se realizaron utilizando dos archivos JSON, llamados corrupto.json (que contenía 29 features) y corrupto2.json (que contenia un valor no numérico en la lista de features).

Los resultados de estas pruebas se pueden obeservar en el archivo informe.md.

---

## Notas sobre el dataset

El dataset Breast Cancer Wisconsin Diagnostic contiene 30 variables numéricas que describen propiedades de núcleos celulares en imágenes digitalizadas. La variable objetivo (target) es binaria:

* 0: tumor maligno
* 1: tumor benigno

---

## Conclusión

El proyecto ejemplifica un flujo de trabajo básico para desplegar un modelo de aprendizaje automático. Comprende desde el entrenamiento y la serialización del  modelo hasta la creación de un servicio web accesible mediante solicitudes HTTP, su despliegue en contenedor Docker y la validación mediante pruebas. Este enfoque constituye la base para escenarios más complejos de integración de modelos en aplicaciones reales.