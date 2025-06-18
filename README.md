# Predicción de Popularidad de Noticias Online con Machine Learning 🚀

Este repositorio contiene el código y la documentación para un proyecto del curso **Aprendizaje de Máquina (1INF02)**, enfocado en predecir la popularidad de artículos de noticias online utilizando técnicas de aprendizaje automático.

## 📝 Descripción del Proyecto

El objetivo principal de este proyecto es desarrollar un modelo de clasificación binaria para predecir si un artículo de noticias publicado por Mashable se volverá "popular" (definido como >1400 compartidos). Para ello, se implementó un pipeline completo de Machine Learning, explorando y optimizando múltiples algoritmos, desde modelos lineales hasta ensambles avanzados como los de Gradient Boosting.

El modelo final seleccionado, un **StackingClassifier**, demostró ser el más performante, logrando un **ROC AUC de 0.7342** en el conjunto de prueba. Este resultado no solo valida la efectividad de las técnicas aplicadas, sino que también **supera el benchmark académico de referencia** para este dataset.

## 📊 Dataset

Se utilizó el dataset **"Online News Popularity"** del Repositorio de Machine Learning de UCI.
*   **Fuente:** [https://archive.ics.uci.edu/dataset/332/online+news+popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity)
*   **Instancias:** 39,644 artículos de noticias.
*   **Características:** 58 atributos predictivos, clasificados en categorías como contenido, canal, keywords, temporalidad y sentimiento.

## 🚀 Tecnologías y Bibliotecas Utilizadas

*   **Lenguaje de Programación:** Python 3.x
*   **Entorno de Desarrollo:** Google Colab (con aceleración por GPU)
*   **Bibliotecas Principales:**
    *   **Pandas & NumPy:** Para manipulación y análisis de datos.
    *   **Scikit-learn:** Para preprocesamiento de datos, pipelines de modelado y métricas de evaluación.
    *   **RAPIDS cuML:** Implementaciones aceleradas por GPU para Regresión Logística y Random Forest.
    *   **XGBoost:** Algoritmo de Gradient Boosting con soporte nativo para GPU (`tree_method='gpu_hist'`).
    *   **LightGBM:** Algoritmo de Gradient Boosting optimizado para velocidad y rendimiento con soporte nativo para GPU (`device='gpu'`).
    *   **Matplotlib & Seaborn:** Para visualización de datos y resultados.

## 📂 Estructura del Repositorio

*   `/notebooks/`: Contiene el Jupyter Notebook principal del proyecto.
    *   `news_popularity_prediction.ipynb`: El notebook completo con todas las fases del proyecto (carga de datos, EDA, preprocesamiento, modelado, evaluación y análisis del modelo final).
*   `/data/`: (Opcional) Carpeta para almacenar el dataset si se descarga localmente.
    *   `OnlineNewsPopularity.csv`: El archivo CSV del dataset (no incluido en el repo, se descarga automáticamente en el notebook).
*   `/informe/`: (Opcional) Carpeta para el informe final del proyecto.
    *   `informe_final.pdf`: Documento completo del informe en formato IEEE.
*   `/presentacion/`: (Opcional) Carpeta para la presentación de diapositivas.
    *   `presentacion_final.pdf`: Archivo PDF de la presentación.
*   `requirements.txt`: Lista de las principales dependencias de Python.
*   `README.md`: Este archivo, que sirve como guía principal del proyecto.

## 🛠️ Cómo Replicar los Resultados

Para replicar los resultados y ejecutar el notebook, siga los siguientes pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/[tu-usuario]/online-news-popularity-prediction.git
    cd online-news-popularity-prediction
    ```

2.  **(Opcional) Crear un entorno virtual e instalar dependencias:**
    Se recomienda usar un entorno virtual para gestionar las dependencias del proyecto.
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *Nota: Las bibliotecas de GPU (RAPIDS cuML, XGBoost con GPU, LightGBM con GPU) requieren una instalación específica y un entorno compatible (ej. Google Colab con GPU activada o un sistema con GPU NVIDIA y CUDA). El notebook incluye una rutina de instalación condicional para Google Colab.*

3.  **Descargar el dataset:**
    El notebook `news_popularity_prediction.ipynb` está configurado para descargar el dataset `OnlineNewsPopularity.zip` directamente desde la URL de UCI y extraer `OnlineNewsPopularity.csv`. Asegúrese de tener conexión a internet. Si la descarga automática falla, puede descargar el archivo manualmente y colocar `OnlineNewsPopularity.csv` en una carpeta `data/` dentro del directorio del proyecto.

4.  **Ejecutar el Notebook:**
    Abra y ejecute el notebook `notebooks/news_popularity_prediction.ipynb` en un entorno como Jupyter Lab, Jupyter Notebook o, **preferiblemente, Google Colab con aceleración por GPU (Tiempo de ejecución -> Cambiar tipo de entorno de ejecución -> Acelerador de hardware: GPU)**. Ejecute todas las celdas en orden.

## 📈 Resumen de Resultados

La siguiente tabla resume el rendimiento en el conjunto de prueba (ROC AUC y Accuracy) de los modelos individuales y los ensambles evaluados:

| Modelo                     | ROC AUC (Prueba) | Accuracy (Prueba) |
| :------------------------- | :--------------- | :---------------- |
| **StackingClassifier**     | **0.7342**       | **0.6732**        |
| XGBoost (GPU)              | 0.7333           | 0.6721            |
| LightGBM (GPU)             | 0.7321           | 0.6728            |
| VotingClassifier (Soft)    | 0.7341           | 0.6734            |
| Random Forest (cuML GPU)   | 0.7235           | 0.6643            |
| Regresión Logística (cuML) | 0.7048           | 0.6525            |
| *Línea Base (Fernandes et al., 2015)* | *~0.7300*        | *~0.6700*         |

## 👤 Autores

*   Ricardo Meléndez Olivo
*   Juan Galindo
*   Joseph Gutiérrez
*   Juan Coronado