Challenge Data Analytics Engineer - MercadoLibre

Descripción del Proyecto

Este proyecto resuelve el challenge de Data Analytics Engineer para el equipo de Advanced Analytics & Machine Learning Commerce de MercadoLibre. El objetivo es desarrollar una solución analítica de clusterización de sellers que permita al equipo comercial identificar segmentos de vendedores y generar estrategias focalizadas.

Problema de Negocio

El equipo comercial necesita realizar estrategias diferenciadas para los sellers, pero actualmente no existe una clasificación que permita identificar aquellos con buen perfil y relevancia para el negocio.
Solución Implementada

Caso Base: Segmentación de sellers usando técnicas de clustering (K-Means)
Extensión GenAI: Clasificador semántico con embeddings LLM para asignar nuevos sellers a clusters predefinidos (Opción A)

Estructura del Proyecto
challenge-meli/
├── Data/
│   └── df_challenge_meli.csv           # Dataset original
├── notebooks/
│   └── analisis_exploratorio.ipynb    # Notebook principal con análisis completo
├── outputs/
│   ├── clusters/
│   │   └── seller_clusters_k3.csv     # Etiquetas de clusters por seller
│   └── reports/
│       └── tabla_desempeno_clusters.csv # Métricas de rendimiento del clasificador
├── requirements.txt                    # Dependencias del proyecto
└── README.md                          # Este archivo
Instalación y Configuración
Prerrequisitos

Python 3.8+
pip

Instalación de Dependencias
bash# Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd challenge-meli

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
requirements.txt
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
sentence-transformers>=2.2.0
jupyter>=1.0.0
Ejecución del Proyecto
1. Preparar los Datos
Descargar el dataset desde el enlace proporcionado en el challenge y colocarlo en la carpeta Data/:
https://drive.google.com/file/d/1Vh7ttgm9t86AFd6BEIRJummjSki3AI--/view?usp=sharing
2. Ejecutar el Análisis
bash# Iniciar Jupyter Notebook
jupyter notebook

# Abrir y ejecutar: notebooks/analisis_exploratorio.ipynb
3. Estructura de Ejecución
El notebook está dividido en las siguientes secciones:

Carga y Exploración de Datos
Análisis Exploratorio Detallado
Ingeniería de Features por Seller
Clustering (K-Means) con Optimización de K
Limpieza de Outliers y Re-clustering
Visualización y Caracterización de Clusters
Generación de Embeddings con LLM
Entrenamiento de Clasificador Supervisado
Evaluación y Ejemplos de Predicción

Metodología
Caso Base: Clusterización de Sellers
Features Engineered por Seller:

n_items: Número total de productos publicados
stock_total: Inventario total acumulado
pct_oos: Porcentaje de productos sin stock
price_mean/median: Estadísticas de precios
disc_mean/share: Métricas de descuentos
nuniq_cat: Diversidad de categorías
topcat_share: Concentración en categoría principal

Proceso de Clustering:

Escalado robusto de features (RobustScaler)
Búsqueda de K óptimo usando método del codo + silhouette score
Limpieza de outliers (percentil 99 en precio y stock)
Re-clustering con datos limpios
Evaluación con K=2 y K=3 para diferentes granularidades

Extensión GenAI: Clasificador Semántico
Pipeline:

Generación de texto por seller: Concatenación de títulos de productos
Embeddings multilingües: paraphrase-multilingual-MiniLM-L12-v2
Clasificador supervisado: Logistic Regression con class_weight="balanced"
Evaluación: Train/test split estratificado con métricas por cluster

Resultados
Segmentación Final (K=3):
Cluster 0 - Vendedores de Precios Altos y Baja Diversidad (~70%)

Características: Precios promedio más altos, stock reducido, poca diversificación
Perfil: Micro-sellers especializados en nichos premium
Estrategia Sugerida: Programa VIP, soporte personalizado, destacar valor agregado

Cluster 1 - Vendedores de Gran Volumen e Inventario (~15%)

Características: Stock total muy alto, más items, precios competitivos
Perfil: Mayoristas o distribuidores principales
Estrategia Sugerida: Condiciones preferenciales, gestión de inventario avanzada, logística optimizada

Cluster 2 - Vendedores Diversificados y Orientados a Descuentos (~15%)

Características: Alta diversidad de categorías, mayor participación en promociones
Perfil: Retailers multi-categoría con estrategia de precio
Estrategia Sugerida: Campañas de descuentos segmentadas, cross-selling, promociones estacionales

Rendimiento del Clasificador:

Modelo: Logistic Regression con embeddings (384 dimensiones)
Enfoque: Balanceado para manejar clases desbalanceadas
Aplicación: Onboarding automático de nuevos sellers

Interpretación de Negocio
Valor Generado:

Segmentación Accionable: 3 grupos con estrategias comerciales diferenciadas
Automatización: Clasificación automática de nuevos sellers
Escalabilidad: Pipeline reproducible para datos futuros

Casos de Uso:

Onboarding: Asignación automática de nuevos sellers a estrategias
Campañas: Segmentación para comunicaciones personalizadas
Análisis: Monitoreo de evolución de sellers entre clusters
Pricing: Estrategias de comisiones diferenciadas por perfil

Archivos de Salida
outputs/clusters/seller_clusters_k3.csv
Mapeo seller_nickname → cluster_k3 para todos los vendedores
outputs/reports/tabla_desempeno_clusters.csv
Métricas de rendimiento del clasificador con ejemplos por cluster
Próximos Pasos
Mejoras Técnicas:

Validación temporal: Evaluar estabilidad de clusters en el tiempo
Features adicionales: Incorporar datos de transacciones, reviews, etc.
Modelos avanzados: Explorar clustering jerárquico o density-based
Monitoreo: Dashboard para tracking de sellers entre clusters

Implementación en Producción:

Pipeline automatizado: Integración con datos en tiempo real
API de clasificación: Endpoint para scoring de nuevos sellers
A/B Testing: Validar efectividad de estrategias por cluster
Feedback loop: Incorporar resultados de campañas para re-entrenar

Tecnologías Utilizadas

Análisis: pandas, numpy, scikit-learn
Visualización: matplotlib, seaborn
GenAI: sentence-transformers (Hugging Face)
Clustering: K-Means con optimización automática de hiperparámetros
ML: Logistic Regression con balanceo de clases

Contacto
Juan Diego Torres Perez - juandiegotorresperez632@gmail.com