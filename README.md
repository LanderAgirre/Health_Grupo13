# Proyecto: Clasificación y Segmentación de Lesiones Dermatológicas

## Descripción

Notebook de análisis y modelos para diagnóstico de lesiones de piel usando el dataset HAM10000.
Incluye extracción de características (ABCD), segmentación básica y avanzada, análisis exploratorio (EDA) y comparación de modelos (Random Forest, SVM, KNN, Gradient Boosting).

## Archivos principales
- Health_grupo13.ipynb: notebook principal con el flujo completo.
- Datos/: carpeta con imágenes y metadatos (p. ej. `HAM10000_metadata`, `HAM10000_images_part_1`,`HAM10000_images_part_2`,
`HAM10000_segmentations_lesion_tschandl`).

## Requisitos e instalación

1. Crear y activar un entorno virtual:
   python -m venv venv
   venv\Scripts\activate

2. Instalar dependencias:
   pip install -r requirements.txt

## Ejecutar
1. Abrir `Health_grupo13.ipynb` en Jupyter Notebook.
2. Ejecutar las celdas en orden (comenzando por las importaciones y la carga de `Datos/HAM10000_metadata`).

## Notas
- Las rutas en el notebook asumen la estructura local mostrada en el repositorio; ajuste `df['path']` o las rutas si sus datos están en otra ubicación.
