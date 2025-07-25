# Sistema de Detección y Segmentación de Ropa en Tiempo Real

Este sistema combina YOLO para detección de personas y un modelo Keras personalizado para segmentación de ropa.

## Características Principales

- ✅ **Detección de personas** en tiempo real usando YOLO
- ✅ **Segmentación de ropa** usando modelo Keras personalizado
- ✅ **Análisis cada 3 segundos** para optimizar rendimiento
- ✅ **Visualización de segmentación** directamente en la webcam
- ✅ **Detección de colores** de prendas superiores e inferiores
- ✅ **Overlay en tiempo real** mostrando la segmentación

## Funcionamiento

1. **Carga los modelos:**
   - `best_human_model.pt` (YOLO para detectar personas)
   - `best_model.keras` (modelo personalizado para segmentación)

2. **Detección continua:**
   - Detecta personas en cada frame
   - Muestra rectángulo verde alrededor de personas

3. **Análisis cada 3 segundos:**
   - Extrae la región de la persona detectada
   - Escala la imagen a 128x128 (entrada del modelo Keras)
   - Genera máscara de segmentación
   - Detecta colores de prendas superiores e inferiores
   - Aplica overlay de segmentación sobre la persona

4. **Visualización:**
   - Segmentación aplicada directamente sobre la webcam
   - Miniatura de la máscara en esquina superior derecha
   - Colores detectados mostrados como texto
   - Countdown para próximo análisis

## Dependencias Necesarias

Instalar con:
```bash
pip install opencv-python numpy tensorflow ultralytics scikit-learn
```

## Controles

- **ESC**: Salir de la aplicación

## Colores de Segmentación

- **Magenta**: Prenda superior (clase 1)
- **Cyan**: Prenda inferior (clase 2)
- **Gris oscuro**: Fondo (clase 0)

## Optimizaciones

- Análisis de segmentación cada 3 segundos para mejor rendimiento
- Detección de personas en cada frame para respuesta inmediata
- Reutilización de resultados entre análisis
- Método automático de detección de colores (dominant/kmeans según diversidad)

## Ejecución

```bash
python camara_all_models.py
```

Asegúrate de que los archivos `best_human_model.pt` y `best_model.keras` estén en el mismo directorio.
