# 🎯 IA de Segmentación de Imágenes para Detección de Ropa

## 📋 Descripción del Proyecto

Este proyecto implementa una **Inteligencia Artificial de Segmentación Semántica** para la detección y clasificación automática de prendas de vestir en imágenes. El sistema utiliza una arquitectura **U-Net** personalizada entrenada con TensorFlow/Keras para identificar y segmentar dos tipos de prendas:

- 🔴 **Prendas de Arriba** (camisetas, blusas, suéteres, etc.)
- 🔵 **Prendas de Abajo** (pantalones, faldas, shorts, etc.)

## 🏗️ Arquitectura del Sistema

### 1. **Pipeline de Procesamiento YOLO (Preparación de Datos)**
- Detección inicial de objetos usando modelos YOLO pre-entrenados
- Extracción y filtrado de regiones de interés (ROIs) de prendas
- Generación automática de datasets balanceados

### 2. **Red Neuronal U-Net Personalizada**
- Arquitectura encoder-decoder optimizada para segmentación semántica
- 32 filtros base con capacidad de escalamiento
- Salida de 3 clases: Fondo, Prenda Arriba, Prenda Abajo

## 🚀 Características Principales

✅ **Eficiencia en Memoria**: Uso de `tf.data.Dataset` para manejo inteligente de grandes volúmenes de datos
✅ **Soporte CUDA**: Entrenamiento acelerado por GPU con TensorFlow
✅ **Predicción en Tiempo Real**: Script de inferencia para imágenes individuales
✅ **Visualización Avanzada**: Overlay colorizado para resultados de segmentación
✅ **Callbacks Inteligentes**: Early stopping, reducción de learning rate automática

## 📁 Estructura del Proyecto

```
kawaii_objects_recognition/
├── segmentation_dataset_balanced/     # Dataset principal
│   ├── train/
│   │   ├── images/                   # Imágenes de entrenamiento
│   │   └── masks/                    # Máscaras de segmentación
│   └── proccess/                     # Resultados y previsualizaciones
├── unet/                             # Código principal U-Net
│   ├── utils/
│   │   ├── dataset_robust.py         # Carga eficiente de datos
│   │   └── model.py                  # Arquitectura U-Net
│   ├── train_with_robust_dataset.py  # Script de entrenamiento principal
│   ├── predict.py                    # Predicción en imágenes nuevas
│   └── model/                        # Modelos entrenados
│       └── best_model.keras          # Mejor modelo guardado
├── yolo_images/                      # Imágenes de prueba YOLO
└── README.md                         # Este archivo
```

## ⚙️ Instalación y Configuración

### 1. **Prerequisitos del Sistema**
```bash
# Verificar GPU NVIDIA (opcional pero recomendado)
nvidia-smi

# Verificar CUDA instalado
nvcc --version
```

### 2. **Crear Entorno Virtual**
```bash
cd kawaii_objects_recognition
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ó .venv\Scripts\activate  # Windows
```

### 3. **Instalar Dependencias**
```bash
pip install tensorflow[and-cuda]
pip install matplotlib pillow numpy scikit-learn
```

### 4. **Configuración CUDA (para GPU)**
```bash
# Instalar CUDA Toolkit
sudo apt install nvidia-cuda-toolkit

# Verificar instalación
nvcc --version
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## 🎯 Uso del Sistema

### **Entrenamiento del Modelo**

#### Opción 1: Entrenamiento Completo (Recomendado)
```bash
cd unet
python train_with_robust_dataset.py
```

**Características del entrenamiento:**
- 📊 Carga eficiente de datos con `tf.data.Dataset`
- 🧠 Uso optimizado de memoria RAM
- 🚀 Aceleración por GPU automática
- 📈 Monitoreo en tiempo real con callbacks inteligentes
- 💾 Guardado automático del mejor modelo

#### Configuración Personalizada
```python
# Editar en train_with_robust_dataset.py
batch_size = 16      # Ajustar según RAM/GPU disponible
epochs = 25          # Número de épocas de entrenamiento
learning_rate = 1e-4 # Tasa de aprendizaje inicial
validation_split = 0.2 # 20% para validación
```

### **Predicción en Imágenes Nuevas**

```bash
cd unet
python predict.py
```

**Funcionalidades de predicción:**
- 🖼️ Carga automática del mejor modelo entrenado
- 🎨 Visualización con colores brillantes y llamativos
- 🔍 Tres vistas: Original, Máscara, Overlay
- 💾 Guardado automático de resultados

#### Predicción Personalizada
```python
# Usar imagen específica
predict_image(
    model_path="model/best_model.keras",
    image_path="ruta/a/tu/imagen.jpg"
)
```

## 🧠 Detalles Técnicos

### **Arquitectura U-Net Personalizada**
```python
# Configuración del modelo
input_size = (128, 128, 3)  # Imágenes RGB de 128x128
n_filters = 32              # Filtros base
n_classes = 3               # Fondo + 2 tipos de prenda
```

### **Pipeline de Datos Eficiente**
- **Carga Lazy**: Solo carga imágenes cuando se necesitan
- **Procesamiento Paralelo**: Usa todos los cores disponibles
- **Prefetching**: Prepara el siguiente batch mientras entrena
- **Augmentación**: Rotaciones y flips automáticos

### **Optimizaciones de Memoria**
```python
# Antes (problemático)
X, y = load_all_images()  # ❌ Carga todo en RAM

# Ahora (eficiente)
dataset = tf.data.Dataset.from_generator(...)  # ✅ Carga bajo demanda
dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)
```

## 📊 Métricas y Resultados

### **Métricas de Entrenamiento**
- **Loss Function**: Sparse Categorical Crossentropy
- **Métrica Principal**: Accuracy
- **Optimizador**: Adam con learning rate adaptativo
- **Callbacks**: Early Stopping, Reduce LR on Plateau

### **Visualizaciones Generadas**
1. **training_dataset_preview.png**: Preview del dataset de entrenamiento
2. **training_history_complete.png**: Gráficos detallados de entrenamiento
3. **training_summary.png**: Resumen visual del proceso
4. **prediction_result.png**: Resultado de predicción con overlay

## 🛠️ Solución de Problemas Comunes

### **Error: "Out of Memory" durante entrenamiento**
```bash
# Reducir batch_size
batch_size = 8  # En lugar de 16
```

### **Error: "CUDA out of memory"**
```python
# Forzar uso de CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### **Error: "libdevice not found"**
```bash
# Crear enlace simbólico
sudo ln -s /usr/lib/cuda/libdevice.10.bc ./libdevice.10.bc
```

### **Modelo no converge**
```python
# Ajustar learning rate
learning_rate = 1e-5  # Más conservador
```

## 🔮 Próximas Mejoras

- [ ] **Aumento de Dataset**: Incorporar más categorías de ropa
- [ ] **Transfer Learning**: Usar modelos pre-entrenados como backbone
- [ ] **Detección en Video**: Procesamiento en tiempo real
- [ ] **API REST**: Servicio web para predicciones
- [ ] **Optimización Mobile**: Versión TensorFlow Lite

## 📈 Casos de Uso

### **Comercio Electrónico**
- Clasificación automática de inventario
- Mejora de motores de búsqueda visual
- Recomendaciones personalizadas

### **Realidad Aumentada**
- Prueba virtual de prendas
- Aplicaciones de moda interactivas

### **Análisis de Tendencias**
- Estudios de mercado automatizados
- Análisis de preferencias de consumo

## 🤝 Contribuciones

Este proyecto forma parte de un trabajo universitario de investigación en Computer Vision y Deep Learning. Las contribuciones y mejoras son bienvenidas.

### **Cómo Contribuir**
1. Fork del repositorio
2. Crear branch para nueva feature
3. Implementar mejoras con pruebas
4. Pull request con descripción detallada

## 📜 Licencia

Proyecto desarrollado con fines académicos y de investigación.

## 👨‍💻 Autor

**Santiago** - Estudiante de Ingeniería en Computer Vision y Deep Learning
- Universidad: [Tu Universidad]
- Proyecto: Reconocimiento Kawaii de Objetos con Segmentación Semántica

---

## 🚀 Quick Start

```bash
# Clonar y configurar
git clone [tu-repo]
cd kawaii_objects_recognition
python -m venv .venv && source .venv/bin/activate

# Instalar dependencias
pip install tensorflow matplotlib pillow numpy scikit-learn

# Entrenar modelo
cd unet && python train_with_robust_dataset.py

# Hacer predicción
python predict.py
```

**¡Tu IA de segmentación de ropa está lista para usar!** 🎉👕👖