# Ejemplo de uso del dataset eficiente en memoria

## 🚀 Solución al problema de memoria RAM

El problema que tenías era que estabas cargando **todas las imágenes** en memoria de una vez con tu función `PreprocessData()`. Esto es muy ineficiente y causa problemas de memoria.

## Nueva solución con tf.data.Dataset

Ahora uso `tf.data.Dataset` que:
- **Carga las imágenes solo cuando las necesita** (lazy loading)
- **Procesa las imágenes en paralelo** para mejor rendimiento
- **Usa prefetching** para preparar el siguiente batch mientras entrena el actual
- **Optimiza automáticamente** el uso de memoria

## 📋 Cómo usar la nueva implementación:

### Opción 1: Usar el archivo modificado
```python
python train.py  # Tu archivo modificado
```

### Opción 2: Usar el archivo optimizado (recomendado)
```python
python train_efficient.py  # Versión completamente optimizada
```

### Opción 3: Usar manualmente en tu código
```python
from unet.utils.dataset_efficient import create_train_validation_datasets

# Crear datasets eficientes
train_dataset, val_dataset = create_train_validation_datasets(
    images_dir="path/to/images",
    masks_dir="path/to/masks", 
    target_shape_img=[128, 128, 3],
    target_shape_mask=[128, 128, 1],
    batch_size=16,
    validation_split=0.2
)

# Entrenar directamente
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

## Ventajas de la nueva implementación:

1. **Memoria constante**: Solo mantiene en memoria un batch a la vez
2. **Procesamiento paralelo**: Las imágenes se procesan usando múltiples cores
3. **Prefetching**: Mientras entrena un batch, prepara el siguiente
4. **Reproducible**: Usa seeds para resultados consistentes
5. **Flexible**: Fácil de modificar batch size y parámetros

## ⚙️ Optimizaciones adicionales incluidas:

- **Callbacks inteligentes**: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
- **Gestión automática de GPU**: Configura memoria dinámica si hay GPU disponible
- **Visualizaciones**: Genera gráficos automáticos del entrenamiento
- **Logging mejorado**: Información clara del progreso

## 🔧 Parámetros configurables:

```python
batch_size = 16        # Ajusta según tu RAM/GPU
validation_split = 0.2 # 20% para validación
target_shape = [128, 128, 3]  # Tamaño de imagen
```

**Resultado**: Tu entrenamiento usará mucha menos RAM y será más eficiente! 🎉
