"""
simplet predictor.py
This script loads a pre-trained model and performs image segmentation on a given image.
It visualizes the original image, the predicted segmentation mask, and the overlay of both.
It also provides statistics on the segmentation results.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_model_safe(model_path):
    """_summary_
        Loads the model safely using different methods to ensure compatibility.
    Args:
        model_path (str): Path to the model file.

    Returns:
        tf.keras.Model: Loaded Keras model or None if loading failed.
    """
    try:
        # Método 1: TensorFlow nativo
        from tensorflow.keras.models import load_model # type: ignore
        model = load_model(model_path)
        print("Modelo cargado con tf.keras")
        return model
    except:
        try:
            # Método 2: Directo desde tf
            import tensorflow.keras as keras # type: ignore
            model = keras.models.load_model(model_path)
            print("Modelo cargado con tensorflow.keras")
            return model
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return None

def predict_image(model_path="best_model.keras", image_path="../yolo_images/ROPA.jpg"):
    """_summary_
        Main function to predict segmentation on a given image using the specified model.  

    Args:
        model_path (str, optional): Path to the model file. Defaults to "best_model.keras".
        image_path (str, optional): Path to the image file. Defaults to "../yolo_images/ROPA.jpg".
    """
    
    print("Iniciando predicción...")
    
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Imagen no encontrada: {image_path}")
        return
    
    model = load_model_safe(model_path)
    if model is None:
        return
    
    print("Procesando imagen...")
    
    original = Image.open(image_path).convert('RGB')
    print(f"Tamaño original: {original.size}")
    
    resized = original.resize((128, 128))
    
    img_array = np.array(resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    print("Haciendo predicción...")
    predictions = model.predict(img_batch, verbose=0)
    
    mask = np.argmax(predictions[0], axis=-1)
    
    # Colores más brillantes y llamativos
    colors = np.array([
        [0, 0, 0],          # Clase 0: Transparente (Fondo - no se colorea)
        [255, 0, 255],      # Clase 1: Magenta brillante (Prenda Arriba)
        [0, 255, 255]       # Clase 2: Cyan brillante (Prenda Abajo)
    ], dtype=np.uint8)
    
    colored_mask = colors[mask]
    
    # Crear overlay manteniendo la imagen original intacta
    original_array = np.array(resized)
    overlay = original_array.copy()
    
    # Solo colorear donde hay prendas detectadas (clases 1 y 2)
    # Para prenda arriba (clase 1): magenta brillante
    prenda_arriba_mask = (mask == 1)
    overlay[prenda_arriba_mask] = overlay[prenda_arriba_mask] * 0.3 + np.array([255, 0, 255]) * 0.7
    
    # Para prenda abajo (clase 2): cyan brillante  
    prenda_abajo_mask = (mask == 2)
    overlay[prenda_abajo_mask] = overlay[prenda_abajo_mask] * 0.3 + np.array([0, 255, 255]) * 0.7
    
    overlay = overlay.astype(np.uint8)
    
    # Visualizar
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(resized)
    plt.title('Imagen Original', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    # Crear máscara visual más clara
    mask_visual = np.zeros((128, 128, 3), dtype=np.uint8)
    mask_visual[mask == 0] = [50, 50, 50]      # Fondo gris oscuro
    mask_visual[mask == 1] = [255, 0, 255]     # Magenta para prenda arriba
    mask_visual[mask == 2] = [0, 255, 255]     # Cyan para prenda abajo
    plt.imshow(mask_visual)
    plt.title('Máscara Segmentación', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Imagen con Destacado Brillante', fontsize=12)
    plt.axis('off')
    
    plt.suptitle('� Segmentación con Colores Brillantes', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Estadísticas
    unique, counts = np.unique(mask, return_counts=True)
    total = mask.size
    
    print("\n📊 RESULTADO:")
    classes = {0: "Fondo", 1: "Prenda Arriba", 2: "Prenda Abajo"}
    for class_id, count in zip(unique, counts):
        percent = (count/total) * 100
        print(f"   {classes[class_id]}: {percent:.1f}%")
    
    # Guardar resultado
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(resized)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title('Segmentado')
    plt.axis('off')
    
    plt.savefig('results/prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("💾 Resultado guardado en: results/prediction.png")
    print("¡Predicción completada!")

if __name__ == "__main__":
    print("🚀 PREDICTOR DE SEGMENTACIÓN DE ROPA")
    print("=" * 45)
    
    # Buscar modelo automáticamente
    model_candidates = [
        "best_model.keras",
        "model/best_model.keras", 
        "../best_model.keras"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break
    
    if model_path:
        print(f"📁 Modelo encontrado: {model_path}")
        predict_image(model_path)
    else:
        print("No se encontró el modelo best_model.keras")
        print("💡 Asegúrate de que esté en una de estas ubicaciones:")
        for candidate in model_candidates:
            print(f"   - {candidate}")
