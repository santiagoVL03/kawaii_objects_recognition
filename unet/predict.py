"""
Simple predictor.py

This script loads a pre-trained model and performs image segmentation on a given image.
It visualizes the original image, the predicted segmentation mask, and the overlay of both.
It also provides statistics on the segmentation results.
ENHANCED: Now detects and displays real colors of detected garments with floating indicators.
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
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

def extract_dominant_color(image_array, mask, class_id, method='auto'):
    """
    Extrae el color dominante y retorna también el método utilizado.
    
    Returns:
        tuple: ((R, G, B), método_usado, razón)
    """
    class_pixels = image_array[mask == class_id]
    if len(class_pixels) == 0:
        return None, 'none', 'Sin píxeles detectados'

    class_pixels = np.clip(class_pixels, 0, 255).astype(np.uint8)

    reason = ""
    if method == 'auto':
        rounded_pixels = (class_pixels // 16) * 16
        pixel_tuples = [tuple(pixel.astype(int)) for pixel in rounded_pixels]
        from collections import Counter
        color_counts = Counter(pixel_tuples)
        most_common_color, count = color_counts.most_common(1)[0]
        percentage = (count / len(pixel_tuples)) * 100

        if percentage >= 40:
            method_to_use = 'dominant'
            reason = f"Color más frecuente ({percentage:.1f}% del total)"
        else:
            method_to_use = 'kmeans'
            reason = f"Alta diversidad de colores (ninguno supera el 40%)"
    else:
        method_to_use = method
        reason = "Modo manual"

    if method_to_use == 'dominant':
        rounded_pixels = (class_pixels // 12) * 12
        pixel_tuples = [tuple(pixel.astype(int)) for pixel in rounded_pixels]
        from collections import Counter
        color_counts = Counter(pixel_tuples)
        dominant_color = color_counts.most_common(1)[0][0]

    elif method_to_use == 'kmeans':
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(class_pixels)
            labels = kmeans.labels_
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique_labels[np.argmax(counts)]
            dominant_color = tuple(kmeans.cluster_centers_[dominant_cluster].astype(int))
        except ImportError:
            return extract_dominant_color(image_array, mask, class_id, 'dominant')

    elif method_to_use == 'median':
        dominant_color = tuple(np.median(class_pixels, axis=0).astype(int))

    else:  # average
        dominant_color = tuple(np.mean(class_pixels, axis=0).astype(int))

    dominant_color = tuple(max(0, min(255, int(c))) for c in dominant_color)
    return dominant_color, method_to_use, reason


def get_multiple_colors(image_array, mask, class_id, num_colors=3):
    """
    Extrae múltiples colores dominantes de una prenda (para prendas multicolor)
    
    Args:
        image_array: Array de la imagen original
        mask: Máscara de segmentación  
        class_id: ID de la clase
        num_colors: Número de colores a extraer
        
    Returns:
        list: Lista de tuplas (R, G, B) con los colores más dominantes
    """
    class_pixels = image_array[mask == class_id]
    
    if len(class_pixels) == 0:
        return []
    
    # Método mejorado: Usar agrupamiento para encontrar colores principales
    from collections import Counter
    
    # Agrupar colores similares con mayor precisión
    rounded_pixels = (class_pixels // 8) * 8  # Más preciso
    pixel_tuples = [tuple(pixel.astype(int)) for pixel in rounded_pixels]
    
    color_counts = Counter(pixel_tuples)
    
    # Obtener los N colores más frecuentes
    most_common = color_counts.most_common(num_colors)
    
    colors = []
    total_pixels = len(pixel_tuples)
    
    for color, count in most_common:
        percentage = (count / total_pixels) * 100
        # Solo incluir colores que representen al menos 5% de la prenda
        if percentage >= 5.0:
            colors.append({
                'color': tuple(max(0, min(255, int(c))) for c in color),
                'percentage': percentage
            })
    
    return colors

def rgb_to_hex(rgb):
    """Convierte RGB a formato hexadecimal"""
    # Asegurar que los valores estén en el rango correcto
    r, g, b = [max(0, min(255, int(c))) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def get_text_color(bg_color):
    """Determina si usar texto blanco o negro basado en el brillo del fondo"""
    # Asegurar que los valores estén en el rango correcto
    r, g, b = [max(0, min(255, int(c))) for c in bg_color]
    
    # Calcular brillo usando la fórmula estándar
    brightness = (r * 0.299 + g * 0.587 + b * 0.114)
    return 'white' if brightness < 128 else 'black'

def find_center_point(mask, class_id):
    """Encuentra el punto central (centroide) de una región segmentada"""
    y_coords, x_coords = np.where(mask == class_id)
    if len(y_coords) == 0:
        return None
    
    # Calcular el centroide real de la región
    center_y = int(np.mean(y_coords))
    center_x = int(np.mean(x_coords))
    return (center_x, center_y)

def calculate_indicator_position(center_point, image_shape, class_id):
    """
    Calcula la posición óptima para el indicador basado en el centro de la máscara
    
    Args:
        center_point: (x, y) centro de la máscara
        image_shape: forma de la imagen (height, width)
        class_id: 1 para arriba, 2 para abajo
    
    Returns:
        tuple: (line_end_x, line_end_y, box_x, box_y)
    """
    center_x, center_y = center_point
    height, width = image_shape[:2]
    
    if class_id == 1:  # Prenda arriba
        # Para prenda arriba, el indicador va hacia arriba y a la izquierda
        if center_x > width // 2:  # Si está en la mitad derecha
            line_end_x = max(15, center_x - 40)
            line_end_y = max(20, center_y - 30)
        else:  # Si está en la mitad izquierda
            line_end_x = min(width - 35, center_x + 40)
            line_end_y = max(20, center_y - 30)
    else:  # Prenda abajo (class_id == 2)
        # Para prenda abajo, el indicador va hacia abajo y a la izquierda
        if center_x > width // 2:  # Si está en la mitad derecha
            line_end_x = max(15, center_x - 40)
            line_end_y = min(height - 25, center_y + 30)
        else:  # Si está en la mitad izquierda
            line_end_x = min(width - 35, center_x + 40)
            line_end_y = min(height - 25, center_y + 30)
    
    # La caja va en el extremo de la línea
    box_x = line_end_x - 15  # Centrar la caja en el extremo de la línea
    box_y = line_end_y - 7.5  # Centrar verticalmente
    
    return (line_end_x, line_end_y, box_x, box_y)

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
    
    # Obtener imagen original como array para extracción de colores
    original_array = np.array(resized)
    
    # Extraer colores reales de las prendas (con método mejorado)
    color_prenda_arriba, metodo_arriba, razon_arriba = extract_dominant_color(original_array, mask, 1, method='auto')
    color_prenda_abajo, metodo_abajo, razon_abajo = extract_dominant_color(original_array, mask, 2, method='auto')


    
    # OPCIONAL: Extraer múltiples colores para análisis avanzado
    colors_arriba = get_multiple_colors(original_array, mask, 1, num_colors=3)
    colors_abajo = get_multiple_colors(original_array, mask, 2, num_colors=3)
    
    # Encontrar puntos centrales para los indicadores
    center_arriba = find_center_point(mask, 1)
    center_abajo = find_center_point(mask, 2)
    
    # Colores para la máscara (mantener los originales)
    colors = np.array([
        [0, 0, 0],          # Clase 0: Transparente (Fondo - no se colorea)
        [255, 0, 255],      # Clase 1: Magenta brillante (Prenda Arriba)
        [0, 255, 255]       # Clase 2: Cyan brillante (Prenda Abajo)
    ], dtype=np.uint8)
    
    colored_mask = colors[mask]
    
    # Crear overlay manteniendo la imagen original intacta
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Imagen Original CON INDICADORES DE COLOR
    ax1 = axes[0]
    ax1.imshow(resized)
    ax1.set_title('Imagen Original con Indicadores', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Agregar indicadores de color flotantes DINÁMICOS
    if color_prenda_arriba and center_arriba:
        # Calcular posición dinámica para indicador de prenda arriba
        line_end_x, line_end_y, box_x, box_y = calculate_indicator_position(
            center_arriba, resized.size + (3,), 1
        )
        
        # Línea indicadora para prenda arriba
        ax1.plot([center_arriba[0], line_end_x], [center_arriba[1], line_end_y], 
                'k-', linewidth=2.5, zorder=5)
        
        # Punto en el centro de la máscara
        ax1.plot(center_arriba[0], center_arriba[1], 'ko', markersize=4, zorder=6)
        
        # Cuadro indicador para prenda arriba (ENCIMA DE TODO)
        color_box = FancyBboxPatch(
            (box_x, box_y), 30, 15,
            boxstyle="round,pad=2",
            facecolor=np.array(color_prenda_arriba)/255,
            edgecolor='black',
            linewidth=2,
            zorder=10  # MUY ALTO para estar encima de todo
        )
        ax1.add_patch(color_box)
        
        # Texto del color (ENCIMA DE TODO)
        hex_color = rgb_to_hex(color_prenda_arriba)
        text_color = get_text_color(color_prenda_arriba)
        ax1.text(line_end_x, line_end_y, f'Arriba\n{hex_color}', 
                ha='center', va='center', 
                fontsize=8, fontweight='bold',
                color=text_color, zorder=11)  # MUY ALTO
    
    if color_prenda_abajo and center_abajo:
        # Calcular posición dinámica para indicador de prenda abajo
        line_end_x, line_end_y, box_x, box_y = calculate_indicator_position(
            center_abajo, resized.size + (3,), 2
        )
        
        # Línea indicadora para prenda abajo
        ax1.plot([center_abajo[0], line_end_x], [center_abajo[1], line_end_y], 
                'k-', linewidth=2.5, zorder=5)
        
        # Punto en el centro de la máscara
        ax1.plot(center_abajo[0], center_abajo[1], 'ko', markersize=4, zorder=6)
        
        # Cuadro indicador para prenda abajo (ENCIMA DE TODO)
        color_box = FancyBboxPatch(
            (box_x, box_y), 30, 15,
            boxstyle="round,pad=2",
            facecolor=np.array(color_prenda_abajo)/255,
            edgecolor='black',
            linewidth=2,
            zorder=10  # MUY ALTO para estar encima de todo
        )
        ax1.add_patch(color_box)
        
        # Texto del color (ENCIMA DE TODO)
        hex_color = rgb_to_hex(color_prenda_abajo)
        text_color = get_text_color(color_prenda_abajo)
        ax1.text(line_end_x, line_end_y, f'Abajo\n{hex_color}', 
                ha='center', va='center', 
                fontsize=8, fontweight='bold',
                color=text_color, zorder=11)  # MUY ALTO
    
    # Máscara Segmentación
    ax2 = axes[1]
    mask_visual = np.zeros((128, 128, 3), dtype=np.uint8)
    mask_visual[mask == 0] = [50, 50, 50]      # Fondo gris oscuro
    mask_visual[mask == 1] = [255, 0, 255]     # Magenta para prenda arriba
    mask_visual[mask == 2] = [0, 255, 255]     # Cyan para prenda abajo
    ax2.imshow(mask_visual)
    ax2.set_title('Máscara Segmentación', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Imagen con destaque
    ax3 = axes[2]
    ax3.imshow(overlay)
    ax3.set_title('Imagen con Destacado Brillante', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.suptitle('Segmentación con Detección de Colores Reales', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Estadísticas
    unique, counts = np.unique(mask, return_counts=True)
    total = mask.size
    
    print("\nRESULTADO:")
    classes = {0: "Fondo", 1: "Prenda Arriba", 2: "Prenda Abajo"}
    for class_id, count in zip(unique, counts):
        percent = (count/total) * 100
        print(f"   {classes[class_id]}: {percent:.1f}%")
    
    # Mostrar colores detectados
    print("\n COLORES DETECTADOS:")
    if color_prenda_arriba:
        hex_arriba = rgb_to_hex(color_prenda_arriba)
        print(f"   Prenda Arriba: RGB{color_prenda_arriba} | {hex_arriba}")
        print(f"      Método usado: {metodo_arriba} → {razon_arriba}")

    else:
        print("   Prenda Arriba: No detectada")
        
    if color_prenda_abajo:
        hex_abajo = rgb_to_hex(color_prenda_abajo)
        print(f"   Prenda Abajo: RGB{color_prenda_abajo} | {hex_abajo}")
        print(f"      Método usado: {metodo_abajo} → {razon_abajo}")

    else:
        print("   Prenda Abajo: No detectada")
    
    # Información sobre métodos disponibles
    print("\n MÉTODOS DE DETECCIÓN DISPONIBLES:")
    print("   - 'dominant': Color más frecuente (bueno para tonos uniformes)")
    print("   - 'kmeans'  : Agrupamiento K-means (mejor para prendas multicolor)")
    print("   - 'median'  : Mediana de color (robusto contra outliers)")
    print("   - 'average' : Promedio de colores (clásico)")
    print("   - 'auto'    : Selección automática basada en la diversidad de color")
    print("                ↪ Usa 'dominant' si ≥40% de píxeles comparten color")
    print("                ↪ Usa 'kmeans' si hay alta variedad de colores")

    # Guardar resultado
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
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
    
    print("Resultado guardado en: results/prediction.png")
if __name__ == "__main__":
    print("🚀 PREDICTOR DE SEGMENTACIÓN DE ROPA - DETECTOR DE COLORES")
    print("=" * 60)
    
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
        print("Asegúrate de que esté en una de estas ubicaciones:")
        for candidate in model_candidates:
            print(f"   - {candidate}")