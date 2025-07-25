import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import Counter
import time
import os

# Forzar ejecución en CPU para evitar problemas con CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

def load_model_safe(model_path):
    """Carga el modelo de manera segura usando diferentes métodos y forzando CPU"""
    print(f"Cargando modelo desde: {model_path}")
    
    # Asegurar que TensorFlow use solo CPU
    with tf.device('/CPU:0'):
        try:
            # Método 1: Usando tf.keras directamente
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Modelo cargado con tf.keras (CPU)")
            return model
        except Exception as e1:
            print(f"Método 1 falló: {e1}")
            try:
                # Método 2: Importando keras separadamente
                import keras
                model = keras.models.load_model(model_path, compile=False)
                print("Modelo cargado con keras (CPU)")
                return model
            except Exception as e2:
                print(f"Método 2 falló: {e2}")
                try:
                    # Método 3: Con configuración explícita
                    from tensorflow.keras.models import load_model
                    model = load_model(model_path, compile=False)
                    print("Modelo cargado con load_model directo (CPU)")
                    return model
                except Exception as e3:
                    print(f"Método 3 falló: {e3}")
                    print(f"Error cargando modelo: No se pudo cargar con ningún método")
                    return None


print("Cargando modelos...")
print(f"TensorFlow versión: {tf.__version__}")
print(f"Dispositivos GPU disponibles: {len(tf.config.list_physical_devices('GPU'))}")
print("Forzando ejecución en CPU...")

# Cargar modelos
model_person = YOLO('best_human_model.pt')
print("Modelo YOLO de personas cargado")

# Modelo Keras para generar máscaras
model_keras = load_model_safe('best_model.keras')
if model_keras is None:
    print("Error cargando modelo Keras")
    exit()
print("Modelo Keras cargado")

def preprocess_image_for_keras(image):
    """
    Preprocesa la imagen para el modelo Keras
    Escala a 128x128 como requiere el modelo
    """
    
    resized = cv2.resize(image, (128, 128))
    
    normalized = resized.astype(np.float32) / 255.0
    
    batch_input = np.expand_dims(normalized, axis=0)
    return batch_input

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
            
            rounded_pixels = (class_pixels // 12) * 12
            pixel_tuples = [tuple(pixel.astype(int)) for pixel in rounded_pixels]
            color_counts = Counter(pixel_tuples)
            dominant_color = color_counts.most_common(1)[0][0]

    elif method_to_use == 'median':
        dominant_color = tuple(np.median(class_pixels, axis=0).astype(int))

    else:  
        dominant_color = tuple(np.mean(class_pixels, axis=0).astype(int))

    dominant_color = tuple(max(0, min(255, int(c))) for c in dominant_color)
    return dominant_color, method_to_use, reason

def rgb_to_hex(rgb):
    """Convierte RGB a formato hexadecimal"""
    r, g, b = [max(0, min(255, int(c))) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def find_center_point(mask, class_id):
    """Encuentra el punto central (centroide) de una región segmentada"""
    y_coords, x_coords = np.where(mask == class_id)
    if len(y_coords) == 0:
        return None
    
    
    center_y = int(np.mean(y_coords))
    center_x = int(np.mean(x_coords))
    return (center_x, center_y)

def create_segmentation_overlay(original_image, mask, color_arriba=None, color_abajo=None):
    """
    Crea una visualización de la segmentación sobre la imagen original
    usando los colores reales detectados
    """
    overlay = original_image.copy()
    
    # Colores por defecto si no se detectan
    default_color_arriba = [255, 0, 255]  # Magenta
    default_color_abajo = [0, 255, 255]   # Cyan
    
    # Usar colores detectados o colores por defecto
    if color_arriba is not None:
        # Convertir RGB a BGR para OpenCV
        arriba_bgr = [color_arriba[2], color_arriba[1], color_arriba[0]]
    else:
        arriba_bgr = default_color_arriba
    
    if color_abajo is not None:
        # Convertir RGB a BGR para OpenCV
        abajo_bgr = [color_abajo[2], color_abajo[1], color_abajo[0]]
    else:
        abajo_bgr = default_color_abajo
    
    # Aplicar colores a las máscaras
    prenda_arriba_mask = (mask == 1)
    if np.any(prenda_arriba_mask):
        overlay[prenda_arriba_mask] = overlay[prenda_arriba_mask] * 0.3 + np.array(arriba_bgr) * 0.7
    
    prenda_abajo_mask = (mask == 2)
    if np.any(prenda_abajo_mask):
        overlay[prenda_abajo_mask] = overlay[prenda_abajo_mask] * 0.3 + np.array(abajo_bgr) * 0.7
    
    return overlay.astype(np.uint8)

def create_mask_visualization(mask, size=(200, 200), color_arriba=None, color_abajo=None):
    """
    Crea una visualización coloreada de la máscara usando los colores detectados
    """
    # Colores por defecto
    default_color_arriba = [255, 0, 255]  # Magenta
    default_color_abajo = [0, 255, 255]   # Cyan
    
    # Usar colores detectados o colores por defecto
    if color_arriba is not None:
        # Convertir RGB a BGR para OpenCV
        arriba_bgr = [color_arriba[2], color_arriba[1], color_arriba[0]]
    else:
        arriba_bgr = default_color_arriba
    
    if color_abajo is not None:
        # Convertir RGB a BGR para OpenCV
        abajo_bgr = [color_abajo[2], color_abajo[1], color_abajo[0]]
    else:
        abajo_bgr = default_color_abajo
    
    # Crear máscara visual con colores detectados
    mask_visual = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_visual[mask == 0] = [50, 50, 50]      # Fondo gris oscuro
    mask_visual[mask == 1] = arriba_bgr        # Color detectado para prenda arriba
    mask_visual[mask == 2] = abajo_bgr         # Color detectado para prenda abajo
    
    # Redimensionar para mostrar
    mask_display = cv2.resize(mask_visual, size)
    return mask_display


print("Iniciando webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Webcam iniciada. Presiona ESC para salir.")


last_analysis_time = 0
analysis_interval = 3.0  
current_mask = None
current_overlay = None
color_prenda_arriba = None
color_prenda_abajo = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame.")
        break

    output_frame = frame.copy()
    current_time = time.time()
    
    
    results_person = model_person.predict(source=frame, conf=0.5, verbose=False)
    
    person_detected = False
    person_crop = None
    person_box = None
    
    for r in results_person:
        if r.boxes is not None and len(r.boxes) > 0:
            
            boxes_data = r.boxes.data
            
            for box_data in boxes_data:
                
                x1, y1, x2, y2 = map(int, box_data[:4])
                person_box = (x1, y1, x2, y2)
                
                
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, "Persona detectada", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                person_detected = True
                
                
                person_crop = frame[y1:y2, x1:x2]
                break  
            
            if person_detected:
                break
    
    
    if person_detected and person_crop is not None and person_crop.size > 0:
        
        
        if current_time - last_analysis_time >= analysis_interval:
            print("🔍 Realizando análisis de segmentación...")
            last_analysis_time = current_time
            
            try:
                # Preprocesar para el modelo Keras (128x128)
                keras_input = preprocess_image_for_keras(person_crop)
                
                # Generar máscara con modelo Keras en CPU
                with tf.device('/CPU:0'):
                    predictions = model_keras.predict(keras_input, verbose=0)
                mask = np.argmax(predictions[0], axis=-1)
                
                # Redimensionar máscara al tamaño original de person_crop
                current_mask = cv2.resize(mask.astype(np.uint8), 
                                        (person_crop.shape[1], person_crop.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # Extraer colores dominantes usando el método mejorado
                # Convertir person_crop de BGR a RGB para el análisis de color
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                
                # Redimensionar la imagen RGB para que coincida con la máscara
                person_crop_rgb_resized = cv2.resize(person_crop_rgb, (128, 128))
                
                # Crear overlay de segmentación con colores detectados
                color_prenda_arriba, _, _ = extract_dominant_color(person_crop_rgb_resized, mask, 1, method='auto')
                color_prenda_abajo, _, _ = extract_dominant_color(person_crop_rgb_resized, mask, 2, method='auto')
                
                current_overlay = create_segmentation_overlay(person_crop, current_mask, color_prenda_arriba, color_prenda_abajo)
                
                print("Análisis completado")
                
            except Exception as e:
                print(f"Error en análisis: {e}")
                print(f"Tipo de error: {type(e).__name__}")
                current_mask = None
                current_overlay = None
                color_prenda_arriba = None
                color_prenda_abajo = None
        
        
        if current_overlay is not None and person_box is not None:
            x1, y1, x2, y2 = person_box
            
            
            overlay_resized = cv2.resize(current_overlay, (x2-x1, y2-y1))
            
            
            output_frame[y1:y2, x1:x2] = overlay_resized
            
            
            y_offset = y2 + 25
            if color_prenda_arriba:
                hex_arriba = rgb_to_hex(color_prenda_arriba)
                cv2.putText(output_frame, f"Prenda Arriba: {hex_arriba}", 
                           (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                y_offset += 25
            
            if color_prenda_abajo:
                hex_abajo = rgb_to_hex(color_prenda_abajo)
                cv2.putText(output_frame, f"Prenda Abajo: {hex_abajo}", 
                           (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        
        if current_mask is not None:
            mask_display = create_mask_visualization(current_mask, size=(150, 150), color_arriba=color_prenda_arriba, color_abajo=color_prenda_abajo)
            h, w = output_frame.shape[:2]
            output_frame[10:160, w-160:w-10] = mask_display
            
            
            cv2.putText(output_frame, "Segmentacion", 
                       (w-160, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        
        time_until_next = analysis_interval - (current_time - last_analysis_time)
        if time_until_next > 0:
            cv2.putText(output_frame, f"Proximo analisis: {time_until_next:.1f}s", 
                       (50, output_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if not person_detected:
        cv2.putText(output_frame, "No se detecta persona", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        current_mask = None
        current_overlay = None
        color_prenda_arriba = None
        color_prenda_abajo = None
    
    
    cv2.imshow("Deteccion y Segmentacion de Ropa", output_frame)
    
    
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
print("Aplicación cerrada correctamente")
