import cv2
import numpy as np
from collections import Counter

def get_dominant_color_name(image):
    """
    Obtiene el nombre del color dominante de una imagen
    """
    if image is None or image.size == 0:
        return "No detectado"
    
    # Convertir a HSV para mejor análisis de color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Obtener píxeles válidos (no muy oscuros)
    mask = hsv[:,:,2] > 30  # Filtrar píxeles muy oscuros
    
    if not np.any(mask):
        return "Negro"
    
    # Extraer valores de Hue de píxeles válidos
    hue_values = hsv[mask, 0]
    saturation_values = hsv[mask, 1]
    value_values = hsv[mask, 2]
    
    colors = []
    
    for h, s, v in zip(hue_values, saturation_values, value_values):
        # Clasificar colores basado en HSV
        if s < 30 or v < 30:  # Colores desaturados o muy oscuros
            if v < 50:
                colors.append("Negro")
            elif v > 200:
                colors.append("Blanco")
            else:
                colors.append("Gris")
        elif h < 10 or h > 170:
            colors.append("Rojo")
        elif h < 25:
            colors.append("Naranja")
        elif h < 35:
            colors.append("Amarillo")
        elif h < 85:
            colors.append("Verde")
        elif h < 125:
            colors.append("Azul")
        elif h < 150:
            colors.append("Violeta")
        else:
            colors.append("Rosa")
    
    # Obtener color más común
    if colors:
        color_counter = Counter(colors)
        dominant_color = color_counter.most_common(1)[0][0]
        return dominant_color
    
    return "No identificado"
