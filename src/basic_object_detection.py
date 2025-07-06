from ultralytics import YOLO
import os
# Verifica que pasaste una imagen por consola
# Ruta de imagen

image_file_folder = "yolo_images/"

if not os.path.exists(image_file_folder):
    print(f"El directorio {image_file_folder} no existe. Por favor, crea el directorio y coloca las imagenes del video.")
    exit(1)

# La idea es usar un video en vivo como la webcam y sacar las imágenes de cada frame pero por el momento vamos a usar una imagen de ejemplo

images = []

for filename in os.listdir(image_file_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        images.append(os.path.join(image_file_folder, filename))

image_path = images[0]

# Por el momento solo trabajaremos con una imagen, pero puedes modificar el código para procesar múltiples imágenes o un video completo

# Cargar el modelo preentrenado o uno entrenado por ti
model = YOLO("yolo11n.pt")  # Usa "best.pt" si ya entrenaste con DeepFashion

# Ejecutar detección
results = model(image_path)

# Guardar resultados en una carpeta de salida
output_folder = "yolo_output/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Guardar las imágenes con las detecciones
for i, result in enumerate(results):
    output_path = os.path.join(output_folder, f"result_{i}.jpg")
    result.save(output_path)

# Mostrar resultados
for result in results:
    print("Detecciones:")
    for cls_id, conf, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
        class_id = int(cls_id.item())
        confidence = round(conf.item(), 2)
        bbox = box.tolist()
        print(f"Clase ID: {class_id} | Confianza: {confidence} | BBox: {bbox}")
