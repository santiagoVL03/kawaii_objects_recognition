import os
import pandas as pd
from ast import literal_eval
from PIL import Image, ImageDraw
from tqdm import tqdm
import random

# Rutas
base_dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/thusharanair/deepfashion2-256x256/versions/2/DeepFashion2 Resized")
input_csv_path = os.path.join(base_dataset_path, "input")
resized_img_path = os.path.join(base_dataset_path, "resized")

# Output de imágenes con bounding boxes visualizadas
output_debug_dir = "debug_boxes"
os.makedirs(output_debug_dir, exist_ok=True)

# Archivo que deseas procesar ("train" o "val")
split_name = "train"  # Cambia a "val" si lo deseas

# Leer CSV y agrupar por imagen
df = pd.read_csv(os.path.join(input_csv_path, f"{split_name}.csv"))
grouped = df.groupby("path")

# Elegir 20 imágenes al azar
sample_paths = random.sample(list(grouped.groups.keys()), 20)

for path in tqdm(sample_paths, desc=f"Visualizando {split_name} (20 muestras)"):
    group = grouped.get_group(path)
    img_filename = os.path.basename(path) # type: ignore
    img_path = os.path.join(resized_img_path, split_name, img_filename)

    if not os.path.exists(img_path):
        print(f"Imagen no encontrada: {img_path}")
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for _, row in group.iterrows():
            try:
                bbox = literal_eval(row["b_box"])  # [x, y, w, h]
                x, y, w, h = map(float, bbox)

                if w <= 0 or h <= 0:
                    continue

                # Dibujar bbox en rojo
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

                # Clase opcional
                class_id = int(row["category_id"])
                draw.text((x, y), str(class_id), fill="yellow")

            except Exception as e:
                print(f"Error procesando {img_filename}: {e}")
                continue

        # Guardar imagen con cajas
        img.save(os.path.join(output_debug_dir, img_filename))

    except Exception as e:
        print(f"Error con imagen {img_filename}: {e}")
        continue

print(f"\n🖼️ Proceso completado. Visualizaciones guardadas en: {output_debug_dir}")
