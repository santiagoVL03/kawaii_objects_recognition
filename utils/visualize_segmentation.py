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

# Carpeta de salida para debug
output_debug_dir = "debug_segmentations"
os.makedirs(output_debug_dir, exist_ok=True)

# División a usar ("train" o "val")
split_name = "train"  # Puedes cambiar a "val"
csv_path = os.path.join(input_csv_path, f"{split_name}.csv")

# Leer el dataset completo
df = pd.read_csv(csv_path)

# Elegir 20 imágenes aleatorias únicas
sample_paths = random.sample(df["path"].unique().tolist(), 20)

# Filtrar el dataframe por las imágenes seleccionadas
df_sample = df[df["path"].isin(sample_paths)]

# Agrupar por imagen
grouped = df_sample.groupby("path")

for path, group in tqdm(grouped, desc="Visualizando segmentaciones (20 muestras)"):
    img_filename = os.path.basename(path) # type: ignore
    img_path = os.path.join(resized_img_path, split_name, img_filename)

    if not os.path.exists(img_path):
        print(f"❌ Imagen no encontrada: {img_path}")
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for _, row in group.iterrows():
            try:
                segmentation = literal_eval(row["segmentation"])
                if not segmentation or not isinstance(segmentation, list):
                    continue

                for seg in segmentation:
                    if isinstance(seg, list) and len(seg) >= 6:  # mínimo 3 puntos
                        # Convertir a pares de coordenadas (x, y)
                        polygon = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                        draw.polygon(polygon, outline="green")

            except Exception as e:
                print(f"Error en segmentación de {img_filename}: {e}")
                continue

        # Guardar imagen con segmentación
        img.save(os.path.join(output_debug_dir, img_filename))

    except Exception as e:
        print(f"❌ Error con imagen {img_filename}: {e}")
        continue

print(f"\nSegmentaciones visualizadas guardadas en: {output_debug_dir}")
