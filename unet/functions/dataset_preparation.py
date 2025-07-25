import os
import shutil
import pandas as pd
from PIL import Image, ImageDraw
from ast import literal_eval
from tqdm import tqdm
import numpy as np

# Configuración general
base_dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/thusharanair/deepfashion2-256x256/versions/2/DeepFashion2 Resized")
input_csv_path = os.path.join(base_dataset_path, "input")
resized_img_path = os.path.join(base_dataset_path, "resized")
output_base = "segmentation_dataset"

splits = {"train": "train", "validation": "validation"}
IMG_SIZE = (256, 256)  # Assumed target size

# Clases agrupadas por tipo
superior_ids = {0, 1, 2, 3, 4, 5, 9, 10, 11, 12}  # categorías de tops y vestidos
inferior_ids = {6, 7, 8}                         # shorts, trousers, skirt

# Crear carpetas de salida
for split in splits.values():
    os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, "masks"), exist_ok=True)

# Procesar cada split
for split_name_csv, split_folder in splits.items():
    print(f"\n📂 Procesando {split_name_csv}.csv...")
    df = pd.read_csv(os.path.join(input_csv_path, f"{split_name_csv}.csv"))
    grouped = df.groupby("path")

    for img_rel_path, group in tqdm(grouped, total=len(grouped)):
        img_filename = os.path.basename(img_rel_path) # type: ignore
        img_path = os.path.join(resized_img_path, split_name_csv, img_filename)

        if not os.path.exists(img_path):
            print(f"Imagen no encontrada: {img_path}")
            continue

        # Cargar imagen y crear máscara vacía
        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.new("L", IMG_SIZE, 0)  # 'L' = 8-bit pixels, black and white
            draw = ImageDraw.Draw(mask)

            for _, row in group.iterrows():
                try:
                    segmentation = literal_eval(row["segmentation"]) # type: ignore
                    class_id = int(row["category_id"]) - 1  # type: ignore # Convertir a 0-12
                    if class_id < 0 or class_id > 12:
                        continue  # Ignorar clases fuera del rango
                    label = 0  # Fondo
                    if class_id in superior_ids:
                        label = 1  # Superior
                    elif class_id in inferior_ids:
                        label = 2 # Inferior
                    else:
                        continue  # Ignorar clases no deseadas

                    for polygon in segmentation:
                        if isinstance(polygon, list) and len(polygon) >= 6:
                            xy = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                            draw.polygon(xy, fill=label)

                except Exception as e:
                    print(f"Error procesando segmentación para {img_filename}: {e}")
                    continue

            # Guardar imagen y máscara
            out_img_path = os.path.join(output_base, split_folder, "images", img_filename)
            out_mask_path = os.path.join(output_base, split_folder, "masks", os.path.splitext(img_filename)[0] + ".png")

            img.save(out_img_path)
            mask.save(out_mask_path)

        except Exception as e:
            print(f"Error procesando {img_filename}: {e}")
            continue

print("\nDataset preparado para segmentación.")
