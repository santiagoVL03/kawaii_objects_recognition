import os
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

# Rutas
base_dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/thusharanair/deepfashion2-256x256/versions/2/DeepFashion2 Resized")
input_csv_path = os.path.join(base_dataset_path, "input")
split_name = "train"  # o "val"

csv_file = os.path.join(input_csv_path, f"{split_name}.csv")
df = pd.read_csv(csv_file)

# Lista donde almacenaremos los nuevos bbox
bbox_fixed = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Corrigiendo bboxes"):
    try:
        segmentations = literal_eval(row["segmentation"])

        if not segmentations or not isinstance(segmentations, list):
            bbox_fixed.append(None)
            continue

        # Combinar todos los puntos de todos los polígonos
        all_x, all_y = [], []
        for seg in segmentations:
            if isinstance(seg, list) and len(seg) >= 6:
                coords = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                xs, ys = zip(*coords)
                all_x.extend(xs)
                all_y.extend(ys)

        if not all_x or not all_y:
            bbox_fixed.append(None)
            continue

        x_min = min(all_x)
        y_min = min(all_y)
        x_max = max(all_x)
        y_max = max(all_y)
        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0:
            bbox_fixed.append(None)
        else:
            bbox_fixed.append([x_min, y_min, w, h])

    except Exception as e:
        bbox_fixed.append(None)

# Agregar columna al DataFrame
df["bbox_fixed"] = bbox_fixed

# Guardar CSV actualizado
output_path = os.path.join(input_csv_path, f"{split_name}_with_bbox_fixed.csv")
df.to_csv(output_path, index=False)
print(f"\nBounding boxes corregidos guardados en: {output_path}")
