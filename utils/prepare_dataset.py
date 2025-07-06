import os
import shutil
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

# Rutas base
base_dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/thusharanair/deepfashion2-256x256/versions/2/DeepFashion2 Resized")
input_csv_path = os.path.join(base_dataset_path, "input")
resized_img_path = os.path.join(base_dataset_path, "resized")
output_base = "datasets/clothes"
log_path = os.path.join(output_base, "corrupt_labels.txt")

# Splits del dataset
splits = {"train": "train", "validation": "val"}

# Crear carpetas
for split_dir in splits.values():
    os.makedirs(os.path.join(output_base, f"images/{split_dir}"), exist_ok=True)
    os.makedirs(os.path.join(output_base, f"labels/{split_dir}"), exist_ok=True)

# Log para imágenes con coordenadas corruptas
corrupt_log = []

def bbox_valida(xc, yc, w, h):
    return all(0.0 <= v <= 1.0 for v in [xc, yc, w, h]) and w > 0 and h > 0

# Procesar cada archivo CSV
for split_csv, split_dir in splits.items():
    print(f"\n📂 Procesando '{split_csv}.csv'...")
    df = pd.read_csv(os.path.join(input_csv_path, f"{split_csv}.csv"))

    grouped = df.groupby("path")

    for path, group in tqdm(grouped, total=len(grouped)):
        img_filename = os.path.basename(path)  # type: ignore
        label_filename = os.path.splitext(img_filename)[0] + ".txt"

        src_img_path = os.path.join(resized_img_path, split_csv, img_filename)
        dst_img_path = os.path.join(output_base, f"images/{split_dir}", img_filename)
        dst_label_path = os.path.join(output_base, f"labels/{split_dir}", label_filename)

        if not os.path.exists(src_img_path):
            print(f"[❌] Imagen no encontrada: {src_img_path}")
            continue

        shutil.copy(src_img_path, dst_img_path)

        try:
            img_w = 256.0
            img_h = 256.0
        except:
            print(f"[❌] Error leyendo dimensiones de {img_filename}")
            continue

        label_lines = []
        image_corrupt = False

        for _, row in group.iterrows():
            try:
                bbox = literal_eval(row["b_box"])  # [x, y, w, h]
                x, y, w, h = map(float, bbox)

                # Evitar divisiones por cero o anchos/altos negativos
                if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
                    image_corrupt = True
                    continue

                # Normalizar
                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                wn = w / img_w
                hn = h / img_h

                if not bbox_valida(xc, yc, wn, hn):
                    image_corrupt = True
                    corrupt_log.append(
                        f"{img_filename} - OUT OF BOUNDS: xc={xc:.3f}, yc={yc:.3f}, w={wn:.3f}, h={hn:.3f}, img_w={img_w}, img_h={img_h}, bbox={bbox}"
                    )
                    continue

                # Corregir índice (clase 1-13 -> 0-12)
                class_id = int(row["category_id"]) - 1
                if not (0 <= class_id < 13):
                    corrupt_log.append(f"{img_filename} - CLASE FUERA DE RANGO: {class_id+1}")
                    continue

                label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            except Exception as e:
                corrupt_log.append(f"{img_filename} - ERROR: {e}")
                continue

        if label_lines:
            with open(dst_label_path, "w") as f:
                f.write("\n".join(label_lines) + "\n")

# Guardar log final
if corrupt_log:
    with open(log_path, "w") as f:
        f.write("\n".join(corrupt_log))
    print(f"\n📄 Guardado log de imágenes corruptas en: {log_path}")
else:
    print("\n✅ No se detectaron imágenes con coordenadas fuera de rango.")
print("\n✅ Proceso completado exitosamente.")