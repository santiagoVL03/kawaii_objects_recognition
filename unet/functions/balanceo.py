import os
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image

images_dir = "segmentation_dataset/train/images"
masks_dir = "segmentation_dataset/train/masks"

output_base = "segmentation_dataset_balanced/train"
images_out = os.path.join(output_base, "images")
masks_out = os.path.join(output_base, "masks")
os.makedirs(images_out, exist_ok=True)
os.makedirs(masks_out, exist_ok=True)

MAX_PER_CLASS = 40000
counts = {1: 0, 2: 0}

mask_files = sorted(os.listdir(masks_dir))

print("🧮 Recorriendo dataset para balancear...")
for fname in tqdm(mask_files):
    mask_path = os.path.join(masks_dir, fname)
    image_path = os.path.join(images_dir, os.path.splitext(fname)[0] + ".jpg")

    if not os.path.exists(image_path):
        continue

    try:
        mask = np.array(Image.open(mask_path))
        labels_in_mask = set(np.unique(mask))

        for label in [1, 2]:
            if label in labels_in_mask and counts[label] < MAX_PER_CLASS:
                shutil.copy(image_path, os.path.join(images_out, os.path.basename(image_path)))
                shutil.copy(mask_path, os.path.join(masks_out, fname))
                counts[label] += 1
                break

        if all(v >= MAX_PER_CLASS for v in counts.values()):
            print("✅ Se alcanzó el límite de 40k por clase.")
            break

    except Exception as e:
        print(f"⚠️ Error al procesar {fname}: {e}")
        continue

print("\n🎯 Proceso finalizado.")
print(f"Total clase 1 (abajo): {counts[1]}")
print(f"Total clase 2 (arriba): {counts[2]}")
