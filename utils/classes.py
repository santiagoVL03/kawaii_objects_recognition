import pandas as pd
import os

base_dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/thusharanair/deepfashion2-256x256/versions/2/DeepFashion2 Resized")
input_csv_path = os.path.join(base_dataset_path, "input")

# Leer y combinar train y validation
df_train = pd.read_csv(os.path.join(input_csv_path, "train.csv"))
df_val = pd.read_csv(os.path.join(input_csv_path, "validation.csv"))

df_combined = pd.concat([df_train, df_val], ignore_index=True)

# Eliminar duplicados para obtener solo los nombres distintos por ID
clases_df = df_combined[["category_id", "category_name"]].drop_duplicates().sort_values("category_id")

# Mostrar resumen
print(f"Total de clases: {len(clases_df)}")
print("\nLista de clases:")
for i, row in clases_df.iterrows():
    print(f"{row['category_id']}: {row['category_name']}")

# Si quieres la lista en formato lista YAML
names_list = clases_df["category_name"].tolist()
print("\nPara tu .yaml:")
print(f"names: {names_list}")
