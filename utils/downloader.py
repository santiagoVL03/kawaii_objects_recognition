import kagglehub
import os

# Download latest version
path_download = "./datasets/deepfashion2-256x256"

if not os.path.exists(path_download):
    os.makedirs(path_download)
    
path = kagglehub.dataset_download("thusharanair/deepfashion2-256x256")
print("Path to dataset files:", path)