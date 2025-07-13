import kagglehub

# Download latest version
path = kagglehub.dataset_download("thusharanair/deepfashion2-256x256")

print("Path to dataset files:", path)