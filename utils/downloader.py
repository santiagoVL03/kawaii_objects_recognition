import kagglehub

# Download latest version
path = kagglehub.dataset_download("pengcw1/market-1501")

print("Path to dataset files:", path)