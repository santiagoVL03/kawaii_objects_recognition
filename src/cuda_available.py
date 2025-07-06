import torch
print("GPU disponible:", torch.cuda.is_available())
print("Nombre:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Ninguna")
