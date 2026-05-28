import torch

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
