import torch

print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))

    x = torch.randn(10000, 10000, device="cuda")
    y = x @ x

    print("Matrix multiplication on GPU succeeded.")
else:
    print("CUDA is NOT available.")