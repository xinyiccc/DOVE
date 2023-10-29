import torch

if torch.cuda.is_available():
    print("GPU is available")
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("GPU memory usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0)/1024**3,1), "GB")
else:
    print("GPU is not available")