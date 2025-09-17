import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# Force Torch to use GPU:1
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def print_peak_gpu_memory():
    peak = torch.cuda.max_memory_allocated()/(1024**2)
    print(f"Peak GPU memory: {peak:.2f} MB")
    torch.cuda.reset_peak_memory_stats()

def print_gpu_utilization():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f"GPU memory used: {info.used//(1024**2)} MB")
