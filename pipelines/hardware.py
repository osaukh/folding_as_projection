import platform
import torch
import subprocess

print("=== System ===")
print("Hostname:", platform.node())
print("OS:", platform.platform())
print("Python:", platform.python_version())

print("\n=== CPU ===")
try:
    out = subprocess.check_output(["lscpu"], text=True)
    print(out)
except Exception as e:
    print("lscpu not available:", e)

print("=== RAM ===")
try:
    out = subprocess.check_output(["free", "-h"], text=True)
    print(out)
except Exception as e:
    print("free not available:", e)

print("=== GPU / CUDA ===")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, {props.total_memory / (1024**3):.1f} GB VRAM")
    print("cuDNN version:", torch.backends.cudnn.version())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA (build):", torch.version.cuda)