import os
import subprocess
import re

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return None

print("=== CUDA CHECK ===")

# 1) CUDA_PATH
cuda_path = os.environ.get("CUDA_PATH")
print("CUDA_PATH:", cuda_path)

# 2) nvcc version
nvcc_out = run("nvcc --version")
print("\n--- nvcc --version ---")
print(nvcc_out if nvcc_out else "nvcc not found")

# parse nvcc version
if nvcc_out:
    m = re.search(r"release\s+([\d.]+)", nvcc_out)
    if m:
        print("✅ CUDA Toolkit Version:", m.group(1))

# 3) nvidia-smi info
smi_out = run("nvidia-smi")
print("\n--- nvidia-smi ---")
print(smi_out if smi_out else "nvidia-smi not found")

# parse driver CUDA version
if smi_out:
    m = re.search(r"CUDA Version:\s*([\d.]+)", smi_out)
    if m:
        print("✅ NVIDIA Driver Supported CUDA Version:", m.group(1))
