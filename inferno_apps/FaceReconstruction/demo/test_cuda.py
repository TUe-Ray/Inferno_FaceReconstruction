# PyTorch3D 安裝驗證（純 Python 版）
# 直接以 `python this_file.py` 執行，或在 Notebook/VS Code 中跑整段即可。

import os
import subprocess

try:
    import torch
except Exception as e:
    print("[Error] 無法匯入 torch：", e)
    raise

print(f"torch = {torch.__version__} | torch.cuda = {getattr(torch.version, 'cuda', None)}")
print("CUDA_HOME =", os.environ.get("CUDA_HOME"))
print("CUDA available =", torch.cuda.is_available())

# 額外：顯示 nvcc 版本（若系統有安裝且在 PATH 中）
def show_nvcc_version():
    print(" --- nvcc --version ---")
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        print(out.strip())
    except FileNotFoundError:
        print("找不到 nvcc（可能未安裝 CUDA Toolkit 或不在 PATH）。")
    except Exception as e:
        print("執行 nvcc 時發生錯誤：", e)

# 主要：驗證 PyTorch3D 原生擴充 (_C) 是否能載入
try:
    from pytorch3d import _C  # 若安裝與編譯成功，應可載入
    print("_C loaded:", _C is not None)
except Exception as e:
    print("[Error] 無法匯入 pytorch3d._C：", e)

show_nvcc_version()
