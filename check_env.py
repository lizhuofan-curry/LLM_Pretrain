import sys
import torch

print("=" * 50)
print("环境检查")
print("=" * 50)

# 1. Python
print("Python 版本:", sys.version.split()[0])

# 2. PyTorch
print("PyTorch 版本:", torch.__version__)

# 3. PyTorch 所使用的 CUDA Runtime
print("CUDA Runtime:", torch.version.cuda)

# 4. GPU 是否可用
print("CUDA 是否可用:", torch.cuda.is_available())

# 5. 如果 GPU 可用，再获取 GPU 信息
if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())
    print("当前 GPU:", torch.cuda.get_device_name(0))

    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / 1024 ** 3

    print(f"GPU 显存: {total_memory_gb:.2f} GB")

    # 检查 bf16 支持
    print("BF16 支持:", torch.cuda.is_bf16_supported())

print("=" * 50)