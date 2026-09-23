# LLM Pretrain Lab

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_13.0-EE4C2C?logo=pytorch&logoColor=white)
![uv](https://img.shields.io/badge/env-uv-6E56CF)
![Status](https://img.shields.io/badge/status-learning_in_public-16A34A)
[![GitHub Discussions](https://img.shields.io/github/discussions/lizhuofan-curry/LLM_Pretrain?logo=github)](https://github.com/lizhuofan-curry/LLM_Pretrain/discussions)

> 从一个 BPE 分词器开始，亲手搭出大模型预训练的完整知识链。

这是 **lizhuofan-curry 的 LLM 预训练学习实验室**。仓库不追求一上来就堆出一个庞大的训练框架，而是把 Tokenizer、数据、模型、训练系统和评测拆成可以理解、运行和验证的小实验。

当前重点：**先真正理解数据如何变成 token，以及模型训练为什么受显存、算力、带宽和多卡通信共同约束。**

## ✨ 为什么值得看

- **从原理到工具**：既有手写 `SimpleBPE`，也有 Hugging Face `tokenizers` 的 Byte-Level BPE。
- **不是只贴代码**：每个实验保留中文注释，强调中间过程和设计原因。
- **环境可复现**：使用 `uv`、`pyproject.toml` 和 `uv.lock` 固定 Python 与依赖。
- **面向真实训练**：从 RAM/VRAM、FLOPs、MFU、带宽一路讲到 DDP 的 `rank` 与 `world_size`。
- **持续生长**：Tokenizer 只是起点，后续将沿数据工程、Transformer、预训练和评测继续扩展。

## 🧭 学习路线

```text
文本
  ↓
Tokenizer：BPE / Byte-Level BPE
  ↓
Token IDs + 训练语料
  ↓
Transformer 语言模型
  ↓
单卡训练 → 混合精度 → 多卡训练
  ↓
评测、记录与复现实验
```

| 阶段 | 内容 | 状态 |
| --- | --- | :---: |
| 00 | Python、uv、PyTorch/CUDA 环境检查 | ✅ |
| 01 | 手写最小 BPE，理解“统计—合并—编码—解码” | ✅ |
| 02 | 训练 Byte-Level BPE，消除普通字符级 BPE 的 OOV 问题 | ✅ |
| 03 | 与 tiktoken、Qwen Tokenizer 做基础对比 | ✅ |
| 04 | 显存、FLOPs、MFU、带宽和分布式训练前置 | ✅ |
| 05 | 语料清洗、去重与数据管线 | 🚧 |
| 06 | 从零实现 Transformer 语言模型 | ⏳ |
| 07 | 单卡/多卡预训练与评测 | ⏳ |

> ✅ 已有内容　🚧 正在扩展　⏳ 计划中

## 🚀 快速开始

### 1. 克隆仓库

```powershell
git clone https://github.com/lizhuofan-curry/LLM_Pretrain.git
cd LLM_Pretrain
```

### 2. 确认显卡驱动

```powershell
nvidia-smi
```

当前项目默认使用 **PyTorch CUDA 13.0** 构建。`nvidia-smi` 中显示的 `CUDA Version` 表示驱动最高支持的 CUDA 版本，不等同于本机一定安装了对应 CUDA Toolkit。

没有 NVIDIA GPU 的读者仍可阅读和运行纯 Python 的 `SimpleBPE`；完整依赖配置目前面向 CUDA 13.0 环境。

### 3. 用 uv 同步环境

```powershell
uv sync
```

项目固定使用 Python 3.11。`uv` 会依据：

- `.python-version` 选择 Python；
- `pyproject.toml` 读取直接依赖；
- `uv.lock` 恢复解析后的精确依赖版本；
- `.venv` 保存项目私有环境。

### 4. 验证 PyTorch 和 GPU

```powershell
uv run python check_env.py
```

你会看到 Python、PyTorch、CUDA Runtime、GPU 名称、显存和 BF16 支持情况。真正判断 PyTorch 能否使用 GPU，要看：

```python
torch.cuda.is_available()
```

而不能只看 `nvidia-smi`。

## 🧪 Tokenizer 实验

### 实验一：手写最小 BPE

```powershell
uv run python examples/tokenizer/simple_bpe.py
```

重点观察：

1. 如何统计词频；
2. 如何寻找最高频相邻符号对；
3. 合并规则为什么必须按学习顺序保存；
4. 编码与解码如何复用这些规则；
5. 普通字符级 BPE 为什么仍可能遇到 OOV。

### 实验二：Byte-Level BPE

```powershell
uv run python examples/tokenizer/byte_bpe.py
```

它会读取同目录的 `corpus.txt`，训练分词器并生成 `my_tokenizer.json`。Byte-Level 方案把文本转换为 UTF-8 字节，借助完整的 256 字节初始字母表覆盖中文、英文、emoji 和罕见符号。

### 实验三：基础效率评估

```powershell
uv run python examples/tokenizer/evaluate_tokenizer.py
```

当前示例关注两个直观指标：

- `chars_per_token`：每个 token 平均覆盖多少字符，越大通常表示压缩率越高；
- `tokens_per_char`：每个字符平均需要多少 token，越小通常表示越紧凑。

这些只是基础统计，不能单独代表一个 Tokenizer 的完整质量。真实评估还应覆盖多语言、代码、数字、特殊符号、长尾字符和下游任务表现。

### 实验四：与 Qwen Tokenizer 对比

```powershell
uv run python examples/tokenizer/compare_tokenizers.py
```

首次运行会从 Hugging Face 下载 `Qwen/Qwen2.5-7B` 的 Tokenizer 文件，需要联网。

## 🧠 系统知识前置

开始模型预训练前，建议先读：

### [模型预训练系统前置](docs/模型预训练系统前置.md)

它回答这些经常被混在一起的问题：

- RAM 和 VRAM 到底有什么区别？
- 为什么一个 1B 模型训练时远不止占 2 GB？
- FLOPs 和 FLOPS 为什么不是一回事？
- MFU 在衡量什么？
- 算力强为什么仍可能跑不快？
- `rank`、`local_rank`、`world_size` 和 `process group` 分别是什么？

## 📁 仓库结构

```text
LLM_Pretrain/
├── docs/
│   └── 模型预训练系统前置.md     # 训练系统基础概念
├── examples/
│   └── tokenizer/
│       ├── simple_bpe.py          # 从零实现的最小 BPE
│       ├── byte_bpe.py            # Hugging Face Byte-Level BPE
│       ├── evaluate_tokenizer.py  # 基础压缩效率评估
│       ├── compare_tokenizers.py  # 与 Qwen Tokenizer 对比
│       ├── test_tiktoken.py       # tiktoken 编解码观察
│       ├── corpus.txt             # 教学用微型语料
│       └── my_tokenizer.json      # 示例训练产物
├── src/llm_pretrain/              # 后续可复用的训练包
├── check_env.py                   # Python/PyTorch/CUDA 自检
├── pyproject.toml                 # 项目与直接依赖声明
├── uv.lock                        # 可复现的精确依赖锁定
└── .python-version                # Python 3.11
```

## 🛡️ 仓库卫生

以下内容默认不会进入 Git：

- `.env`、私钥和本地配置；
- `.venv`、缓存和 IDE 文件；
- 原始数据集与 Hugging Face 缓存；
- `wandb` 日志和普通日志；
- `.pt`、`.pth`、`.ckpt`、`.safetensors` 等模型权重；
- `checkpoints/`、`outputs/` 等训练产物。

如果未来确实需要发布大模型权重或数据，请使用 Hugging Face Hub、对象存储或 Git LFS，而不是直接提交到普通 Git 历史中。

## 📌 当前边界

这个仓库目前处于“打牢基础并逐步实现”的阶段：

- Tokenizer 示例使用的是教学级小语料，不代表真实生产质量；
- `6ND`、每参数约 16 Bytes 等都是估算模型，不是所有训练配置下的固定常数；
- 当前还没有提交完整 Transformer 预训练、分布式启动和基准结果；
- README 只记录已经存在或明确规划的内容，不虚构训练指标。

## 🤝 参与讨论

如果你也在从零学习 LLM 预训练，欢迎来到 [GitHub Discussions](https://github.com/lizhuofan-curry/LLM_Pretrain/discussions)：

- **Q&A**：询问 uv、Tokenizer、数据工程、Transformer 或训练系统相关问题；
- **Ideas**：提出新的学习路线、实验设计或仓库改进建议；
- **Show and tell**：分享你的复现过程、Tokenizer 对比或训练结果；
- **General**：交流学习心得，一起把“会运行”推进到“真正理解”。

如果你发现了能够稳定复现的代码错误、文档错误，或有边界明确的开发任务，请提交 [Issue](https://github.com/lizhuofan-curry/LLM_Pretrain/issues)。

---

**Build it. Measure it. Understand it.**

这个仓库会随着我的学习持续更新。
