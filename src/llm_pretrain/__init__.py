import sys


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("LLM Pretrain Lab")
    print("从 Tokenizer 出发，逐步走向可复现的大模型预训练。")
    print("\n建议从这里开始：")
    print("  uv run python check_env.py")
    print("  uv run python examples/tokenizer/simple_bpe.py")
    print("  阅读 docs/模型预训练系统前置.md")
