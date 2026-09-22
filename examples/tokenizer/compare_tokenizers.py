from pathlib import Path
import sys

# 底层 tokenizer
from tokenizers import Tokenizer
# 更高级的自动加载器,这个和AutoModel 不同，只加载 tokenizer
from transformers import AutoTokenizer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tokenizer_path = Path(__file__).with_name("my_tokenizer.json")

my_tokenizer = Tokenizer.from_file(str(tokenizer_path))

## 加载 Qwen tokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# 这里的测试集要完全相同
test_texts = [
    "卷积神经网络在图像识别中的表现优异",
    "梯度下降是优化神经网络参数的核心算法"
]

# 统一取 IDs 的函数
def get_ids(tokenizer,text):
    result = tokenizer.encode(text,add_special_tokens=False)

    # 如果 result = Encoding 对象
    if hasattr(result,"ids"):
        return result.ids
    else:
        return result

def get_tokens(tokenizer, text):
    result = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    # HuggingFace tokenizers.Tokenizer
    if hasattr(result, "tokens"):
        return result.tokens

    # transformers.AutoTokenizer
    return tokenizer.convert_ids_to_tokens(result)

def evaluate(tokenizer, texts):
    total_tokens = 0
    total_chars = 0

    for text in texts:
        ids = get_ids(tokenizer, text)

        total_tokens += len(ids)
        total_chars += len(text)

    chars_per_token = total_chars / total_tokens
    tokens_per_char = total_tokens / total_chars

    return chars_per_token, tokens_per_char

tokenizers_to_compare = {
    "My Byte-BPE": my_tokenizer,
    "Qwen2.5": qwen_tokenizer,
    # "Llama3.2": llama_tokenizer
}

print("=" * 60)
print("Tokenizer 对比")
print("=" * 60)

for name, tokenizer in tokenizers_to_compare.items():

    for text in test_texts:
        tokens = get_tokens(tokenizer, text)

        print(f"\n文本：{text}")
        print(f"tokens：{tokens}")
        print(f"token数量：{len(tokens)}")

    chars_per_token, tokens_per_char = evaluate(
        tokenizer,
        test_texts
    )

    print(f"\n{name}")
    print(f"每个 token 平均覆盖字符数: {chars_per_token:.2f}")
    print(f"每个字符平均需要 token 数: {tokens_per_char:.2f}")

