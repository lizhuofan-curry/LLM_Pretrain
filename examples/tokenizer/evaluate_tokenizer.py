from pathlib import Path
import sys

from tokenizers import Tokenizer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def evaluate(tokenizer,texts):
    """
    评估分词器(tokenizer)在一批文本上的效率

    参数：
        tokenizer : 已训练好的分词器对象，需具备 encode() 方法
                    调用 encoder(text).ids 可返回该文本对应的 token id列表
        texts :     一个可迭代对象(通常是 list[str]),包含若干条测试文本

    返回:
        chars_per_token : 每个 token 平均覆盖多少个字符(越大 -> 压缩率越高)
        tokens_per_char ： 每个字符平均被切成多少个 token (越小 -> 越高效)
    """
    total_tokens = 0    # 累计所有文本产生的 token 总数
    total_chars = 0     # 累计所有文本的字符总数(按 Python str 的字符计，一个汉字算 1)

    # 遍历每一条测试文本，分别统计 token数和字符数
    for t in texts:
        # 调用分词器把文本 t 编码成 token，取 .ids 取到 token id 的整数列表
        # 例如 “你好” 可能被编码成 [1234,5678]
        ids = tokenizer.encode(t).ids

        total_tokens += len(ids)    # 该条文本被切成的 token 数，累加到总数
        total_chars += len(t)       # 该条文本的字符数(len(str),返回字符个数)，累加到总数

    # ------- 计算两个核心指标 -----------
    # 指标 1 : 平均每个 token 覆盖多少字符 = 总字符数/总 token数
    #          值越大，说明一个 token 承载的信息量过多，词表压缩效果越好
    #          英文 BPE 一般在 3~4 之间，中文若做的好可以 > 1.5,甚至接近 2
    chars_per_token = total_chars / total_tokens

    # 指标 2 : 平均每个字符被切成多少 token = 总 token数 / 总字符数
    #          它是指标1的倒数 ，值越小越高效
    #          中文场景更常用这个指标来对比不同分词器
    #          例如 0.6 表示平均 5 个汉字只需 3个 token
    tokens_per_char = total_tokens / total_chars
    return chars_per_token, tokens_per_char
if __name__ == "__main__":

    tokenizer_path = Path(__file__).with_name('my_tokenizer.json')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    test_texts = [
        "卷积神经网络在图像识别中的表现优异",
        "梯度下降是优化神经网络参数的核心算法"
    ]

    # 调用评估函数，拿到两个指标
    cpt,tpc = evaluate(tokenizer,test_texts)

    # 打印结果，保留 2 位小数便于阅读
    print(f"每个 token 平均 {cpt:.2f}个字符")  # 越大越好
    print(f"每个字符平均 {tpc:.2f}个 token")   # 越小越好
