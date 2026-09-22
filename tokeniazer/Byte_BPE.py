# 从 tokenizers 库导入所需的核心组件
# Tokenizer : 分词器主类，用于组装和使用整个分词流水线
# models : 分词模型(如 BPE,WordPiece,Unigram 等)
# trainers : 训练器，用于从语料库中学习分词规则
# pre_tokenizers : 预分词器，在正式 BPE 合并之前对文本做初步切分
# decoders : 解码器，把 token_id 序列还原回原始字符串
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders

# 选择分词模型：BPE (Byte-Pair Encoding, 字节对编码)
# BPE 的核心思想：
# - 从最小单元(字符或字节)开始，统计语料中相邻符号对的出现频率
# - 反复合并频率最高的一堆，形成新的 subword
# - 直到词表达到指定大小
# 优点:能兼顾“高频完整词”和“低频子词”，对未登录词(OOV)有天然鲁棒性
# GPT-2,GPT-3,GPT-4,LLaMA 等主流大模型都使用 BPE 及其变体
tokenizer = Tokenizer(models.BPE())

# 设置字节级(Byte-Level)预分词器 —— 这是彻底消除 OOV 的关键
# ByteLevel 与分词器的工作原理
# - 先把文本按 UTF-8 编码转成字节序列(每个字符 1~4 字节)
# - 再把每个字节映射到一个“可见字符”(通过 GPT-2 的 bytes_to_unicode 映射表)
# - 由此保证任何字符 —— 中文，emoji,罕见符号，二进制 —— 都能被表示
#
# add_perfix_space = False 的含义 :
# - 不在句首自动加空格
# - GPT-2 原版是 True (因为 GPT-2 用空格区分词首和词中的 token)
# - 如果你的下游任务不需要这种区分，可以设 False
# - 中文场景通常设 Fasle , 因为中文本身不用空格分词
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)

# 解码器也必须用 ByteLevel,才能把编码的字节映射反向还原
# 如果编码器用 ByteLevel 而解码不用，输出会是一堆奇怪的符号
tokenizer.decoder = decoders.ByteLevel()

# 配置 BPE 训练器
trainer  = trainers.BpeTrainer(
    # 目标词表大小 : 32000 是中英混合场景常见的规模
    # 参考 : GPT-2 是 50257，LLaMA 是 32000，BRRT-base 是 30522
    # 词表越大，单个 token 表达能力越强，但 Embedding 矩阵也越大
    vocab_size  = 32000,

    # 特殊 token : 会被放在词表最前面，占据 id 0,1...
    # <|endoftext|> : 文档/序列结束标记，也常兼作 padding 或 BOS
    # <|pad|>:        填充 token , 批量训练时把不同长度的序列补齐
    # 你也可以按需再加 <|user|>,<|assistant|> 等对话角色标记
    special_tokens= ["<|endoftext|>","<|pad|>"],

    # 初始字母表 : 把全部 256 个字节都塞进初始词表
    # 这是“字节级 BPE ”能保证零 OOV 的根本原因:
    #   哪怕一个字符从未在训练语料中出现过，它的 UTF-8 字节一定在词表里
    # ByteLevel.alphabet() 返回的就是这 256 个字节对应的 Unicode 表示
    initial_alphabet= pre_tokenizers.ByteLevel.alphabet(),
)

# 在语料文件上训练
# train() 接受一个文件路径列表，可以同时用多份语料
# 例如：tokenizer.train(['zh_corpus.txt','en_corpus.txt','code.txt'],trainer)
# 文件应该是纯文本，每行一段(或一句) ， UTF-8 编码
# 训练过程：
# 1.按预分词器切分 -> 得到初始 token 序列
# 2.统计所有相邻 token 对的概率
# 3.合并最高频的一对，加入词表
# 4.重复直到词表达到 vocab_size
tokenizer.train(['corpus.txt'],trainer)

# 保存分词器，方便后续直接加载使用
# 保存为单个 JSON 文件，里面包含：
# - 模型类型(BPE)
# - 完整词表(token -> id 映射)
# - 所有合并规则(merges)
# - 预分词器/解码器配置
# - 特殊 token 信息
# 加载时只需 : tokenizer = Tokenizer.from_file("my_tokenizer.json")
tokenizer.save("my_tokenizer.json")

# 打印最终词表大小，应约等于 vocab_size (可能小，取决于合并结果)
print("词表大小：",tokenizer.get_vocab_size())

# 试用: 对一段中英文混合文本编码
texts = [
    "我喜欢深度学习",
    "我喜欢机器学习",
    "我热爱深度学习",
    "深度学习很有意思",
    "我喜欢深度学习算法",
]
for text in texts:
    output = tokenizer.encode(text)

    print("=" * 50)
    print("文本：", text)
    print("tokens：", output.tokens)
    print("token数量：", len(output.ids))
# encoder() 返回一个 Encoding 对象，包含丰富信息：
# - tokens : 切分后的 token 字符串列表(人类可读形式)
# - ids :    每个 token 对应的整数 id (送入模型的实际输入)
# - offsets: 每个token 在原文的起止字符位置 (用于 NER 等任务)
# - attention_mask : 注意力掩码(批处理时用)
# outputs = tokenizer.encode(text)
#
# # tokens 里可能看到特殊符号Ġ，它其实是“空格”的字符级表示
# print("token:",outputs.tokens)
#
# # ids 是最终送入模型 embedding 层的整数序列
# print("ids :",outputs.ids)
# print("decode:", tokenizer.decode(outputs.ids))