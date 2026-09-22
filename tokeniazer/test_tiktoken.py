import tiktoken

# 加载 GPT-4 使用的编码
enc = tiktoken.get_encoding('cl100k_base')

for text in ["I am learing","我喜欢深度学习"]:
    ids = enc.encode(text)
    print("原文本：",text)
    print("Token IDs:",ids)

    # 把每个 id 单独解码，看他对应哪段文字

    for token_id in ids:
        token_bytes = enc.decode_single_token_bytes(token_id)
        print(f"id={token_id:<8}"
              f"bytes={token_bytes}")

    print("完整解码：",enc.decode(ids))



