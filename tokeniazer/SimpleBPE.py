from collections import defaultdict

class SimpleBPE:
    '''
    一个最小可运行的 BPE 分词器 (词级 + 字符级 )
    真实分词器会用字节级 + 更复杂的预切分
    BPE 核心思想 ：通过迭代合并最高频的相邻符号，逐步构建词表
    例如：初始 (l,o) 频次最高 -> 合并成 lo -> 然后 (lo,w) 频次最高 -> 合并成 low
    '''
    def __init__(self):
        self.merges = []    # 合并规则列表，按学习顺序存放
        # 例如 : [('e','s'),('es','t'),('est','</w>'),......]
        # 这个顺序很关键！编码时必须按照这个顺序套用规则

        self.vocab = {}  # token 字符串 -> id 的映射表
        # 例如 : {'l':0 ,'o': 1, 'w':2,'</w>':3,'lo':4}

        self.id_to_token = {} # id -> token 字符串的反向映射
        # 用于解码 : 把 id 还原回 token

    def _get_pair_freqs(self,splits,word_freqs):
        '''
        统计所有相邻符号的频次 (按词频加权)
        关键概念：
        - splits : 当前每个词的拆分结果,例如 {“low”:['l','o','w',</w>]}
        - word_freqs : 每个词在语料中出现的次数，例如 {“low”:5,......}
        - 返回值 : 每个符号出现的“加权频次”

        例如：
        - “newest” 出现 6 次，里面有符号对 (e,s),那么 pair_freqs[(e,s)] += 6
        - "widest" 出现 3 次， 里面也有 (e,s), 那么 pair_freqs[(e,s)] += 3
        - 最终 pair_freqs[(e,s)] = 6 + 3 =9
        这样做的原因 ： 高频词内的符号对应优先被合并，这样能更有效压缩文本
        '''
        pair_freqs = defaultdict(int) # 初始化为 0 , 防止 KeyError

        for word,freq in word_freqs.items():
            symbols = splits[word]  # 去这个词当前的符号序列
            # 例如 “newest” = ['n','e','w','e','s','t','</w>']

            # 遍历相邻的两个符号，统计所有的符号对
            for i in range (len(symbols)-1):
                pair = (symbols[i],symbols[i+1])
                pair_freqs[pair] += freq    # 累加这个词的出现次数
                # 这样做保证了高频词贡献更多的频次

        return pair_freqs

    def _merge_pair(self,pair,splits,word_freqs):
        """
        把指定的符号对合并成一个新符号，更新所有词的拆分

        例如: 合并 (e,s) 时
        - “newest” 中的 ['n','e','w','e','s','t','</w>']
        - 会变成 ['n','e','w','es','t','</w>'] (第 3,4 个位置的 'e','s' 合并成 'es')

        参数：
        - pair : 要合并的符号对，例如 ('e','s')
        - splits : 所有词的当前拆分，会被原地修改
        - word_freqs : 词汇表 (这里用来遍历所有词)

        返回 : 更新后的 splits 字典
        """
        a,b = pair # 拆包 a = 'e' , b = 's'

        for word in word_freqs:
            symbols = splits[word]  # 取当前词的符号序列
            new_symbols = []    # 保存合并后的新序列
            i = 0

            # 逐个扫描符号，遇到要合并的那对就合并
            while i < len(symbols) :
                # 检查当前位置及下一个位置是否恰好是要合并的那一对
                if i < len(symbols) -1 and symbols[i] == a and symbols[i+1] == b:
                    new_symbols.append(a+b)
                    # 例如 ‘e’+'s' = 'es'
                    i += 2  # 跳过被合并的两个符号
                else :
                    # 不是要合并的那一对，原样保留
                    new_symbols.append(symbols[i])
                    i += 1

            splits[word] = new_symbols  # 更新这个词的拆分结果
        return splits

    def train(self,corpus,num_merges,verbose= True):
        '''
        在语料上训练,学出合并规则表和词表

        参数 :
        - corpus : 字符串列表，例如 ["low low low low low","lower lower",......]
        - num_merges : 要进行多少次合并 (决定最终词表大小)
        - verbose : 是否打印详细的训练过程

        返回 : 无。但会设置 self.merges , self.vocab , self.id_to_token

        流程：
        1.统计词频 (按空格切词)
        2.把每个词拆成字符，加上 </w> 结尾标记
        3.迭代 num_merges 次：
          - 统计所有相邻符号对的频次
          - 找出频次最高的那一对
          - 合并他
          - 记录这条规则
        4. 给词表中的所有 token 分配 id
        '''

        # ========= 第一步 : 统计词频 =========
        word_freqs = defaultdict(int)
        for line in corpus:
            for word in line.strip().split():
                word_freqs[word] += 1
        print(f'语料词数：{len(word_freqs)}')
        print(f"语料词频示例：{dict(list(word_freqs.items())[:5])}")

        # ======== 第二步 : 初始拆分 ==========
        # 每个词拆成单个字符，末尾加 </w> 标记
        # 例如 : 'low' -> ['l','o','w','</w>']
        # </w> 的作用是区分词尾和词中间的符号 (比如词尾的 “st” vs 词中间的 “st”)
        splits = {}
        for word in word_freqs :
            splits[word] = list(word) + ['</w>']
        print("初始拆分示例:")
        for word in list(word_freqs.keys())[:4]:
            print(f"{word:10s} -> {splits[word]}")
        print()

        # ========= 第三步 : 初始词表 ============
        # 收集所有出现过的“原始字符”，包括 </w>
        # set 实现自动去重
        vocab = set()
        for symbols in splits.values():
            vocab.update(symbols)

        print(f'初始词表大小（只有单字符）：{len(vocab)}')

        # ========== 第四步 : 迭代合并 ===========
        # 这是 BPE 的核心 : 反复找出最高频的相邻对，合并他
        for step in range(num_merges):
            # 计算当前所有相邻符号对的频次
            pair_freqs = self._get_pair_freqs(splits,word_freqs)

            if not pair_freqs:
                # 没有可合并的对了 (每个词都已经是单个符号)
                print(f'提前停止 : 在第 {step} 步时已无可合并的对')
                break

            # 选出频次最高的那一对
            # max() 在并列时返回字典遍历顺序中最先遇到的那个
            # 在我们的例子里,(e,s) 最先出现，所以它会被选中
            best_pair = max(pair_freqs,key = pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            # 进行合并，更新所有词的拆分
            splits=self._merge_pair(best_pair,splits,word_freqs)

            # 生成新的 token (把两个字符串拼接)
            new_token = best_pair[0] + best_pair[1]

            # 记录这条合并规则 (编码时需要用到)
            self.merges.append(best_pair)

            # 把新 token 加入词表
            vocab.add(new_token)

            if verbose:
                print(f"步骤 {step + 1 :2d} : 合并 {best_pair} -> {new_token}"
                      f"频次: {best_freq}")
                # 可选：显示几个词的合并后样子
                if step < 3 : # 只显示前 3 步的例子
                    example_word = list(word_freqs.keys())[2]   # 比如显示 "newest"
                    print(f"            示例 :{example_word} -> {splits[example_word]}")

        # =========== 第五步 : 词表编号 ===========
        # 给词表中的每个 token 分配一个唯一的 id
        # 按字母顺序排序，保证可复现性
        for i,token in enumerate(sorted(vocab)):
            self.vocab[token] = i   # token -> i
            self.id_to_token[i] = token  # id -> token

        if verbose:
            print(f"\n训练完成：")
            print(f"  合并次数： {len(self.merges)}")
            print(f"  词表大小：  {len(self.vocab)}")
            print(f"  前 10 个 token : {sorted(vocab)[:10]}")

    def encode(self,text):
        '''
        把文本编码成 token id 列表

        关键思想: 按训练时学到的合并规则顺序来编码
        - 先套用第一条规则:合并 (e,s)
        - 再套用第二条规则:合并 (es,t)
        - 再套用第三条规则:合并 (est,</w>)
        - .....

        这样才能和训练时的符号表对齐

        例如编码 “newest”
        初始 : ['n','e','w','e','s','t','</w>']
        套用规则 1 (e,s)->es:['n','e','w','es','t','</w>']
        套用规则 2 (es,t)->est:['n','e','w','est','</w>']
        套用规则 3 (est,</w>):['n','e','w',est</w>']
        最后转成 id : [id_n,id_e,id_w,id_est</w>]
        '''
        ids = []

        for word in text.strip().split():   # 按空格切词
            # 初始拆分: 字符 + </w>
            symbols = list(word) + ['</w>']

            # 关键 : 按学习顺序，一条一条套用合并规则
            # 这保证了符号序列和训练时一致
            for a,b in self.merges: # 遍历所有合并规则
                i = 0
                new_symbols = []

                # 在当前符号序列中查找并合并符号对
                while i < len(symbols) :
                    if i < len(symbols) -1 and symbols[i] == a and symbols[i+1] == b:
                        new_symbols.append(a+b)
                        i += 2
                    else:
                        # 不是要合并的，原样保留
                        new_symbols.append(symbols[i])
                        i += 1

                symbols = new_symbols  # 更新符号序列，为下一条规则做准备
            # 现在 symbols 中的每个元素都应该在词表里了
            # (除非遇到训练时没见过的字符，会报 KeyError)
            for s in symbols:
                ids.append(self.vocab[s])   # 查词表，转成id
        return ids

    def decode(self,ids):
        '''
        把 token id 列表还原回文本

        流程:
        1. 把 id 转回 token 字符串
        2. 把所有 token 拼接在一起
        3. 把 </w> 替换成空格 (还原词边界)

        例如 :
        [id_n,id_e,id_w,id_est</w>]
        -> ['n','e','w','est</w>']
        -> 'newest' (</w>替换成空格，然后 strip)
        '''
        tokens = [self.id_to_token[i] for i in ids]    # id -> tokens
        text = ''.join(tokens) # 拼接所有 token
        text = text.replace('</w>',' ') # </w> 还原成空格 (词边界)
        return text.strip() # 去掉首尾空格

# ============= 运行演示 =================
if __name__ == "__main__":
    print("=" * 60)
    print("SimpleBPE 分词器演示")
    print("=" * 60)
    print()

    # 构造一个小语料 （用重复来模拟词频）
    corpus = [
        "low low low low low",   # 'low 出现 5 次'
        "lower lower",  # "lower" 出现 2 次
        "newest newest newest newest newest newest",   # "newest" 出现 6 次
        "widest widest widest" # "widest" 出现 3 次
    ]

    bpe = SimpleBPE()
    print("【训练阶段】")
    bpe.train(corpus,num_merges=5)

    print("\n" + "=" * 60)
    print("【编码/解码测试】")
    print("=" * 60)
    print()

    # 测试 1 ： 训练语料中的词
    text1 = "low newest"
    ids1 = bpe.encode(text1)
    tokens1 = [bpe.id_to_token[i] for i in ids1]
    print(f"输入文本：{text1}")
    print(f"编码结果：{ids1}")
    print(f"对应的 token：{tokens1}")
    print(f"解码结果：'{bpe.decode(ids1)}'")
    print()

    # 测试 2 : 训练语料中没有的词 (但由训练过的符号组成)
    text2 = "lowest"
    try:
        ids2 = bpe.encode(text2)
        tokens2 = [bpe.id_to_token[i] for i in ids2]
        print(f'编码结果: {ids2}')
        print(f'对应的token : {tokens2}')
        print(f"解码结果 : '{bpe.decode(ids2)}' ")
    except KeyError as e:
        print(f"× 出现 KeyError: {e}")
        print(f"原因:字符 '{e.args[0]}'在训练时没有出现过")
        print(f"        （词级 BPE 的局限 : 无法处理训练外的字符）")
    print()

    # 测试 3： 显示词表
    print("词表样本(共{}个token):".format(len(bpe.vocab)))
    sorted_tokens = sorted(bpe.vocab.items(),key = lambda x:x[1])
    for token,token_id in sorted_tokens[:20]:
        print(f"  id {token_id:2d}: '{token}'")
    if len(sorted_tokens) > 20:
        print(f" ... 还有 {len(sorted_tokens) -20} 个 token ...")
