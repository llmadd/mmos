# from openai import OpenAI, AsyncOpenAI
# import os
# from typing import List, Dict, Any, Union, Optional
# import json
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import jieba
# """
# 短期记忆模块

# 可能需要：
# 1. 语言检测


# 接收messages，返回messages格式。
# 1. 小于2轮对话直接返回
# 2. 大于2轮对话，最后两轮不改变，其余内容进行压缩
# 3. 判定对话相关性（基于内容、基于时间等）不相关对话上下文剥离
# 4. AI与算法两种模式，以及混合模型


# 纯算法实现逻辑：

# # 对话消息处理技术方案（基于OpenAI messages格式）

# ## 一、相关性判定技术栈

# ### 1. 基于TF-IDF的语义指纹技术
# **技术组成**：
# - Scikit-learn的TfidfVectorizer
# - 余弦相似度计算
# - 关键词扩展算法

# **处理流程**：
# 1. 对所有user消息构建TF-IDF矩阵
# 2. 为每条消息生成语义指纹（Top-N高权重词）
# 3. 计算新消息与历史消息指纹的相似度

# **作用解释**：
# - 将每条消息转换为数值向量
# - 通过向量距离衡量语义相似性
# - 解决表面词不匹配但语义相关的问题（如"价格"-"多少钱"）

# **消息处理方式**：
# - 只比较user消息之间的相关性
# - assistant消息的相关性由其对应的user消息决定
# - 示例：
#   ```python
#   # messages示例
#   [
#       {"role": "user", "content": "推荐餐厅"}, 
#       {"role": "assistant", "content": "附近有三家..."},
#       {"role": "user", "content": "有什么好吃的"}
#   ]
#   # 比较"推荐餐厅"和"有什么好吃的"的相似度
#   ```

# ### 2. 基于TextRank的对话主题图
# **技术组成**：
# - NetworkX图算法库
# - 关键词共现分析
# - 窗口滑动算法

# **处理流程**：
# 1. 将每3轮对话作为一个分析窗口
# 2. 构建关键词共现图
# 3. 使用PageRank算法识别核心主题词

# **作用解释**：
# - 识别对话中的核心话题节点
# - 检测话题漂移和切换点
# - 建立跨轮次的主题关联

# **消息处理方式**：
# - 对user和assistant消息统一分析
# - 标记包含相同核心主题词的消息块
# - 示例主题图：
#   ```
#   [餐厅]--(推荐)-->[美食]--(川菜)-->[辣度]
#   ```

# ### 3. 基于布隆过滤器的快速筛选
# **技术组成**：
# - PyBloom实现
# - 关键词哈希映射
# - 多重哈希函数

# **处理流程**：
# 1. 提取所有历史消息的实体名词
# 2. 构建布隆过滤器
# 3. 新消息关键词快速匹配

# **作用解释**：
# - 实现O(1)时间复杂度的初筛
# - 过滤明显无关的消息
# - 减少后续复杂计算量

# **消息处理方式**：
# - 对user消息进行完整过滤
# - 对assistant消息只检查其包含的实体

# ## 二、内容压缩技术栈

# ### 1. 基于TF-IDF的层次化压缩
# **技术组成**：
# - 关键词权重排序
# - 句子重要性评分
# - 实体保留机制

# **处理流程**：
# 1. 计算消息中每个词的TF-IDF权重
# 2. 按权重降序选取关键成分
# 3. 重组保留核心语义的压缩句

# **作用解释**：
# - 保留信息密度最高的内容
# - 去除冗余修饰语
# - 维持基本语法结构

# **消息处理方式**：
# - **user消息压缩**：
#   - 保留疑问词和否定词
#   - 示例：
#     ```
#     原始："我想找一个价格不太贵而且环境比较好的西餐厅"
#     压缩："找不贵环境好的西餐厅"
#     ```

# - **assistant消息压缩**：
#   - 保留数据性内容和结论
#   - 示例：
#     ```
#     原始："推荐ABC西餐厅，人均200元，评分4.5，有露天座位"
#     压缩："ABC西餐厅：200元/4.5分/有露台"
#     ```

# ### 2. 基于TextRank的摘要生成
# **技术组成**：
# - 句子相似度矩阵
# - 图节点权重计算
# - 重要性阈值过滤

# **处理流程**：
# 1. 将多轮对话构建为句子图
# 2. 计算每个句子的TextRank值
# 3. 提取权重最高的1-2个句子

# **作用解释**：
# - 识别对话中最具代表性的语句
# - 生成连贯的摘要
# - 保持上下文完整性

# **消息处理方式**：
# - 对连续相关的3-5轮对话整体压缩
# - 示例：
#   ```python
#   # 原始messages
#   [
#       {"role": "user", "content": "推荐景点"},
#       {"role": "assistant", "content": "故宫是必去的..."},
#       {"role": "user", "content": "门票多少钱"},
#       {"role": "assistant", "content": "旺季60元..."}
#   ]
#   # 压缩后
#   "用户询问景点推荐和门票，推荐故宫(旺季60元)"
#   ```

# ### 3. 基于问答对的模板化压缩
# **技术组成**：
# - 预定义问答模板
# - 槽位填充机制
# - 实体关系抽取

# **处理流程**：
# 1. 识别问答对模式
# 2. 匹配预定义模板
# 3. 提取关键信息填充槽位

# **作用解释**：
# - 结构化存储对话核心
# - 极大提升压缩率
# - 便于后续检索

# **消息处理方式**：
# - 将相邻的user-assistant消息对作为处理单元
# - 示例模板：
#   ```
#   [景点推荐] 
#   问：{景点类型} 
#   答：{推荐名称}{价格}{特色}
#   ```

# ## 三、消息处理策略（无时间维度）

# ### 1. 角色区分处理原则
# - **User消息优先**：作为相关性判断的基准和压缩的重点
# - **Assistant消息从属**：其相关性由引发它的user消息决定
# - **系统消息特殊处理**：标记但不参与常规分析

# ### 2. 处理流程设计
# ```
# 原始messages输入
#     ↓
# [角色分离器] → 分离user/assistant消息流
#     ↓
# [相关性判定层] ← 只分析user消息
#     ↓
# [消息分组器] → 按相关性分组对话块
#     ↓
# [内容压缩器] → 对每组分别压缩
#     ↓ 
# [重组输出] → 保持最后两轮完整
# ```

# ### 3. 关键设计决策
# 1. **比较基准选择**：
#    - 始终以最新user消息为比较基准
#    - 历史user消息作为被比较对象

# 2. **压缩粒度控制**：
#    - 单条user消息：关键词提取
#    - 问答对：模板化压缩
#    - 多轮对话：TextRank摘要

# 3. **信息保留策略**：
#    - 必保留项：数字、实体、结论
#    - 可丢弃项：修饰语、重复解释
#    - 特殊保留：否定词和疑问词

# 这种方案完全基于消息内容和角色进行处理，不依赖时间维度，符合OpenAI messages的基本处理逻辑，在保持较高信息保真度的同时可以实现60-70%的压缩率。

# """





# # 测试消息
# with open("/home/qichen/zh/mmos/test/mes.json", "r", encoding="utf-8") as f:
#     mes = json.load(f)


# ## 方案一：基于TF-IDF和布谷过滤器加向量近余弦匹配实现相关性判定
# with open("/home/qichen/zh/mmos/test/stopwords.txt", "r", encoding="utf-8") as f:
#     stopwords = f.read().splitlines()


# def tokenize(text: str) -> List[str]:
#     """中文分词示例（需安装 jieba）"""
#     sentences = []
#     words = []
#     start = 0
#     for i, char in enumerate(text):
#         if char in {'。', '？', '！', '；'}:
#             sentence = text[start:i+1].strip()
#             if sentence:
#                 sentences.append(sentence)
#             start = i+1
#     last_sentence = text[start:].strip()
#     if last_sentence:
#         sentences.append(last_sentence)
#     for sentence in sentences:
#         words.extend(jieba.lcut(sentence))
#     return words


# def load_stopwords() -> List[str]:
#     """加载停用词表"""
#     return stopwords 

# vectorizer = TfidfVectorizer(
#             tokenizer=tokenize,  # 自定义分词函数
#             stop_words=load_stopwords(),  # 停用词表
#             max_features=100,  # 限制特征数量
#             encoding="utf-8"
#         )

# def clean_text(text: str) -> str:
#     """清理文本，仅保留中文字符"""
#     # 匹配中文字符（Unicode范围：\u4e00-\u9fff）
#     chinese_pattern = re.compile(r'[^\u4e00-\u9fff]')
#     # 移除非中文字符
#     cleaned_text = re.sub(chinese_pattern, '', text)
#     return cleaned_text

# texts = [clean_text(msg["content"]) for msg in mes[:2] if "content" in msg]
# X = vectorizer.fit_transform(texts)

# print(vectorizer.get_feature_names_out())


mes_02 = [
{"role": "system", "content": "你是一个健身教练，擅长制定训练计划和饮食建议。"},

{"role": "user", "content": "我想制定一个增肌训练计划，每周去三次健身房，有什么建议？"},
{"role": "assistant", "content": "好的增肌计划应该包含：\n1. 复合动作（深蹲/硬拉/卧推）\n2. 每组8-12次，4组\n3. 每次训练侧重不同肌群\n需要我具体说明每日训练内容吗？"},

{"role": "user", "content": "先说说周一练什么比较好？"},
{"role": "assistant", "content": "推荐周一练胸肌和三头肌：\n🔸 杠铃卧推 4x8\n🔸 哑铃飞鸟 3x10\n🔸 双杠臂屈伸 3x力竭\n要注意热身5-10分钟哦"},

{"role": "user", "content": "对了突然想起来！我周末想做蛋糕，零基础该从哪里开始？"},  # 话题跳转
{"role": "assistant", "content": "新手烘焙建议从简单的开始：\n🎂 戚风蛋糕（需电动打蛋器）\n🍰 杯子蛋糕（失败率低）\n需要食谱吗？还是更关心工具准备？"},  # 自然衔接新话题

{"role": "user", "content": "杯子蛋糕需要哪些材料？"},
{"role": "assistant", "content": "基础材料清单：\n1. 低筋面粉 100g\n2. 鸡蛋 2个\n3. 细砂糖 80g\n4. 无盐黄油 50g\n5. 泡打粉 1小勺\n（可加香草精/可可粉调味）"},

{"role": "user", "content": "烤箱温度和时间怎么控制？"},
{"role": "assistant", "content": "关键参数：\n🔥 预热180℃\n⏰ 烤15-20分钟\n⚠️ 判断熟度：牙签插入无面糊粘连\n要特别注意每个烤箱实际温度可能有差异"},

{"role": "user", "content": "明白了，回到健身话题，周三该练什么？"},  # 主动转回原话题
{"role": "assistant", "content": "周三建议练背部和二头肌：\n🔸 引体向上 4组（辅助带可用）\n🔸 杠铃划船 3x10\n🔸 哑铃弯举 3x12\n需要调整难度吗？"},  # 无缝衔接

{"role": "user", "content": "引体向上做不起来怎么办？"},
{"role": "assistant", "content": "替代方案：\n1. 使用弹力带辅助\n2. 做离心训练（跳起后缓慢下降）\n3. 先练高位下拉器械\n建议每周增加1-2次专项训练"},

{"role": "user", "content": "周五的训练和饮食能一起说说吗？"},
{"role": "assistant", "content": "【周五计划】\n🏋️ 练腿日：深蹲/腿举/箭步蹲\n🍗 饮食建议：\n- 训练后补充蛋白质（鸡胸/蛋白粉）\n- 碳水选择糙米/红薯\n需要计算具体热量摄入吗？"},

{"role": "user", "content": "你认为最好吃的蛋糕是什么？"},
]

import jieba
import re
from datasketch import MinHash




def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    文本预处理：清洗文本并分词
    
    Args:
        text: 原始文本
        remove_stopwords: 是否去除停用词
        
    Returns:
        处理后的文本（词语以空格分隔）
    """
    # 默认停用词列表
    stopwords = ["的", "了", "在", "是", "我", "有", "和", "就", "不", 
                "人", "都", "一", "一个", "上", "也", "很", "到", "说", 
                "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "什么"]
    
    # 1. 清洗文本
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text)
    # 移除换行符等
    text = re.sub(r'[\n\r\t]', ' ', text)
    # 移除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 英文部分转小写
    text = text.lower()
    
    # 2. 分词
    words = list(jieba.cut(text))
    
    # 3. 过滤词语
    if remove_stopwords:
        words = [word for word in words if word not in stopwords and len(word.strip()) > 1]
    else:
        words = [word for word in words if len(word.strip()) > 1]
    
    return " ".join(words)


def create_minhash(text: str, num_perm: int = 256) -> MinHash:
    """
    为文本创建MinHash签名
    
    Args:
        text: 预处理后的文本
        num_perm: 排列数，越高越准确但计算越慢
        
    Returns:
        MinHash对象
    """
    minhash = MinHash(num_perm=num_perm)
    text = preprocess_text(text)
    for word in text.split():
        minhash.update(word.encode('utf8'))
        
    return minhash


def calculate_minhash_similarity(minhash1: MinHash, minhash2: MinHash) -> float:
    """
    计算两个MinHash的Jaccard相似度
    
    Args:
        minhash1: 第一个MinHash
        minhash2: 第二个MinHash
        
    Returns:
        Jaccard相似度(0-1之间)
    """
    return minhash1.jaccard(minhash2)


def split_messages_by_user_role(messages):
    metadata = []
    messages = []
    current_chunk = []
    
    for message in messages:
        if message["role"] == "user":
            # 如果当前块不为空，添加块信息并将其加入结果
            if current_chunk:
                chunk_info = analyze_chunk(current_chunk)
                metadata.append(chunk_info)
                messages.append(current_chunk)
            # 创建新块，以当前user消息开始
            current_chunk = [message]
        else:
            # 非user消息，添加到当前块
            current_chunk.append(message)
    
    # 添加最后一个块（如果有）
    if current_chunk:
        chunk_info = analyze_chunk(current_chunk)
        metadata.append(chunk_info)
        messages.append(current_chunk)
    
    return {
        "messages": messages,
        "metadata": metadata
    }
def analyze_chunk(chunk):
    """分析消息块并生成元信息"""
    roles = set(msg["role"] for msg in chunk)
    role_counts = {}
    for role in roles:
        role_counts[role] = sum(1 for msg in chunk if msg["role"] == role)
    
    # 为user role的消息计算哈希值
    user_content_hashes = []
    for msg in chunk:
        if msg["role"] == "user":
            minhash = create_minhash(msg["content"])
            user_content_hashes.append(minhash)
    
    return {
        "roles": list(roles),                    # 包含的角色类型列表
        "message_count": len(chunk),             # 消息总数
        "has_user": "user" in roles,            # 是否包含用户消息
        "role_counts": role_counts,             # 各角色消息数量统计
        "user_content_hashes": user_content_hashes  # user消息内容的哈希值列表
    }


minhash = create_minhash(split_messages_by_user_role(mes_02)[-1]["messages"][0]["content"])



def run():
    mes_chunk = split_messages_by_user_role(mes_02)
    
    for i in split_messages_by_user_role(mes_02)[:-1]:
        print(i)
        print("-"*100)
        if i["metadata"]["has_user"]:
            minhash1 = create_minhash(i["messages"][0]["content"])
            print(calculate_minhash_similarity(minhash1,  minhash))
            print("-"*100)
            
