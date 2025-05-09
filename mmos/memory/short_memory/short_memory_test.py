from openai import OpenAI, AsyncOpenAI
import os
from typing import List, Dict, Any, Union, Optional
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
"""
短期记忆模块

可能需要：
1. 语言检测


接收messages，返回messages格式。
1. 小于2轮对话直接返回
2. 大于2轮对话，最后两轮不改变，其余内容进行压缩
3. 判定对话相关性（基于内容、基于时间等）不相关对话上下文剥离
4. AI与算法两种模式，以及混合模型


纯算法实现逻辑：

# 对话消息处理技术方案（基于OpenAI messages格式）

## 一、相关性判定技术栈

### 1. 基于TF-IDF的语义指纹技术
**技术组成**：
- Scikit-learn的TfidfVectorizer
- 余弦相似度计算
- 关键词扩展算法

**处理流程**：
1. 对所有user消息构建TF-IDF矩阵
2. 为每条消息生成语义指纹（Top-N高权重词）
3. 计算新消息与历史消息指纹的相似度

**作用解释**：
- 将每条消息转换为数值向量
- 通过向量距离衡量语义相似性
- 解决表面词不匹配但语义相关的问题（如"价格"-"多少钱"）

**消息处理方式**：
- 只比较user消息之间的相关性
- assistant消息的相关性由其对应的user消息决定
- 示例：
  ```python
  # messages示例
  [
      {"role": "user", "content": "推荐餐厅"}, 
      {"role": "assistant", "content": "附近有三家..."},
      {"role": "user", "content": "有什么好吃的"}
  ]
  # 比较"推荐餐厅"和"有什么好吃的"的相似度
  ```

### 2. 基于TextRank的对话主题图
**技术组成**：
- NetworkX图算法库
- 关键词共现分析
- 窗口滑动算法

**处理流程**：
1. 将每3轮对话作为一个分析窗口
2. 构建关键词共现图
3. 使用PageRank算法识别核心主题词

**作用解释**：
- 识别对话中的核心话题节点
- 检测话题漂移和切换点
- 建立跨轮次的主题关联

**消息处理方式**：
- 对user和assistant消息统一分析
- 标记包含相同核心主题词的消息块
- 示例主题图：
  ```
  [餐厅]--(推荐)-->[美食]--(川菜)-->[辣度]
  ```

### 3. 基于布隆过滤器的快速筛选
**技术组成**：
- PyBloom实现
- 关键词哈希映射
- 多重哈希函数

**处理流程**：
1. 提取所有历史消息的实体名词
2. 构建布隆过滤器
3. 新消息关键词快速匹配

**作用解释**：
- 实现O(1)时间复杂度的初筛
- 过滤明显无关的消息
- 减少后续复杂计算量

**消息处理方式**：
- 对user消息进行完整过滤
- 对assistant消息只检查其包含的实体

## 二、内容压缩技术栈

### 1. 基于TF-IDF的层次化压缩
**技术组成**：
- 关键词权重排序
- 句子重要性评分
- 实体保留机制

**处理流程**：
1. 计算消息中每个词的TF-IDF权重
2. 按权重降序选取关键成分
3. 重组保留核心语义的压缩句

**作用解释**：
- 保留信息密度最高的内容
- 去除冗余修饰语
- 维持基本语法结构

**消息处理方式**：
- **user消息压缩**：
  - 保留疑问词和否定词
  - 示例：
    ```
    原始："我想找一个价格不太贵而且环境比较好的西餐厅"
    压缩："找不贵环境好的西餐厅"
    ```

- **assistant消息压缩**：
  - 保留数据性内容和结论
  - 示例：
    ```
    原始："推荐ABC西餐厅，人均200元，评分4.5，有露天座位"
    压缩："ABC西餐厅：200元/4.5分/有露台"
    ```

### 2. 基于TextRank的摘要生成
**技术组成**：
- 句子相似度矩阵
- 图节点权重计算
- 重要性阈值过滤

**处理流程**：
1. 将多轮对话构建为句子图
2. 计算每个句子的TextRank值
3. 提取权重最高的1-2个句子

**作用解释**：
- 识别对话中最具代表性的语句
- 生成连贯的摘要
- 保持上下文完整性

**消息处理方式**：
- 对连续相关的3-5轮对话整体压缩
- 示例：
  ```python
  # 原始messages
  [
      {"role": "user", "content": "推荐景点"},
      {"role": "assistant", "content": "故宫是必去的..."},
      {"role": "user", "content": "门票多少钱"},
      {"role": "assistant", "content": "旺季60元..."}
  ]
  # 压缩后
  "用户询问景点推荐和门票，推荐故宫(旺季60元)"
  ```

### 3. 基于问答对的模板化压缩
**技术组成**：
- 预定义问答模板
- 槽位填充机制
- 实体关系抽取

**处理流程**：
1. 识别问答对模式
2. 匹配预定义模板
3. 提取关键信息填充槽位

**作用解释**：
- 结构化存储对话核心
- 极大提升压缩率
- 便于后续检索

**消息处理方式**：
- 将相邻的user-assistant消息对作为处理单元
- 示例模板：
  ```
  [景点推荐] 
  问：{景点类型} 
  答：{推荐名称}{价格}{特色}
  ```

## 三、消息处理策略（无时间维度）

### 1. 角色区分处理原则
- **User消息优先**：作为相关性判断的基准和压缩的重点
- **Assistant消息从属**：其相关性由引发它的user消息决定
- **系统消息特殊处理**：标记但不参与常规分析

### 2. 处理流程设计
```
原始messages输入
    ↓
[角色分离器] → 分离user/assistant消息流
    ↓
[相关性判定层] ← 只分析user消息
    ↓
[消息分组器] → 按相关性分组对话块
    ↓
[内容压缩器] → 对每组分别压缩
    ↓ 
[重组输出] → 保持最后两轮完整
```

### 3. 关键设计决策
1. **比较基准选择**：
   - 始终以最新user消息为比较基准
   - 历史user消息作为被比较对象

2. **压缩粒度控制**：
   - 单条user消息：关键词提取
   - 问答对：模板化压缩
   - 多轮对话：TextRank摘要

3. **信息保留策略**：
   - 必保留项：数字、实体、结论
   - 可丢弃项：修饰语、重复解释
   - 特殊保留：否定词和疑问词

这种方案完全基于消息内容和角色进行处理，不依赖时间维度，符合OpenAI messages的基本处理逻辑，在保持较高信息保真度的同时可以实现60-70%的压缩率。

"""





# 测试消息
with open("/home/qichen/zh/mmos/test/mes.json", "r", encoding="utf-8") as f:
    mes = json.load(f)


## 方案一：基于TF-IDF和布谷过滤器加向量近余弦匹配实现相关性判定
with open("/home/qichen/zh/mmos/test/stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = f.read().splitlines()


def tokenize(text: str) -> List[str]:
    """中文分词示例（需安装 jieba）"""
    sentences = []
    words = []
    start = 0
    for i, char in enumerate(text):
        if char in {'。', '？', '！', '；'}:
            sentence = text[start:i+1].strip()
            if sentence:
                sentences.append(sentence)
            start = i+1
    last_sentence = text[start:].strip()
    if last_sentence:
        sentences.append(last_sentence)
    for sentence in sentences:
        words.extend(jieba.lcut(sentence))
    return words


def load_stopwords() -> List[str]:
    """加载停用词表"""
    return stopwords 

vectorizer = TfidfVectorizer(
            tokenizer=tokenize,  # 自定义分词函数
            stop_words=load_stopwords(),  # 停用词表
            max_features=100,  # 限制特征数量
            encoding="utf-8"
        )

def clean_text(text: str) -> str:
    """清理文本，仅保留中文字符"""
    # 匹配中文字符（Unicode范围：\u4e00-\u9fff）
    chinese_pattern = re.compile(r'[^\u4e00-\u9fff]')
    # 移除非中文字符
    cleaned_text = re.sub(chinese_pattern, '', text)
    return cleaned_text

texts = [clean_text(msg["content"]) for msg in mes[:2] if "content" in msg]
X = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())