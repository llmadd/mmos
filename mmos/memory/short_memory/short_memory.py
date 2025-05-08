from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from typing import List, Dict, Any, Optional, Union, Iterable, Literal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, pairwise_distances
import json
load_dotenv()

os.environ["OPENAI_API_KEY"] = "sk-proj-1234567890"
os.environ["OPENAI_BASE_URL"] = "http://180.153.21.76:12118/v1"
os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"
os.environ["CHROMA_DB_PATH"] = "mmos/vector_db"




test_cases = """
[

  {
    "messages": [
      {"role": "user", "content": "推荐几个巴黎的景点"},
      {"role": "assistant", "content": "埃菲尔铁塔、卢浮宫、蒙马特高地都值得一去"},
      {"role": "user", "content": "卢浮宫需要预约吗"}
    ],
    "expected_result": true
  },


  {
    "messages": [
      {"role": "user", "content": "Python的异常处理怎么写"},
      {"role": "assistant", "content": "建议使用try-except结构"},
      {"role": "user", "content": "那finally什么时候用"}
    ],
    "expected_result": true
  },


  {
    "messages": [
      {"role": "user", "content": "如何煮意大利面"},
      {"role": "assistant", "content": "水开后煮8分钟加盐"},
      {"role": "user", "content": "特斯拉股票今天涨了吗"}
    ],
    "expected_result": false
  },

  {
    "messages": [
      {"role": "user", "content": "介绍下Transformer架构"},
      {"role": "assistant", "content": "基于自注意力机制的深度学习模型"},
      {"role": "user", "content": "它的训练成本有多高"}
    ],
    "expected_result": true
  },

  {
    "messages": [
      {"role": "user", "content": "明天北京天气怎样"},
      {"role": "assistant", "content": "晴天，15-22℃"},
      {"role": "user", "content": "上海呢"}
    ],
    "expected_result": true
  }
]
"""

test_cases = json.loads(test_cases)

class ShortMemory:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chroma_client = PersistentClient(path=os.getenv("CHROMA_DB_PATH", "mmos/vector_db"))

    def _get_embedding(self, input: str | List[str] | Iterable[int] | Iterable[Iterable[int]],) -> List:
        print(input)
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=input,
        )
        return [embedding.embedding for embedding in response.data]
    
    def _calculate_vector_similarity(self, vector1: List[float], vector2: List[float], method: Literal["cosine", "euclidean", "dot_product", "manhattan", "jaccard"] = "cosine") -> float:
        """计算两个向量之间的相似度。
        
        参数:
            vector1: 第一个向量
            vector2: 第二个向量
            method: 计算方法
                - cosine: 余弦相似度，1：完全同向（相似）0：正交（无关）-1：完全反向（不相似）
                - euclidean: 欧几里得距离，值越小表示相似度越高（0最相似）
                - dot_product: 点积，值越大表示相似度越高
                - manhattan: 曼哈顿距离，值越小表示相似度越高（0最相似）
                - jaccard: 杰卡德相似度，值越大表示相似度越高（1最相似，0最不相似）
        
        返回:
            相似度得分
        """
        # 将输入向量转换为2D数组（sklearn要求的格式）
        v1 = np.array(vector1).reshape(1, -1)
        v2 = np.array(vector2).reshape(1, -1)
        
        if method == "cosine":
            # 余弦相似度：测量两个向量夹角的余弦值
            return cosine_similarity(v1, v2)
        elif method == "euclidean":
            # 欧几里得距离：测量两点之间的直线距离
            return euclidean_distances(v1, v2)[0][0]
        elif method == "dot_product":
            # 点积：直接计算向量点积，适用于归一化向量
            return np.dot(v1, v2.T)[0][0]
        elif method == "manhattan":
            # 曼哈顿距离：测量沿坐标轴的距离总和
            return manhattan_distances(v1, v2)[0][0]
        elif method == "jaccard":
            # 杰卡德相似度：集合相似度度量
            # 对于二元向量或稀疏特征，使用pairwise_distances计算Jaccard距离
            # 首先将向量转换为二进制形式（>0的元素设为1）
            binary_v1 = (v1 > 0).astype(int)
            binary_v2 = (v2 > 0).astype(int)
            
            # 如果两个向量都是零向量，返回1.0（完全相似）
            if not np.any(binary_v1) and not np.any(binary_v2):
                return 1.0
                
            # 计算Jaccard距离（1减去相似度）
            jaccard_dist = pairwise_distances(binary_v1, binary_v2, metric='jaccard')[0][0]
            # 返回Jaccard相似度（1减去距离）
            return 1.0 - jaccard_dist
        
    def split_message(self, messages: List[Dict[str, str]],instant_count: int = 1, similarity_threshold: float = 0.5) -> List[Dict[str, str]]:
        """
        将消息列表按即时消息数量分割成多个子列表
        
        参数:
            messages: 消息列表
            instant_count: 每个子列表中即时消息的数量
            similarity_threshold: 相似度阈值
        返回:
            分割后的消息列表
        """
        user_messages = []
        for item in messages:
            if item.get("role") == "user" and item.get("content") is not None:
                user_messages.append(item["content"])
            
        for i in range(0, len(messages), instant_count):
            result.append(messages[i:i+instant_count])
        return result

if __name__ == "__main__":
    short_memory = ShortMemory()
    # 指代消解（Coreference Resolution）实现
    for i in test_cases:
        embedding = short_memory._get_embedding([i["messages"][0]["content"],i["messages"][0]["content"]+" "+i["messages"][2]["content"]])
        result = short_memory._calculate_vector_similarity(embedding[0], embedding[1])
        print(result)
