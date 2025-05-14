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

# 根据user进行消息切割为块也就是每轮对话
# 最近2-3轮对话保持不变
# 3-20轮对话进行筛选排序，保留3-10轮
# 10轮之外对话历史进行判定相关性特低对话进行切割，切割后进行事件抽取，记忆压缩等成为长期记忆。


test_cases = [
    {"role": "system", "content": "你是一个经验丰富的AI助手，擅长回答用户的问题。"},
    {"role": "user", "content": "Python里怎么读取JSON文件"},
    {"role": "assistant", "content": "可以用json模块的load()方法，记得先用open()打开文件"},
    
    {"role": "user", "content": "东京有什么必去的购物中心"},
    {"role": "assistant", "content": "推荐银座六丁目、涩谷109和新宿伊势丹"},
    
    {"role": "user", "content": "json.load()会抛出哪些异常"},
    {"role": "assistant", "content": "常见的有JSONDecodeError和FileNotFoundError"},
    
    {"role": "user", "content": "日本签证需要准备什么材料"},
    {"role": "assistant", "content": "通常需要护照、照片、在职证明和银行流水"},
    
    {"role": "user", "content": "怎么处理JSONDecodeError"},
    {"role": "assistant", "content": "建议用try-except捕获异常，并检查文件内容格式"},
    
    {"role": "user", "content": "浅草寺的开放时间是"},
    {"role": "assistant", "content": "一般是6:00-17:00，季节不同可能有调整"},
    
    {"role": "user", "content": "Python解析XML用什么库比较好"},
    {"role": "assistant", "content": "标准库xml.etree.ElementTree就够用，lxml性能更好"},
    
    {"role": "user", "content": "京都的和服体验店推荐"},
    {"role": "assistant", "content": "冈本织物和梦馆口碑都不错，记得提前预约"},
    
    {"role": "user", "content": "ElementTree怎么处理命名空间"},
    {"role": "assistant", "content": "可以用{URI}localname的格式或register_namespace()方法"},
    
    {"role": "user", "content": "大阪环球影城要买快速通票吗"},
    {"role": "assistant", "content": "旺季建议购买，能节省3-4小时排队时间"},
    {"role": "user", "content": "东京塔的开放时间"},
  ]


# test_cases = json.loads(test_cases)

class ShortMemory:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chroma_client = PersistentClient(path=os.getenv("CHROMA_DB_PATH", "mmos/vector_db"))

    def _get_embedding(self, input: str | List[str] | Iterable[int] | Iterable[Iterable[int]],) -> List:
        print(f"embedding输入: {input}")
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
            return cosine_similarity(v1, v2)[0][0]  
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
        
    def split_messages_by_user_role(self,messages):
        result = []
        current_chunk = []
        
        for message in messages:
            if message["role"] == "user":
                # 如果当前块不为空，将其添加到结果中
                if current_chunk:
                    result.append(current_chunk)
                # 创建新块，以当前user消息开始
                current_chunk = [message]
            else:
                # 非user消息，添加到当前块
                current_chunk.append(message)
        
        # 添加最后一个块（如果有）
        if current_chunk:
            result.append(current_chunk)
        
        return result

    def related_judgement(self,content0:str,content1:str) -> float:
        weights = np.array([0.3, 0.7])
        embedding = self._get_embedding([content0,content1,content0+" "+content1])
        source0 = self._calculate_vector_similarity(embedding[0],embedding[1])

        if source0 > 0.5:
            return source0
        elif source0 > 0.2:
            source1 = self._calculate_vector_similarity(embedding[0],embedding[2])
            if source1 > 0.5:               
                sources = np.array([source0, source1])
                return np.average(sources, weights=weights, axis=0)
            else:
                sources = np.array([source0, source1])
                return np.average(sources, weights=weights, axis=0)
        return source0
    def split_message(self, messages: List[Dict[str, str]], instant_count: int = 1) -> List[Dict[str, str]]:
        """
        将消息列表按相关性排序，保持最近的instant_count轮对话不变，对历史对话进行排序
        
        参数:
            messages: 消息列表，格式类似OpenAI messages格式
            instant_count: 保持不变的最近对话轮数
            
        返回:
            重新排序后的消息列表，相关性高的对话排在靠近最近消息的位置
        """
        # 如果消息少于instant_count轮，直接返回
        if len(messages) <= instant_count * 2 + 2:
            return messages
            
        # 分离最近的instant_count轮对话
        messages_list = self.split_messages_by_user_role(messages)
        
        # 获取最后一个用户消息的内容
        last_user_message = ""

        for item in messages_list[-1]:
            if item["role"] == "user" and item["content"] is not None:
                last_user_message = item["content"]
                break
        if not last_user_message:
            return messages  # 如果找不到最后的用户消息，返回原始消息列表
        
  
        
        # 计算每轮对话与最后用户消息的相关性
        round_similarities = []
        reordered_messages = []
        for round_msgs in messages_list[:-instant_count-1]:
            user_content = ""
            for msg in round_msgs:
                if msg.get("role") == "user" and msg.get("content") is not None:
                    user_content = msg.get("content")
                    break
            
            if user_content:
                similarity = self.related_judgement(user_content, last_user_message)
                round_similarities.append((round_msgs, similarity))
            else:
                reordered_messages.append(round_msgs)
        
        print(f"轮次相似度: {round_similarities}")
        
        # 根据相关性从低到高排序对话轮次
        sorted_rounds = sorted(round_similarities, key=lambda x: x[1])
        
        # 重建消息列表
        for round_msgs, _ in sorted_rounds:
            reordered_messages.extend(round_msgs)
            print(_)
        
        # 添加最近的instant_count轮不变的对话
        reordered_messages.extend(messages_list[-instant_count-1:])
        
        return reordered_messages




if __name__ == "__main__":
    short_memory = ShortMemory()
    # print(short_memory.split_messages_by_user_role(test_cases))
    import time
    start_time = time.time()
    # 指代消解（Coreference Resolution）实现
  
    print(f"重排序后的消息: {short_memory.split_message(test_cases)}")
    end_time = time.time()
    print(f"运行时间: {end_time - start_time} 秒")


