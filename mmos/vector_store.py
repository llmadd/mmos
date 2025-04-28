"""
简单的向量存储模块，用于语义搜索
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable

from .models import Memory


class SimpleVectorStore:
    """简单的向量存储实现"""
    
    def __init__(self, 
                 embedding_function: Optional[Callable[[str], np.ndarray]] = None,
                 dimension: int = 384):
        """
        初始化向量存储
        
        参数:
            embedding_function: 将文本转换为向量的函数，如果为None则使用随机向量模拟
            dimension: 向量维度，仅在使用随机向量时有效
        """
        self.vectors: Dict[str, np.ndarray] = {}  # memory_id -> vector
        self.dimension = dimension
        
        if embedding_function:
            self.embedding_function = embedding_function
        else:
            # 如果没有提供嵌入函数，使用随机向量模拟
            self._rng = np.random.RandomState(42)  # 固定随机种子以保持一致性
            self.embedding_function = self._mock_embedding
    
    def _mock_embedding(self, text: str) -> np.ndarray:
        """生成模拟的嵌入向量，仅用于演示"""
        # 基于文本内容生成一个伪随机但一致的向量
        # 实际应用中应替换为真实的嵌入模型
        self._rng.seed(hash(text) % 2**32)
        vector = self._rng.random(self.dimension) - 0.5  # -0.5~0.5范围的随机值
        return vector / np.linalg.norm(vector)  # 归一化
    
    def add_memory(self, memory: Memory) -> None:
        """
        将记忆添加到向量存储中
        
        参数:
            memory: 要添加的记忆对象
        """
        vector = self.embedding_function(memory.content)
        self.vectors[memory.id] = vector
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        基于语义相似度搜索记忆
        
        参数:
            query: 查询字符串
            top_k: 返回的最大结果数
            
        返回:
            包含(memory_id, 相似度分数)的列表，按相似度从高到低排序
        """
        if not self.vectors:
            return []
            
        query_vector = self.embedding_function(query)
        
        results = []
        for memory_id, vector in self.vectors.items():
            # 计算余弦相似度
            similarity = np.dot(query_vector, vector)
            results.append((memory_id, float(similarity)))
        
        # 按相似度从高到低排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def remove_memory(self, memory_id: str) -> bool:
        """
        从向量存储中移除记忆
        
        参数:
            memory_id: 要移除的记忆ID
            
        返回:
            是否成功移除
        """
        if memory_id in self.vectors:
            del self.vectors[memory_id]
            return True
        return False
    
    def update_memory(self, memory: Memory) -> None:
        """
        更新记忆的向量表示
        
        参数:
            memory: 包含新内容的记忆对象
        """
        self.add_memory(memory)  # 直接覆盖现有向量
    
    def clear(self) -> None:
        """清空向量存储"""
        self.vectors = {} 