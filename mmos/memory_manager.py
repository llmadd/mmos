import json
import os
from typing import List, Dict, Any, Optional, Union, Callable
import time

from .models import Memory
from pydantic import BaseModel


class MemoryManager:
    """记忆管理器类，用于管理AI系统的记忆"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化记忆管理器
        
        参数:
            storage_path: 记忆存储路径，如果为None则仅在内存中存储
        """
        self.memories: Dict[str, Memory] = {}
        self.storage_path = storage_path
        if storage_path and os.path.exists(storage_path):
            self.load_from_storage()
    
    def store(self, content: str, tags: Optional[List[str]] = None, 
              metadata: Optional[Dict[str, Any]] = None, 
              importance: float = 0.5) -> Memory:
        """
        存储新记忆
        
        参数:
            content: 记忆内容
            tags: 记忆标签
            metadata: 附加元数据
            importance: 记忆重要性 (0-1)
            
        返回:
            存储的记忆对象
        """
        memory = Memory(content=content, tags=tags, metadata=metadata, importance=importance)
        self.memories[memory.id] = memory
        
        if self.storage_path:
            self.save_to_storage()
            
        return memory
    
    def retrieve(self, query: str, limit: int = 10, 
                 filter_func: Optional[Callable[[Memory], bool]] = None) -> List[Memory]:
        """
        检索记忆
        
        参数:
            query: 查询字符串
            limit: 返回结果数量限制
            filter_func: 过滤函数
            
        返回:
            匹配的记忆列表
        """
        # 这里实现的是一个简单的关键词匹配
        # 真实应用中可能需要使用向量数据库或更复杂的语义搜索
        results = []
        
        for memory in self.memories.values():
            if filter_func and not filter_func(memory):
                continue
                
            if query.lower() in memory.content.lower():
                memory.access()
                results.append(memory)
            
            if len(results) >= limit:
                break
                
        return results
    
    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """根据ID获取记忆"""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access()
        return memory
    
    def get_by_tags(self, tags: List[str], match_all: bool = False) -> List[Memory]:
        """根据标签获取记忆"""
        results = []
        
        for memory in self.memories.values():
            if match_all:
                # 所有标签都必须匹配
                if all(tag in memory.tags for tag in tags):
                    memory.access()
                    results.append(memory)
            else:
                # 匹配任意标签
                if any(tag in memory.tags for tag in tags):
                    memory.access()
                    results.append(memory)
                    
        return results
    
    def update(self, memory_id: str, content: Optional[str] = None, 
               tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
               importance: Optional[float] = None) -> Optional[Memory]:
        """更新记忆"""
        memory = self.memories.get(memory_id)
        if not memory:
            return None
            
        if content is not None:
            memory.content = content
            
        if tags is not None:
            memory.tags = tags
            
        if metadata is not None:
            memory.metadata = metadata
            
        if importance is not None:
            memory.update_importance(importance)
            
        memory.access()
        
        if self.storage_path:
            self.save_to_storage()
            
        return memory
    
    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        if memory_id in self.memories:
            del self.memories[memory_id]
            
            if self.storage_path:
                self.save_to_storage()
                
            return True
        return False
    
    def save_to_storage(self) -> None:
        """保存记忆到存储"""
        if not self.storage_path:
            return
            
        data = {
            "memories": {mid: memory.to_dict() for mid, memory in self.memories.items()},
            "last_saved": time.time()
        }
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_storage(self) -> None:
        """从存储加载记忆"""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
            
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if "memories" in data:
                self.memories = {
                    mid: Memory.from_dict(mdata) 
                    for mid, mdata in data["memories"].items()
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"加载记忆时出错: {e}")
    
    def clear(self) -> None:
        """清空所有记忆"""
        self.memories = {}
        
        if self.storage_path and os.path.exists(self.storage_path):
            self.save_to_storage()
    
    def get_all(self) -> List[Memory]:
        """获取所有记忆"""
        return list(self.memories.values())
    
    def count(self) -> int:
        """获取记忆数量"""
        return len(self.memories) 