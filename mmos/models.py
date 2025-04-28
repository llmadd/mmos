"""
记忆数据模型
"""

import time
from typing import List, Dict, Any, Optional
import uuid


class Memory:
    """记忆对象类"""
    
    def __init__(
        self, 
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ):
        """
        初始化一个新的记忆对象
        
        参数:
            content: 记忆内容
            tags: 记忆标签列表
            metadata: 附加元数据
            importance: 记忆重要性 (0.0-1.0)
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.tags = tags or []
        self.metadata = metadata or {}
        self.importance = max(0.0, min(1.0, importance))  # 限制在0-1范围内
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        
    def access(self) -> None:
        """更新记忆访问信息"""
        self.last_accessed = time.time()
        self.access_count += 1
        
    def update_importance(self, new_importance: float) -> None:
        """更新记忆重要性"""
        self.importance = max(0.0, min(1.0, new_importance))
        
    def to_dict(self) -> Dict[str, Any]:
        """将记忆转换为字典表示"""
        return {
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "metadata": self.metadata,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """从字典创建记忆对象"""
        memory = cls(
            content=data["content"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5)
        )
        memory.id = data.get("id", memory.id)
        memory.created_at = data.get("created_at", memory.created_at)
        memory.last_accessed = data.get("last_accessed", memory.last_accessed)
        memory.access_count = data.get("access_count", memory.access_count)
        return memory
    
    def __repr__(self) -> str:
        return f"Memory(id={self.id}, content={self.content[:30]}{'...' if len(self.content) > 30 else ''})" 