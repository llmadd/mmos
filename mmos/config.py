from typing import Dict, Literal, Union, Optional, List, Any
from pydantic import BaseModel, Field, field_validator

# 支持的所有模块和策略类型
ModuleName = Literal["short_memory", "long_memory", "persona", "event"]
StrategyType = Literal["auto", "ai", "algorithm"]
VectorDBType = Literal["chromadb", "pgvector"]
GraphDBType = Literal["neo4j", "arangodb", "none"]

class StorageConfig(BaseModel):
    """
    存储引擎配置
    
    控制不同类型的存储引擎选项和行为
    """
    enabled: bool = Field(
        default=False,
        description="是否启用存储引擎"
    )
    vector_db: VectorDBType = Field(
        default="chromadb",
        description="向量数据库选择 (default: chromadb)"
    )
    graph_db: GraphDBType = Field(
        default="neo4j",
        description="图数据库选择 (default: neo4j)"
    )
    custom_storage: Optional[Dict] = Field(
        default=None,
        description="自定义存储引擎配置"
    )
    data_path: Optional[str] = Field(
        default="./mmdata",
        description="存储数据的路径"
    )
    cache_size: int = Field(
        default=1024,
        description="内存缓存大小(MB)"
    )
    auto_save: bool = Field(
        default=True,
        description="是否自动保存更改"
    )

    @field_validator("custom_storage")
    def check_custom_storage(cls, v, values):
        if values.get("vector_db") == "custom" and not v:
            raise ValueError("custom_storage必须配置当使用自定义向量库时")
        return v

class ModuleConfig(BaseModel):
    """
    单个模块的配置
    
    控制模块的开关状态、实现策略和专用参数
    """
    enabled: bool = Field(default=True, description="是否启用该模块")
    strategy: StrategyType = Field(
        default="auto",
        description="实现策略 (auto/ai/algorithm)"
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="模块专用参数"
    )
    priority: int = Field(
        default=0,
        description="模块优先级，数值越高优先级越高"
    )
    dependencies: List[ModuleName] = Field(
        default_factory=list,
        description="该模块依赖的其他模块"
    )

class AIConfig(BaseModel):
    """
    AI模型配置
    
    用于配置AI驱动的实现策略
    """
    enabled: bool = Field(
        default=True,
        description="是否启用AI模型"
    )
    chat_model: str = Field(
        default="gpt-3.5-turbo",
        description="AI模型名称"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding模型名称"
    )
    api_key: Optional[str] = Field(
        default=None, 
        description="API密钥"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="自定义API端点"
    )

class MMOSConfig(BaseModel):
    """
    全局配置中心
    
    管理MMOS系统的所有配置选项，包括模块开关、存储引擎和调试选项等
    """
    version: str = Field(
        default="1.0.0",
        description="配置版本号"
    )
    modules: Dict[ModuleName, ModuleConfig] = Field(
        default_factory=lambda: {
            "short_memory": ModuleConfig(enabled=True),
            "long_memory": ModuleConfig(enabled=False),
            "persona": ModuleConfig(enabled=False),
            "event": ModuleConfig(enabled=False)
        },
        description="模块开关及策略配置"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="存储引擎配置"
    )
    ai: AIConfig = Field(
        default_factory=AIConfig,
        description="AI模型配置"
    )
    debug: bool = Field(
        default=False,
        description="是否输出调试信息"
    )
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="日志级别"
    )

    def get_active_modules(self) -> list:
        """获取所有激活的模块名"""
        return [name for name, cfg in self.modules.items() if cfg.enabled]
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MMOSConfig":
        """从字典创建配置"""
        return cls.model_validate(data)
    
    def save_to_file(self, filepath: str) -> None:
        """保存配置到文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "MMOSConfig":
        """从文件加载配置"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)