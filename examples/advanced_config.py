"""
MMOS高级配置示例
演示配置的导入导出功能
"""

import os
from mmos import MMOSConfig, ModuleConfig, StorageConfig, AIConfig

def main():
    # 创建配置目录
    os.makedirs("./config", exist_ok=True)
    
    # 创建详细的自定义配置
    config = MMOSConfig(
        version="1.1.0",
        modules={
            "short_memory": ModuleConfig(
                enabled=True,
                strategy="algorithm",
                params={
                    "max_items": 500,
                    "recency_weight": 0.7,
                    "importance_weight": 0.3
                },
                priority=10
            ),
            "long_memory": ModuleConfig(
                enabled=True,
                strategy="ai",
                params={
                    "retention_days": 90,
                    "embedding_model": "text-embedding-ada-002",
                    "chunk_size": 1000
                },
                priority=5,
                dependencies=["short_memory"]
            ),
            "persona": ModuleConfig(
                enabled=True,
                strategy="ai",
                params={
                    "update_frequency": "daily",
                    "max_traits": 20
                },
                priority=8
            ),
            "event": ModuleConfig(
                enabled=True,
                strategy="algorithm",
                params={
                    "event_ttl": 30,  # 事件保留30天
                    "importance_threshold": 0.6  # 重要性阈值
                },
                priority=3,
                dependencies=["long_memory"]
            )
        },
        storage=StorageConfig(
            vector_db="chromadb",
            graph_db="neo4j",
            data_path="./data/memory_storage",
            cache_size=2048,
            auto_save=True
        ),
        ai=AIConfig(
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=2000
        ),
        debug=True,
        log_level="debug"
    )
    
    # 保存配置到文件
    config_path = "./config/mmos_config.json"
    config.save_to_file(config_path)
    print(f"配置已保存到: {config_path}")
    
    # 加载配置
    loaded_config = MMOSConfig.load_from_file(config_path)
    print("\n加载配置成功!")
    
    # 查看和修改配置
    print(f"配置版本: {loaded_config.version}")
    print(f"启用的模块: {loaded_config.get_active_modules()}")
    print(f"长期记忆参数: {loaded_config.modules['long_memory'].params}")
    
    # 修改配置
    loaded_config.modules["long_memory"].params["retention_days"] = 120
    loaded_config.ai.model_name = "gpt-3.5-turbo-16k"
    
    # 保存修改后的配置
    loaded_config.save_to_file("./config/mmos_config_modified.json")
    print("\n修改后的配置已保存到: ./config/mmos_config_modified.json")
    
    # 展示不同策略的模块
    ai_modules = [name for name, cfg in loaded_config.modules.items() 
                 if cfg.enabled and cfg.strategy == "ai"]
    
    algorithm_modules = [name for name, cfg in loaded_config.modules.items() 
                        if cfg.enabled and cfg.strategy == "algorithm"]
    
    print(f"\nAI驱动的模块: {ai_modules}")
    print(f"算法驱动的模块: {algorithm_modules}")
    
    # 按优先级排序的模块
    priority_modules = [(name, cfg.priority) for name, cfg in loaded_config.modules.items() 
                        if cfg.enabled]
    priority_modules.sort(key=lambda x: x[1], reverse=True)
    
    print("\n按优先级排序的模块:")
    for name, priority in priority_modules:
        print(f"- {name}: {priority}")

if __name__ == "__main__":
    main() 