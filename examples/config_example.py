"""
MMOS配置和模块化记忆系统示例
"""

from mmos import MMOSConfig, ModuleConfig, StorageConfig, MMOSMemorySystem

def main():
    # 创建自定义配置
    config = MMOSConfig(
        modules={
            "short_memory": ModuleConfig(
                enabled=True,
                strategy="algorithm",
                params={"max_items": 100}
            ),
            "long_memory": ModuleConfig(
                enabled=True,
                strategy="ai",
                params={"retention_days": 30}
            ),
            "persona": ModuleConfig(
                enabled=True,
                strategy="auto"
            ),
            "event": ModuleConfig(
                enabled=False
            )
        },
        storage=StorageConfig(
            vector_db="chromadb",
            graph_db="neo4j"
        ),
        debug=True
    )
    
    # 打印活跃模块
    print("配置的活跃模块:", config.get_active_modules())
    
    # 初始化记忆系统
    memory_system = MMOSMemorySystem(config)
    
    # 存储一些记忆
    memory_system.store_memory(
        "用户喜欢蓝色的衣服", 
        tags=["用户偏好", "服装"],
        importance=0.8
    )
    
    memory_system.store_memory(
        "用户是软件工程师", 
        tags=["用户信息", "职业"],
        importance=0.7
    )
    
    # 检索记忆
    results = memory_system.retrieve_memory("用户喜欢什么颜色")
    print("\n检索结果:")
    for memory in results:
        print(f"- {memory.content} (重要性: {memory.importance})")
    
    # 获取短期记忆模块
    short_memory = memory_system.get_module("short_memory")
    print(f"\n短期记忆模块策略: {short_memory.strategy}")
    
    # 获取长期记忆模块
    long_memory = memory_system.get_module("long_memory")
    print(f"长期记忆模块策略: {long_memory.strategy}")
    
    # 获取角色信息模块
    persona = memory_system.get_module("persona")
    print(f"角色信息模块策略: {persona.strategy}")
    
    # 事件模块未启用
    event = memory_system.get_module("event")
    print(f"事件模块: {'已启用' if event else '未启用'}")

if __name__ == "__main__":
    main() 