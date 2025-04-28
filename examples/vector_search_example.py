"""
MMOS向量搜索示例
"""

from mmos import MemoryManager
from mmos.vector_store import SimpleVectorStore
from mmos.models import Memory

def main():
    # 初始化记忆管理器
    memory_manager = MemoryManager()
    
    # 初始化向量存储
    vector_store = SimpleVectorStore()
    
    # 添加一些记忆
    memories = [
        memory_manager.store("用户喜欢蓝色的衣服", tags=["用户偏好", "服装"]),
        memory_manager.store("用户的工作是软件工程师", tags=["用户信息", "职业"]),
        memory_manager.store("用户最喜欢的食物是意大利面", tags=["用户偏好", "食物"]),
        memory_manager.store("用户讨厌下雨天", tags=["用户偏好", "天气"]),
        memory_manager.store("用户养了一只叫小白的猫", tags=["用户信息", "宠物"]),
        memory_manager.store("用户每天早上喜欢喝咖啡", tags=["用户习惯", "饮料"]),
        memory_manager.store("用户偏好使用MacBook电脑工作", tags=["用户偏好", "电子设备"]),
    ]
    
    # 将记忆添加到向量存储
    for memory in memories:
        vector_store.add_memory(memory)
    
    # 执行语义搜索
    print("语义搜索示例：\n")
    
    # 查询示例
    queries = [
        "用户的职业是什么?",
        "用户喜欢什么颜色?",
        "用户有宠物吗?",
        "用户的饮食习惯",
        "用户使用什么电子产品?"
    ]
    
    for query in queries:
        print(f"查询: {query}")
        
        # 执行向量搜索
        results = vector_store.similarity_search(query, top_k=3)
        
        print("最相关的记忆:")
        for i, (memory_id, score) in enumerate(results):
            memory = memory_manager.get_by_id(memory_id)
            print(f"  {i+1}. {memory.content} (相似度: {score:.4f})")
        print()
    
    # 更新记忆示例
    print("更新记忆示例:")
    if memories:
        memory = memories[0]
        print(f"原记忆: {memory.content}")
        
        # 更新记忆内容
        updated_memory = memory_manager.update(memory.id, content="用户喜欢深蓝色的衣服")
        
        # 更新向量存储
        vector_store.update_memory(updated_memory)
        
        print(f"更新后: {updated_memory.content}")
        
        # 再次搜索
        print("\n更新后的搜索结果:")
        results = vector_store.similarity_search("用户喜欢什么颜色的衣服?", top_k=1)
        for memory_id, score in results:
            memory = memory_manager.get_by_id(memory_id)
            print(f"  {memory.content} (相似度: {score:.4f})")

if __name__ == "__main__":
    main() 