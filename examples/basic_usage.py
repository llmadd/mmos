"""
MMOS基本使用示例
"""

from mmos import MemoryManager

def main():
    # 初始化记忆管理器
    memory_manager = MemoryManager()
    
    # 存储一些记忆
    memory_manager.store("用户喜欢蓝色", tags=["用户偏好", "颜色"])
    memory_manager.store("用户的生日是10月1日", tags=["用户信息", "日期"])
    memory_manager.store("用户讨厌辣的食物", tags=["用户偏好", "食物"])
    memory_manager.store("用户今天很高兴", tags=["用户状态", "情绪"], importance=0.8)
    
    # 检索记忆
    print("查找关于颜色偏好的记忆:")
    results = memory_manager.retrieve("颜色")
    for memory in results:
        print(f"- {memory.content} (重要性: {memory.importance})")
    
    print("\n根据标签查找:")
    tag_results = memory_manager.get_by_tags(["用户偏好"])
    for memory in tag_results:
        print(f"- {memory.content} (标签: {', '.join(memory.tags)})")
    
    # 更新记忆
    if tag_results:
        updated = memory_manager.update(
            tag_results[0].id,
            importance=0.9
        )
        if updated:
            print(f"\n已更新: {updated.content} (新重要性: {updated.importance})")
    
    # 统计
    print(f"\n总记忆数: {memory_manager.count()}")

if __name__ == "__main__":
    main() 