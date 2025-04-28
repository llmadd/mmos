# MMOS - AI记忆管理系统

MMOS是一个用于AI系统记忆管理的Python库，它提供了一系列工具来帮助人工智能代理存储、检索和管理记忆信息。

## 特点

- 高效的记忆存储机制
- 语义化检索功能
- 记忆优先级管理
- 长短期记忆集成
- 易于与其他AI系统集成

## 安装

```bash
pip install mmos
```

## 快速开始

```python
from mmos import MemoryManager

# 初始化记忆管理器
memory_manager = MemoryManager()

# 存储新记忆
memory_manager.store("用户喜欢蓝色", tags=["用户偏好", "颜色"])

# 检索记忆
results = memory_manager.retrieve("用户喜欢什么颜色?")
print(results)
```

## 贡献

欢迎提交Pull Request和Issue。

## 许可证

MIT 