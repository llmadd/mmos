from typing import List
from _minhash import MinHash_tools



class ShortMemory:
    def __init__(self,  hot_memory: int = 2, warm_memory: int = 10, cold_memory: int = 20):
        self.hot_memory = hot_memory
        self.cold_memory = cold_memory
        self.warm_memory = warm_memory

    
    def cut_messages(self, messages: List[dict]):
        new_messages = []
        current_chunk = []
        for message in messages:
            if message["role"] == "user":
                if current_chunk:
                    new_messages.append(current_chunk)
                    current_chunk = []
                current_chunk.append(message)
            else:
                current_chunk.append(message)
        if current_chunk:
            new_messages.append(current_chunk)
        return new_messages


    def get_score(self, text: str, num_perm: int = 256):
        minhash = MinHash_tools.create_minhash(text, num_perm)
        
        return minhash
    
    def add_metadata(self, chunks: List[dict]):
        result = []
    
        for chunk in chunks:
            roles = [msg['role'] for msg in chunk]
            
            for msg in chunk:
                processed_msg = {
                    "messages": msg,
                    "metadata": {
                        "role": roles,  
                        "vector": self.get_score(msg['content']) if msg['role'] == 'user' else None,  
                        "include_user": any(m['role'] == 'user' for m in chunk)  
                    }
                }
                result.append(processed_msg)
                
        return result

    def summary_assistant_message(self, messages: List[dict]):
        pass
        
    def rerank_messages(self, messages: List[dict]):
        """
        对消息列表进行重排序，只对包含user消息的块进行排序，其他块保持原位置不变
        
        Args:
            messages: 消息列表，每个元素包含messages和metadata
            
        Returns:
            重排序后的消息列表
        """
        # 分离需要排序的消息和不需要排序的消息
        to_rank = []
        fixed_positions = {}  # 存储不需要排序的消息的原始位置
        
        for i, msg in enumerate(messages):
            if msg['metadata']['include_user']:
                to_rank.append((i, msg))
            else:
                fixed_positions[i] = msg
                
        if not to_rank:  # 如果没有需要排序的消息，直接返回原列表
            return messages
            
        # 获取最后一条用户消息的内容
        last_user_msg = None
        for msg in reversed(messages):
            if msg['messages']['role'] == 'user':
                last_user_msg = msg['messages']['content']
                break
                
        if not last_user_msg:  # 如果没找到用户消息，返回原列表
            return messages
            
        # 计算相似度并排序
        similarities = []
        minhash_tools = MinHash_tools()
        query_minhash = minhash_tools.create_minhash(last_user_msg)
        
        for orig_pos, msg in to_rank:
            # 找到该块中的user消息的vector
            user_vector = None
            for m in messages:
                if m['messages']['role'] == 'user' and m['metadata']['vector'] is not None:
                    user_vector = m['metadata']['vector']
                    break
                    
            if user_vector is not None:
                similarity = minhash_tools.calculate_minhash_similarity(query_minhash, user_vector)
                similarities.append((orig_pos, msg, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # 重建消息列表，保持非user块的位置不变
        result = [None] * len(messages)
        
        # 先放入固定位置的消息
        for pos, msg in fixed_positions.items():
            result[pos] = msg
            
        # 按相似度顺序填入需要排序的消息
        current_pos = 0
        for _, msg, _ in similarities:
            # 找到下一个可用的位置
            while current_pos < len(result) and result[current_pos] is not None:
                current_pos += 1
            if current_pos < len(result):
                result[current_pos] = msg
                
        return result

    def cold_messages_load(self, messages: List[dict]):
        pass

    def warm_messages_load(self, messages: List[dict]):
        pass
    
    def hot_messages_load(self, messages: List[dict]):
        pass

    def run(self, messages: List[dict]):
        mes_chunk = self.cut_messages(messages)
        if len(mes_chunk) <= self.cold_memory:
            return messages
        elif len(mes_chunk) <= self.warm_memory:
            return self.rerank_messages(mes_chunk)
        else:
            return self.summary_assistant_message(mes_chunk)
        
        
        
        



if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "你是一个AI助手，请根据用户的问题给出回答。"},
        {"role": "user", "content": "开心"},
        {"role": "tools", "content": "tools_call"},
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "哈哈哈哈哈"},
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "开心"},
    ]
    short_memory = ShortMemory()
    chunk = short_memory.cut_messages(messages)
    new_chunk = short_memory.add_metadata(chunk)
    res = short_memory.rerank_messages(new_chunk)
    print(res)
    
        
        
        
        