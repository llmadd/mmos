import jieba
import re
from datasketch import MinHash




class MinHash_tools:
    def __init__(self):
        pass
        
    @staticmethod
    def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
        """
        文本预处理：清洗文本并分词
        
        Args:
            text: 原始文本
            remove_stopwords: 是否去除停用词
            
        Returns:
            处理后的文本（词语以空格分隔）
        """
        # 默认停用词列表
        stopwords = ["的", "了", "在", "是", "我", "有", "和", "就", "不", 
                    "人", "都", "一", "一个", "上", "也", "很", "到", "说", 
                    "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "什么"]
        
        # 1. 清洗文本
        # 合并多余空格
        text = re.sub(r'\s+', ' ', text)
        # 移除换行符等
        text = re.sub(r'[\n\r\t]', ' ', text)
        # 移除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        # 英文部分转小写
        text = text.lower()
        
        # 2. 分词
        words = list(jieba.cut(text))
        
        # 3. 过滤词语
        if remove_stopwords:
            words = [word for word in words if word not in stopwords and len(word.strip()) > 1]
        else:
            words = [word for word in words if len(word.strip()) > 1]
        
        return " ".join(words)


    @staticmethod
    def create_minhash(text: str, num_perm: int = 256) -> MinHash:
        """
        为文本创建MinHash签名
        
        Args:
            text: 预处理后的文本
            num_perm: 排列数，越高越准确但计算越慢
            
        Returns:
            MinHash对象
        """
        minhash = MinHash(num_perm=num_perm)
        text = MinHash_tools.preprocess_text(text)
        for word in text.split():
            minhash.update(word.encode('utf8'))
            
        return minhash


    @staticmethod
    def calculate_minhash_similarity(minhash1: MinHash, minhash2: MinHash) -> float:
        """
        计算两个MinHash的Jaccard相似度
        
        Args:
            minhash1: 第一个MinHash
            minhash2: 第二个MinHash
            
        Returns:
            Jaccard相似度(0-1之间)
        """
        return minhash1.jaccard(minhash2)
    

    @staticmethod
    def compare_with_history(new_message: str, history_messages: list) -> list:
        """
        比较新消息与历史消息的相似度
        
        Args:
            new_message: 新的用户消息内容
            history_messages: 历史消息列表（你提供的格式）
            
        Returns:
            包含(消息内容, 相似度分数)的列表，按相似度从高到低排序
        """
        # 为新消息创建MinHash
        minhash_tools = MinHash_tools()
        new_minhash = minhash_tools.create_minhash(new_message)
        
        # 存储相似度结果
        similarities = []
        
        # 遍历历史消息
        for msg in history_messages:
            # 只比较用户消息
            if msg['messages']['role'] == 'user' and msg['metadata']['vector'] is not None:
                # 计算相似度
                similarity = minhash_tools.calculate_minhash_similarity(new_minhash, msg['metadata']['vector'])
                similarities.append({
                    'content': msg['messages']['content'],
                    'similarity': similarity
                })
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities