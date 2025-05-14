import os
import re
import pickle
import numpy as np
import jieba
import pymupdf
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Set, Union, Optional, Literal


# ===== 第一部分：文档加载和预处理 =====

def load_document(filepath: str) -> str:
    """
    从不同格式的文件中加载文本内容
    
    Args:
        filepath: 文件路径
        
    Returns:
        文件内容
    """
    try:
        ext = os.path.splitext(filepath)[1].lower()
        
        # PDF或DOCX文件使用pymupdf处理
        if ext in ['.pdf', '.docx', '.doc']:
            doc = pymupdf.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        
        # 文本文件直接读取
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            print(f"不支持的文件格式: {ext}")
            return ""
            
    except Exception as e:
        print(f"加载文件 {filepath} 时出错: {e}")
        return ""


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
                "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"]
    
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


# ===== 第二部分：MinHash相似度计算 =====

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
    
    for word in text.split():
        minhash.update(word.encode('utf8'))
        
    return minhash


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


def save_minhash(minhash: MinHash, filepath: str) -> None:
    """
    保存MinHash对象到文件
    
    Args:
        minhash: MinHash对象
        filepath: 保存路径
    """
    with open(filepath, "wb") as f:
        pickle.dump(minhash, f)


def load_minhash(filepath: str) -> MinHash:
    """
    从文件加载MinHash对象
    
    Args:
        filepath: MinHash文件路径
        
    Returns:
        MinHash对象
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ===== 第三部分：TF-IDF相似度计算 =====

def calculate_tfidf_similarity(docs: List[str]) -> float:
    """
    计算两个文档的TF-IDF余弦相似度
    
    Args:
        docs: 包含两个预处理后文档的列表
        
    Returns:
        余弦相似度(0-1之间)
    """
    if len(docs) != 2:
        raise ValueError("必须提供两个文档进行比较")
        
    # 为小样本调整参数
    vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


# ===== 第四部分：文档相似度分析 =====

class DocumentSimilarity:
    """文档相似度分析工具"""
    
    def __init__(self, num_perm: int = 256):
        """
        初始化文档相似度分析器
        
        Args:
            num_perm: MinHash排列数
        """
        self.num_perm = num_perm
        # 存储文档和MinHash
        self.documents = {}  # 文档ID -> 原始文本
        self.preprocessed_docs = {}  # 文档ID -> 预处理文本
        self.minhashes = {}  # 文档ID -> MinHash
        self.lsh_index = None  # LSH索引
        
    def add_document(self, doc_id: str, filepath: str) -> bool:
        """
        从文件添加文档
        
        Args:
            doc_id: 文档ID
            filepath: 文件路径
            
        Returns:
            是否成功添加
        """
        # 加载文档
        text = load_document(filepath)
        if not text:
            return False
            
        return self.add_document_text(doc_id, text)
    
    def add_document_text(self, doc_id: str, text: str) -> bool:
        """
        从文本添加文档
        
        Args:
            doc_id: 文档ID
            text: 文档文本
            
        Returns:
            是否成功添加
        """
        # 预处理
        preprocessed = preprocess_text(text)
        
        # 创建MinHash
        minhash = create_minhash(preprocessed, self.num_perm)
        
        # 保存
        self.documents[doc_id] = text
        self.preprocessed_docs[doc_id] = preprocessed
        self.minhashes[doc_id] = minhash
        
        # LSH索引失效
        self.lsh_index = None
        
        return True
    
    def build_lsh_index(self, threshold: float = 0.7) -> None:
        """
        构建LSH索引用于快速查询
        
        Args:
            threshold: 相似度阈值
        """
        self.lsh_index = MinHashLSH(threshold=threshold, num_perm=self.num_perm)
        
        for doc_id, minhash in self.minhashes.items():
            self.lsh_index.insert(doc_id, minhash)
    
    def find_similar(self, query_doc_id: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        查找与指定文档相似的其他文档
        
        Args:
            query_doc_id: 查询文档ID
            threshold: 相似度阈值
            
        Returns:
            (文档ID, 相似度)的列表，按相似度降序排序
        """
        if query_doc_id not in self.minhashes:
            raise ValueError(f"找不到文档: {query_doc_id}")
            
        # 如果没有LSH索引，则构建
        if not self.lsh_index:
            self.build_lsh_index(threshold)
            
        query_minhash = self.minhashes[query_doc_id]
        candidates = self.lsh_index.query(query_minhash)
        
        results = []
        for doc_id in candidates:
            if doc_id != query_doc_id:
                # 计算准确的相似度
                similarity = calculate_minhash_similarity(
                    query_minhash, self.minhashes[doc_id]
                )
                results.append((doc_id, similarity))
                
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def compare_documents(self, doc_id1: str, doc_id2: str) -> Dict[str, float]:
        """
        计算两个文档的相似度
        
        Args:
            doc_id1: 第一个文档ID
            doc_id2: 第二个文档ID
            
        Returns:
            包含不同相似度指标的字典
        """
        if doc_id1 not in self.preprocessed_docs or doc_id2 not in self.preprocessed_docs:
            raise ValueError("找不到一个或两个文档")
            
        # MinHash相似度
        minhash_sim = calculate_minhash_similarity(
            self.minhashes[doc_id1], self.minhashes[doc_id2]
        )
        
        # TF-IDF相似度
        tfidf_sim = calculate_tfidf_similarity([
            self.preprocessed_docs[doc_id1], 
            self.preprocessed_docs[doc_id2]
        ])
        
        # 组合相似度
        combined_sim = (minhash_sim + tfidf_sim) / 2
        
        return {
            "minhash": minhash_sim,
            "tfidf": tfidf_sim,
            "combined": combined_sim
        }
        
    def batch_compare(self, doc_ids: List[str]) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        批量比较多个文档
        
        Args:
            doc_ids: 要比较的文档ID列表
            
        Returns:
            文档对到相似度的映射
        """
        results = {}
        
        for i, doc_id1 in enumerate(doc_ids):
            for doc_id2 in doc_ids[i+1:]:
                results[(doc_id1, doc_id2)] = self.compare_documents(doc_id1, doc_id2)
                
        return results


# ===== 使用示例 =====

def simple_example(file_id: str = "1410937", file_path: Literal["改动较大", "改动较小", "新建文件夹"] = "新建文件夹"):
    """简单的文档比较示例"""


    # 加载两个文档
    pdf_path = f"/home/qichen/zh/test_data/提案公开件/新建文件夹/{file_id}.pdf"
    docx_path = f"/home/qichen/zh/test_data/提案公开件/{file_path}/{file_id}.docx"
    
    # 加载文本
    text1 = load_document(pdf_path)
    text2 = load_document(docx_path)
    
    # 文本预处理
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    # MinHash相似度计算
    minhash1 = create_minhash(processed_text1)
    minhash2 = create_minhash(processed_text2)
    minhash_similarity = calculate_minhash_similarity(minhash1, minhash2)
    print(f"两个文档的MinHash相似度为: {minhash_similarity}")
    
    # TF-IDF相似度计算
    tfidf_similarity = calculate_tfidf_similarity([processed_text1, processed_text2])
    print(f"两个文档的TF-IDF相似度为: {tfidf_similarity}")
    
    # 组合相似度
    combined_similarity = (minhash_similarity + tfidf_similarity) / 2
    print(f"两个文档的组合相似度为: {combined_similarity}")


def analyzer_example():
    """使用DocumentSimilarity类的示例"""
    pdf_path = "/home/qichen/zh/test_data/提案公开件/新建文件夹/1410464.pdf"
    docx_path = "/home/qichen/zh/test_data/提案公开件/新建文件夹/1410937.pdf"
    
    # 创建分析器
    analyzer = DocumentSimilarity()
    
    # 添加文档
    analyzer.add_document("doc1", pdf_path)
    analyzer.add_document("doc2", docx_path)
    
    # 比较文档
    similarity = analyzer.compare_documents("doc1", "doc2")
    print(f"文档相似度分析结果:")
    print(f"  MinHash相似度: {similarity['minhash']}")
    print(f"  TF-IDF相似度: {similarity['tfidf']}")
    print(f"  组合相似度: {similarity['combined']}")


if __name__ == "__main__":
    # 运行简单示例
    simple_example(file_id="1411014", file_path="改动较小")
    
    # 运行分析器示例
    # analyzer_example()

