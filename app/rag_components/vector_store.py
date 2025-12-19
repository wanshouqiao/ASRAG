"""
向量存储模块：管理 FAISS 向量数据库。
"""

import logging
import os
import shutil
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """封装 FAISS 向量库的加载、保存、添加和检索"""

    def __init__(self, vectorstore_dir: str, embeddings: Embeddings):
        self.vectorstore_dir = vectorstore_dir
        self.embeddings = embeddings
        self.vectorstore: Optional[FAISS] = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        """从本地加载 FAISS 向量库"""
        index_path = os.path.join(self.vectorstore_dir, "index.faiss")
        if os.path.exists(index_path):
            logger.info("检测到已保存的向量库，正在加载...")
            faiss_logger = logging.getLogger("faiss.loader")
            original_level = faiss_logger.level
            faiss_logger.setLevel(logging.ERROR)
            try:
                self.vectorstore = FAISS.load_local(
                    self.vectorstore_dir, self.embeddings, allow_dangerous_deserialization=True
                )
                logger.info("✓ 向量库加载成功！")
            except Exception as e:
                logger.error("加载向量库失败: %s，将创建一个新的空库", e)
                self.vectorstore = None
            finally:
                faiss_logger.setLevel(original_level)
        else:
            logger.info("未找到本地向量库，将创建一个新的空库")
            self.vectorstore = None

    def save_vectorstore(self):
        """保存向量库到本地"""
        if self.vectorstore:
            os.makedirs(self.vectorstore_dir, exist_ok=True)
            self.vectorstore.save_local(self.vectorstore_dir)
            logger.info("向量库已保存到: %s", self.vectorstore_dir)

    def add_documents(self, documents: List[Document]):
        """向向量库中添加文档"""
        if not documents:
            return
        if self.vectorstore is None:
            logger.info("创建新向量库...")
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            logger.info("添加到现有向量库...")
            self.vectorstore.add_documents(documents)
        self.save_vectorstore()

    def rebuild_from_documents(self, documents: List[Document]):
        """从文档列表完全重建向量库"""
        logger.info("正在重建向量库...")
        if os.path.exists(self.vectorstore_dir):
            shutil.rmtree(self.vectorstore_dir)
        
        if not documents:
            self.vectorstore = None
            logger.info("没有找到任何文档，向量库已清空")
            return

        logger.info("开始批量向量化 %d 个项目...", len(documents))
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.save_vectorstore()
        logger.info("✓ 向量库重建完成")

    def get_retriever(self, k: int = 5):
        """获取 LangChain 检索器"""
        if self.vectorstore:
            return self.vectorstore.as_retriever(search_kwargs={"k": k})
        return None

    def similarity_search_by_vector(self, embedding: List[float], k: int = 3) -> List[Document]:
        """通过向量进行相似性搜索"""
        if self.vectorstore:
            return self.vectorstore.similarity_search_by_vector(embedding, k=k)
        return []

