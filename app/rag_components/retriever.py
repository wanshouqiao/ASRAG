"""
检索器模块：根据多模態查询从向量库检索上下文。
"""

import logging
import os
from typing import List, Tuple

from langchain_core.documents import Document

from app.rag_components.embedding import VisualizedBGEEmbeddings
from app.rag_components.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class MultimodalRetriever:
    """封装多模态检索逻辑"""

    def __init__(self, embeddings: VisualizedBGEEmbeddings, vector_store_manager: VectorStoreManager):
        self.embeddings = embeddings
        self.vector_store_manager = vector_store_manager

    def retrieve(self, question: str, image_path: str = None) -> Tuple[List[Document], str]:
        """
        执行多模态检索。

        Returns:
            (retrieved_docs, effective_question)
        """
        has_text = bool(question.strip())
        has_image = bool(image_path and os.path.exists(image_path))

        if not has_text and not has_image:
            raise ValueError("问题和图片不能同时为空")

        retrieved_docs: List[Document] = []

        try:
            if has_image:
                # 图检索
                query_vector = self.embeddings.embed_multimodal(image_path, question)
                retrieved_docs = self.vector_store_manager.similarity_search_by_vector(query_vector)
            else:
                # 纯文本检索
                retriever = self.vector_store_manager.get_retriever()
                if retriever:
                    retrieved_docs = retriever.invoke(question)
        except Exception as e:
            logger.warning("多模态检索失败，将回退到单模态：%s", e)
            # 回退策略
            if has_image:
                query_vector = self.embeddings.embed_image(image_path)
                retrieved_docs = self.vector_store_manager.similarity_search_by_vector(query_vector)
            else:
                retriever = self.vector_store_manager.get_retriever()
                if retriever:
                    retrieved_docs = retriever.invoke(question)
        
        return retrieved_docs

