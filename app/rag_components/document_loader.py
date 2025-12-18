"""
文档加载器模块：负责从不同类型的文件中加载和解析内容。
"""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 动态导入 PyMuPDFLoader
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    PyMuPDFLoader = None

logger = logging.getLogger(__name__)

def load_document(file_path: str) -> List[Document]:
    """
    根据文件扩展名加载文档。

    Args:
        file_path: 文档文件的路径。

    Returns:
        加载后的 LangChain Document 对象列表。
    """
    if file_path.lower().endswith(".pdf"):
        if PyMuPDFLoader:
            loader = PyMuPDFLoader(file_path)
            return loader.load()
        logger.warning("PyMuPDFLoader 不可用，回退到 PyPDFLoader")
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    # 默认使用 TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()

