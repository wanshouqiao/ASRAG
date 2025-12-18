"""
文本切分器模块：将文档切分为适合向量化的小块。
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs: List[Document]) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 切分文档。

    Args:
        docs: 待切分的 LangChain Document 对象列表。

    Returns:
        切分后的 LangChain Document 对象列表。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    return splitter.split_documents(docs)

