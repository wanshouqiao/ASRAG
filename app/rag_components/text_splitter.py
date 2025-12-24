"""
文本切分器模块：将文档切分为适合向量化的小块。
"""
import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# 新正则：匹配行首序号，且不以括号开头，避免“（一）”这种子标题
TITLE_PATTERN = re.compile(r'^\s*([一二三四五六七八九十百千万〇零\d]+[、\.．])\s*[^（(].*$', re.M)

fine_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)

def _split_by_title(text: str) -> List[str]:
    """
    先按标题行粗分；返回段落列表（含标题）。
    """
    parts = []
    last_idx = 0
    for m in TITLE_PATTERN.finditer(text):
        start = m.start()
        if start != 0:
            parts.append(text[last_idx:start])
        last_idx = start
    parts.append(text[last_idx:])
    return [p.strip() for p in parts if p.strip()]

def _merge_short_chunks(docs: List[Document], min_len: int = 50) -> List[Document]:
    """
    合并过短的文本块。
    """
    if not docs:
        return []
    merged = []
    current_content = ""
    for doc in docs:
        # 如果当前块本身就很大，直接作为一个独立块
        if len(doc.page_content) >= min_len:
            if current_content:
                merged.append(Document(page_content=current_content, metadata=doc.metadata))
            current_content = doc.page_content
        else:
            current_content += "\n" + doc.page_content
    
    if current_content:
        # 确保最后一个块也被添加
        merged.append(Document(page_content=current_content, metadata=docs[-1].metadata))

    # 进一步处理合并后可能产生的超长块
    final_docs = []
    for doc in merged:
        if len(doc.page_content) > fine_splitter._chunk_size:
            sub_chunks = fine_splitter.split_text(doc.page_content)
            for sub_chunk in sub_chunks:
                final_docs.append(Document(page_content=sub_chunk, metadata=doc.metadata))
        else:
            final_docs.append(doc)
    return final_docs

def split_documents(docs: List[Document]) -> List[Document]:
    """
    三级切分：标题 → 长度 → 合并短块。
    """
    all_fine_docs: List[Document] = []
    for doc in docs:
        raw_text = doc.page_content
        coarse_chunks = _split_by_title(raw_text)

        for coarse in coarse_chunks:
            fine_chunks = fine_splitter.split_text(coarse)
            for chunk in fine_chunks:
                all_fine_docs.append(
                    Document(page_content=chunk, metadata=doc.metadata.copy())
                )

    # 3) 合并过短的块
    results = _merge_short_chunks(all_fine_docs, min_len=150) # 可调阈值

    if results:
        logger.info(f"Split into {len(results)} chunks "
                    f"(max_len={max(len(d.page_content) for d in results)}, "
                    f"min_len={min(len(d.page_content) for d in results)})")
    else:
        logger.warning("Text splitting resulted in 0 chunks.")
    return results
