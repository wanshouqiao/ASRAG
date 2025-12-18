"""
RAG 模块：负责协调 RAG 各组件，对外提供统一接口。
"""

import logging
import os
import time
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from app.rag_components.document_loader import load_document
from app.rag_components.embedding import create_embedding_model
from app.rag_components.generator import AnswerGenerator
from app.rag_components.retriever import MultimodalRetriever
from app.rag_components.text_splitter import split_documents
from app.rag_components.tts import TextToSpeech
from app.rag_components.vector_store import VectorStoreManager
from langchain_core.messages import HumanMessage, SystemMessage


class RAGModule:
    """封装并协调 RAG 工作流"""

    def __init__(
        self,
        llm_api_base: str,
        llm_api_key: str,
        model_name: str,
        base_model_path: str,
        visual_weight_path: str,
        base_dir: str,
    ):
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir

        # 1. 初始化嵌入模型
        self.embeddings = create_embedding_model(base_model_path, visual_weight_path)

        # 2. 初始化向量存储
        vs_dir = os.path.join(self.base_dir, "vectorstores", "combined_kb_bge_m3_visualized")
        self.vector_store_manager = VectorStoreManager(vs_dir, self.embeddings)

        # 3. 初始化检索器
        self.retriever = MultimodalRetriever(self.embeddings, self.vector_store_manager)

        # 4. 初始化 LLM 生成器
        llm_config = {
            "local": {
                "api_base": llm_api_base,
                "api_key": llm_api_key,
                "model_name": model_name,
                "supports_vision": True,
            }
        }
        self.generator = AnswerGenerator(llm_config)

        # 5. 初始化 TTS
        self.tts = TextToSpeech(
            model_path="/data/AI/LlamaCPPProject/tts/ckpts/kokoro-v1.1/kokoro-v1_1-zh.pth",
            config_path="/data/AI/LlamaCPPProject/tts/ckpts/kokoro-v1.1/config.json",
            voice_path="/data/AI/LlamaCPPProject/tts/ckpts/kokoro-v1.1/voices/zf_001.pt",
            repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        )

        # 检查向量库是否存在，不存在则从 uploads 目录构建
        if self.vector_store_manager.vectorstore is None:
            self.logger.info("首次启动或向量库不存在，从 uploads 目录构建...")
            documents_dir = os.path.join(self.base_dir, "uploads", "documents")
            images_dir = os.path.join(self.base_dir, "uploads", "images")
            self.rebuild_vectorstore(documents_dir, images_dir)

    @property
    def current_model_type(self):
        return self.generator.current_model_type

    @property
    def llm_config(self):
        return self.generator.llm_config

    def switch_llm(self, model_type: str):
        """切换 LLM 模型"""
        self.generator.switch_llm(model_type)

    def rebuild_vectorstore(self, documents_dir: str, images_dir: str = None):
        """从目录重建向量库"""
        all_docs: List[Document] = []

        # 扫描文档
        if os.path.exists(documents_dir):
            for filename in os.listdir(documents_dir):
                file_path = os.path.join(documents_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        docs = load_document(file_path)
                        chunks = split_documents(docs)
                        all_docs.extend(chunks)
                        self.logger.info("已加载并切分文档: %s", filename)
                    except Exception as e:
                        self.logger.error("处理文档失败 %s: %s", filename, e)

        # 扫描图片
        if images_dir and os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                file_path = os.path.join(images_dir, filename)
                if os.path.isfile(file_path):
                    title = os.path.splitext(filename)[0]
                    doc = Document(
                        page_content=f"image://{file_path}",
                        metadata={"source": file_path, "type": "image", "title": title},
                    )
                    all_docs.append(doc)
                    self.logger.info("已添加图片: %s (标题: %s)", filename, title)

        self.vector_store_manager.rebuild_from_documents(all_docs)

    def add_document(self, file_path: str):
        """添加单个文档到向量库"""
        try:
            docs = load_document(file_path)
            chunks = split_documents(docs)
            self.vector_store_manager.add_documents(chunks)
            self.logger.info("文档已成功添加到知识库: %s", os.path.basename(file_path))
        except Exception as e:
            self.logger.error("添加文档失败: %s", e)
            raise

    def add_image(self, image_path: str):
        """添加单个图片到向量库"""
        try:
            filename = os.path.basename(image_path)
            title = os.path.splitext(filename)[0]
            doc = Document(
                page_content=f"image://{image_path}",
                metadata={"source": image_path, "type": "image", "title": title},
            )
            self.vector_store_manager.add_documents([doc])
            self.logger.info("图片已成功添加到知识库: %s (标题: %s)", filename, title)
        except Exception as e:
            self.logger.error("添加图片失败: %s", e)
            raise

    def is_question(self, text: str) -> bool:
        """
        判断文本是否是问题或需要回答的请求。
        """
        if not text:
            return False
            
        try:
            messages = [
                SystemMessage(content="你是一个意图识别助手。请判断用户的输入是否是煤矿用电安全相关的问题或请求。如果是，请只回复'YES'。如果不是（例如只是陈述句、感叹句、无意义的词语、自言自语等），请只回复'NO'。"),
                HumanMessage(content=f"文本：{text}")
            ]
            response = self.llm.invoke(messages)
            content = response.content.strip().upper()
            self.logger.info(f"意图识别: '{text}' -> {content}")
            
            return "YES" in content
        except Exception as e:
            self.logger.error(f"意图识别失败: {e}")
            return True 


    def query(self, question: str = "", image_path: str = None) -> Tuple[str, Dict, List[Dict]]:
        """
        RAG 查询，支持文字和图片（图片功能待扩展）
        
        Args:
            question: 文字内容（可为空字符串）
            image_path: 图片文件路径（可选，None 表示无图片）
        
        Returns:
            (answer, timings, sources)
        """
        timings = {}
        
        if not question.strip() and not image_path:
            return "问题不能为空。", timings, []
        
        if self.vector_store_manager.vectorstore is None:
            return "知识库为空，请先上传文档。", timings, []

        try:
            # 1. 意图判断
            intent = self.generator.classify_intent(question, has_image=bool(image_path))
            is_vision_request = (intent == "vision_analysis")

            # 2. 检索
            t0 = time.time()
            if is_vision_request:
                # 图文分析需要图文检索
                retrieved_docs = self.retriever.retrieve(question, image_path)
            else:
                # 文本问答只进行文本检索，忽略图片
                retrieved_docs = self.retriever.retrieve(question, image_path=None)
            timings["retrieval"] = time.time() - t0

            # 3. 生成
            t0 = time.time()
            raw_answer = self.generator.generate_answer(
                question, 
                retrieved_docs, 
                query_image_path=image_path if is_vision_request else None, # 只有图文分析才传递图片路径
                is_vision_request=is_vision_request
            )
            timings["llm_generation"] = time.time() - t0

            # 3.5. 后处理：提取“回答”部分
            answer = raw_answer
            if is_vision_request and "【回答】" in raw_answer:
                answer = raw_answer.split("【回答】", 1)[-1].strip()
            elif is_vision_request and "【第二阶段：回答】" in raw_answer:
                answer = raw_answer.split("【第二阶段：回答】", 1)[-1].strip()

            # 4. 格式化来源
            sources = [
                {
                    "content": doc.page_content if not doc.page_content.startswith('image://') else f"图片: {os.path.basename(doc.metadata.get('source', ''))}",
                    "source": os.path.basename(doc.metadata.get("source", "未知来源")),
                    "type": doc.metadata.get("type", "text"),
                }
                for doc in retrieved_docs
            ]

            return answer, timings, sources
        except Exception as e:
            self.logger.error("RAG 查询失败: %s", e)
            return f"抱歉，生成回答时出现错误: {str(e)}", timings, []

    def text_to_speech(self, text: str):
        """调用 TTS 模块生成语音"""
        return self.tts.synthesize(text)
