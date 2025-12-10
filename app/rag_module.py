#!/usr/bin/env python3
"""
RAG 模块：负责向量库管理、LLM 调用、RAG 查询、TTS。
保持原有逻辑，不增加新功能。
"""

import io
import json
import logging
import os
import time
from typing import Dict, List, Tuple

import soundfile as sf
import torch
from kokoro import KModel, KPipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:  # pragma: no cover - optional dependency
    PyMuPDFLoader = None


class RAGModule:
    """封装 LLM、向量库、TTS 与 RAG 查询"""

    def __init__(
        self,
        kb_path: str,
        llm_api_base: str,
        llm_api_key: str,
        model_name: str,
        embedding_model: str,
        base_dir: str,
    ):
        self.logger = logging.getLogger(__name__)
        self.embedding_model_name = embedding_model
        self.base_dir = base_dir

        # LLM 配置
        self.llm_config = {
            "local": {
                "api_base": llm_api_base,
                "api_key": llm_api_key,
                "model_name": model_name,
            }
        }
        self.current_model_type = "local"
        self.llm = None

        # 初始化
        self.switch_llm("local")
        self.embeddings = self._init_embeddings()
        self.vectorstore = None
        self.retriever = None
        self._init_vectorstore(kb_path)

        # TTS
        self.tts_device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_repo_id = "hexgrad/Kokoro-82M-v1.1-zh"
        tts_model_path = "/data/AI/LlamaCPPProject/tts/ckpts/kokoro-v1.1/kokoro-v1_1-zh.pth"
        tts_config_path = "/data/AI/LlamaCPPProject/tts/ckpts/kokoro-v1.1/config.json"
        voice_path = "/data/AI/LlamaCPPProject/tts/ckpts/kokoro-v1.1/voices/zf_001.pt"
        self.voice_tensor = torch.load(voice_path, weights_only=True)
        self.tts_model = KModel(model=tts_model_path, config=tts_config_path, repo_id=tts_repo_id).to(
            self.tts_device
        ).eval()
        self.en_pipeline = KPipeline(lang_code="a", repo_id=tts_repo_id, model=False)

        def en_callable(text):
            return next(self.en_pipeline(text)).phonemes

        self.tts_pipeline = KPipeline(
            lang_code="z", repo_id=tts_repo_id, model=self.tts_model, en_callable=en_callable
        )
        self.speed_callable = self._build_speed_callable()

    # --- 内部初始化 ---
    def _build_speed_callable(self):
        def speed_callable(len_ps):
            speed = 0.8
            if len_ps <= 83:
                speed = 1
            elif len_ps < 183:
                speed = 1 - (len_ps - 83) / 500
            return speed * 1.1

        return speed_callable

    def _init_embeddings(self):
        embedding_path = os.path.join(self.base_dir, "embedding")
        encode_kwargs = {"normalize_embeddings": True}
        if os.path.exists(embedding_path) and os.path.exists(os.path.join(embedding_path, "modules.json")):
            self.logger.info("使用本地嵌入模型: %s", embedding_path)
            return HuggingFaceEmbeddings(
                model_name=embedding_path, model_kwargs={"device": "cuda"}, encode_kwargs=encode_kwargs
            )
        self.logger.info("使用在线嵌入模型: %s", self.embedding_model_name)
        self.logger.info("注意：首次使用会下载模型到本地缓存，之后会使用缓存")
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs=encode_kwargs,
        )

    def _get_combined_vectorstore_dir(self) -> str:
        model_name = os.path.basename(self.embedding_model_name)
        safe_model_name = (
            model_name.replace("/", "_").replace("\\", "_").replace("-", "_").replace(":", "_")
        )
        return os.path.join(self.base_dir, "vectorstores", f"combined_kb_{safe_model_name}")

    def _init_vectorstore(self, default_kb_path: str):
        vs_dir = self._get_combined_vectorstore_dir()
        if os.path.exists(os.path.join(vs_dir, "index.faiss")):
            self.logger.info("检测到已保存的向量库，正在加载...")
            faiss_logger = logging.getLogger("faiss.loader")
            original_level = faiss_logger.level
            faiss_logger.setLevel(logging.ERROR)
            try:
                self.vectorstore = FAISS.load_local(
                    vs_dir, self.embeddings, allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                self.logger.info("✓ 向量库加载成功！")
                return
            except Exception as e:  # pragma: no cover
                self.logger.error("加载向量库失败: %s，将尝试重新创建", e)
            finally:
                faiss_logger.setLevel(original_level)

        if os.path.exists(default_kb_path):
            self.logger.info("首次启动，从默认文件创建向量库: %s", os.path.basename(default_kb_path))
            self.add_document(default_kb_path)
        else:
            self.logger.warning("未找到已保存的向量库，也未找到默认知识库文件")
            self.logger.info("提示: 启动后可通过 Web 界面上传文档来创建知识库")
            self.vectorstore = None
            self.retriever = None

    # --- 公共方法 ---
    def switch_llm(self, model_type: str):
        if model_type not in self.llm_config:
            raise ValueError(f"不支持的模型类型: {model_type}")
        config = self.llm_config[model_type]
        self.logger.info("切换 LLM 到: %s (%s)", model_type, config["model_name"])
        self.llm = ChatOpenAI(
            openai_api_base=config["api_base"],
            openai_api_key=config["api_key"],
            model_name=config["model_name"],
            temperature=0.01,
            max_tokens=512,
        )
        self.current_model_type = model_type

    def rebuild_vectorstore(self, documents_dir: str, images_dir: str = None):
        """
        重建向量库，扫描文档目录和图片目录
        
        Args:
            documents_dir: 文档目录路径
            images_dir: 图片目录路径（可选，图片向量化功能待扩展）
        """
        self.logger.info("正在重建向量库...")
        self.vectorstore = None
        self.retriever = None
        
        # 扫描文档目录
        doc_files = []
        if os.path.exists(documents_dir):
            doc_files = [f for f in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, f))]
        
        # 扫描图片目录（暂时只记录，不参与向量化）
        image_files = []
        if images_dir and os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
            if image_files:
                self.logger.info("检测到 %d 张图片（图片向量化功能待扩展）", len(image_files))
        
        if not doc_files and not image_files:
            self.logger.info("没有文件，向量库为空")
            vs_dir = self._get_combined_vectorstore_dir()
            import shutil

            if os.path.exists(vs_dir):
                shutil.rmtree(vs_dir)
            return

        # 只对文档进行向量化
        all_chunks = []
        for filename in doc_files:
            file_path = os.path.join(documents_dir, filename)
            try:
                docs = self._load_docs(file_path)
                chunks = self._split_docs(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error("读取文件 %s 失败: %s", filename, e)

        if all_chunks:
            self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            vs_dir = self._get_combined_vectorstore_dir()
            os.makedirs(vs_dir, exist_ok=True)
            self.vectorstore.save_local(vs_dir)
            self.logger.info("向量库重建完成，共 %d 个文档文件", len(doc_files))
            if image_files:
                self.logger.info("另有 %d 张图片（未参与向量化）", len(image_files))

    def add_document(self, file_path: str):
        self.logger.info("正在处理文档: %s", file_path)
        try:
            docs = self._load_docs(file_path)
            chunks = self._split_docs(docs)
            if not chunks:
                self.logger.warning("文档为空或无法切分")
                return
            if self.vectorstore is None:
                self.logger.info("创建新向量库...")
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            else:
                self.logger.info("添加到现有向量库...")
                self.vectorstore.add_documents(chunks)
            vs_dir = self._get_combined_vectorstore_dir()
            os.makedirs(vs_dir, exist_ok=True)
            self.vectorstore.save_local(vs_dir)
            self.logger.info("向量库已更新并保存到: %s", vs_dir)
        except Exception as e:
            self.logger.error("添加文档失败: %s", e)
            raise

    def add_image(self, image_path: str):
        """
        添加图片到知识库（图片向量化功能待扩展）
        
        Args:
            image_path: 图片文件路径
        """
        self.logger.info("图片已保存到知识库: %s（图片向量化功能待扩展）", image_path)
        # 暂时只记录日志，不进行向量化
        # 后续可以添加 OCR 或图片描述生成功能

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
        
        # 校验：文字和图片不能同时为空
        if not question.strip() and not image_path:
            return "问题不能为空。", timings, []
        
        # 图片处理（暂时只记录日志，保留后续扩展）
        if image_path:
            if os.path.exists(image_path):
                self.logger.info("收到图片: %s（图片处理功能待扩展）", image_path)
            else:
                self.logger.warning("图片路径不存在: %s", image_path)
        
        # 如果只有图片没有文字，返回提示
        if not question.strip() and image_path:
            return "当前仅支持文字查询，图片功能待扩展。", timings, []
        
        if not self.retriever:
            return "知识库为空，请先上传文档。", timings, []
        
        self.logger.info("用户问题: %s", question if question.strip() else "[无文字，仅图片]")
        try:
            t0 = time.time()
            docs_retrieved = self.retriever.invoke(question)
            timings["retrieval"] = time.time() - t0
            context = "\n\n".join(d.page_content for d in docs_retrieved)
            prompt = (
                "你是一个中文助理，请严格依据下面提供的知识库内容回答用户问题，"
                "如果知识库中没有相关信息，就说不知道，不要编造，也不要扩展。\n\n"
                f"【知识库内容】:\n{context}\n\n"
                f"【用户问题】:\n{question}\n\n"
                "请用简体中文回答："
            )
            self.logger.info("正在生成回答...")
            t0 = time.time()
            response = self.llm.invoke(prompt)
            timings["llm_generation"] = time.time() - t0
            answer = response.content
            self.logger.info("回答生成完成")
            sources = [
                {"content": doc.page_content, "source": os.path.basename(doc.metadata.get("source", "未知来源"))}
                for doc in docs_retrieved
            ]
            return answer, timings, sources
        except Exception as e:
            self.logger.error("RAG 查询失败: %s", e)
            return f"抱歉，生成回答时出现错误: {str(e)}", timings, []

    def text_to_speech(self, text: str):
        try:
            clean_text = text.replace("**", "").replace("*", "").strip()
            self.logger.info("正在生成语音: %s...", clean_text[:50])
            generator = self.tts_pipeline(clean_text, voice=self.voice_tensor, speed=self.speed_callable)
            result = next(generator)
            wav = result.audio
            wav_io = io.BytesIO()
            sf.write(wav_io, wav, 24000, format="WAV")
            return wav_io.getvalue()
        except Exception as e:
            self.logger.error("TTS 生成失败: %s", e)
            import traceback

            self.logger.error(traceback.format_exc())
            return None

    # --- 工具方法 ---
    def _load_docs(self, file_path: str):
        if file_path.lower().endswith(".pdf"):
            if PyMuPDFLoader:
                loader = PyMuPDFLoader(file_path)
                return loader.load()
            self.logger.warning("PyMuPDFLoader 不可用，回退到 PyPDFLoader")
            loader = PyPDFLoader(file_path)
            return loader.load()
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def _split_docs(self, docs):
        return RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        ).split_documents(docs)


