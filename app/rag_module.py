#!/usr/bin/env python3
"""
RAG 模块：负责向量库管理、LLM 调用、RAG 查询、TTS。
使用 bge-m3 多模态模型进行文本和图片嵌入。
"""

import io
import json
import logging
import os
import shutil
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import soundfile as sf
import torch
import base64
import httpx
from pathlib import Path
import numpy as np
from kokoro import KModel, KPipeline
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

# 动态导入 visual_bge
try:
    from visual_bge.modeling import Visualized_BGE
except ImportError:
    raise ImportError("请确保 visual_bge 已安装或在 Python 路径中。")

try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:  # pragma: no cover - optional dependency
    PyMuPDFLoader = None


class VisualizedBGEEmbeddings(Embeddings):
    """
    LangChain Embeddings 包装类，用于 Visualized_BGE 多模态模型。
    支持文本与图片；图片以前缀 image:// 路径字符串表示。
    """

    def __init__(self, model: Visualized_BGE):
        self.model = model

    def _encode_text(self, text: str) -> List[float]:
        """
        使用 Visualized_BGE 的 encode_text，返回单条文本的向量。
        """
        try:
            device = next(self.model.parameters()).device
            inputs = self.model.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            vec = self.model.encode_text(inputs)  # Tensor [1, dim]
            return vec[0].tolist()
        except Exception as e:
            logging.exception("文本编码失败: %s", text[:50])
            raise

    def _encode_image(self, image_path: str) -> List[float]:
        try:
            image = Image.open(image_path).convert("RGB")
            # 视觉模型需要 BCHW Tensor，使用模型自带预处理
            pixel = self.model.preprocess_val(image)          # CHW tensor
            pixel = pixel.unsqueeze(0)                        # BCHW
            pixel = pixel.to(next(self.model.parameters()).device)
            vec = self.model.encode_image(pixel)              # [1, dim]
            return vec[0].tolist()
        except Exception as e:
            logging.exception("图片编码失败: %s", image_path)
            raise

    def _encode_multimodal(self, image_path: str, text: str) -> List[float]:
        """
        同时使用图片 + 文本生成联合向量（模型原生支持）。
        """
        try:
            res = self.model.encode(image=image_path, text=text)

            # 归一化为 1D 向量
            def to_array(obj):
                if obj is None:
                    return None
                if hasattr(obj, "detach"):
                    obj = obj.detach()
                if hasattr(obj, "cpu"):
                    obj = obj.cpu()
                if hasattr(obj, "numpy"):
                    return obj.numpy()
                try:
                    return np.array(obj)
                except Exception:
                    return None

            candidates = []
            # 直接 tensor/ndarray
            arr = to_array(res)
            if arr is not None:
                candidates.append(arr)

            # list/tuple: 取能转 array 的
            if isinstance(res, (list, tuple)):
                for item in res:
                    arr_item = to_array(item)
                    if arr_item is not None:
                        candidates.append(arr_item)
                    if isinstance(item, dict):
                        dense = item.get("dense_vecs") or item.get("embedding")
                        arr_dense = to_array(dense)
                        if arr_dense is not None:
                            candidates.append(arr_dense)

            # dict
            if isinstance(res, dict):
                dense = res.get("dense_vecs") or res.get("embedding")
                arr_dense = to_array(dense)
                if arr_dense is not None:
                    candidates.append(arr_dense)
                for v in res.values():
                    arr_v = to_array(v)
                    if arr_v is not None:
                        candidates.append(arr_v)
                    if isinstance(v, (list, tuple)):
                        for item in v:
                            arr_item = to_array(item)
                            if arr_item is not None:
                                candidates.append(arr_item)

            for arr in candidates:
                if arr is None:
                    continue
                arr = np.array(arr)
                if arr.ndim > 1:
                    arr = arr[0]
                if arr.ndim == 0:
                    continue
                vec = arr.astype(float).tolist()
                if vec:
                    return vec

            raise ValueError(f"多模态编码返回未知格式，无法提取向量: type={type(res)}")
        except Exception:
            logging.exception("多模态编码失败: image=%s, text=%s", image_path, text[:50])
            raise

    def _encode_text_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本编码，减少 tokenizer 调用次数。
        """
        device = next(self.model.parameters()).device
        inputs = self.model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        vec = self.model.encode_text(inputs)  # [B, dim]
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 分离文本与图片，批量编码文本以减少 tokenizer 调用
        vectors: List[List[float]] = [None] * len(texts)  # type: ignore
        text_items: List[Tuple[int, str]] = []
        for idx, text in enumerate(texts):
            if isinstance(text, str) and text.startswith("image://"):
                vectors[idx] = self._encode_image(text[len("image://") :])
            else:
                text_items.append((idx, text))

        if text_items:
            idxs, text_batch = zip(*text_items)
            text_batch = list(text_batch)
            
            # 分批处理文本，防止显存溢出
            batch_size = 8  # 减小 batch size 以节省显存
            all_text_vecs = []
            
            for i in range(0, len(text_batch), batch_size):
                batch = text_batch[i : i + batch_size]
                try:
                    batch_vecs = self._encode_text_batch(batch)
                    all_text_vecs.extend(batch_vecs)
                    # 清理显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"批次编码失败 (index {i}): {e}")
                    raise e

            for i, v in zip(idxs, all_text_vecs):
                vectors[i] = v

        if any(len(v) == 0 for v in vectors):
            raise ValueError("编码结果为空，可能图片或文本编码失败")
        return vectors

    def embed_query(self, text: str) -> List[float]:
        if isinstance(text, str) and text.startswith("image://"):
            return self._encode_image(text[len("image://") :])
        return self._encode_text(text)

    def embed_image(self, image_path: str) -> List[float]:
        return self._encode_image(image_path)

    def embed_multimodal(self, image_path: str, text: str) -> List[float]:
        return self._encode_multimodal(image_path, text)


class RAGModule:
    """封装 LLM、向量库、TTS 与 RAG 查询"""

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
        self.base_model_path = base_model_path
        self.visual_weight_path = visual_weight_path
        self.base_dir = base_dir

        # LLM 配置
        self.llm_config = {
            "local": {
                "api_base": llm_api_base,
                "api_key": llm_api_key,
                "model_name": model_name,
                "supports_vision": True,  # 假设本地接口兼容 OpenAI 图文格式
            }
        }
        self.current_model_type = "local"
        self.llm = None

        # 初始化
        self.switch_llm("local")
        self.embeddings = self._init_embeddings()
        self.vectorstore = None
        self.retriever = None
        self._init_vectorstore()

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
        # 如果使用CUDA，尝试使用半精度（FP16）加速
        # 注意：暂时禁用FP16，因为可能与voice_tensor不兼容
        # 如果需要启用，需要确保voice_tensor也正确转换为FP16
        self.use_fp16 = False
        # if self.tts_device == "cuda" and torch.cuda.is_available():
        #     try:
        #         self.tts_model = self.tts_model.half()
        #         self.use_fp16 = True
        #         self.logger.info("TTS模型已切换到FP16半精度模式以加速推理")
        #     except Exception as e:
        #         self.logger.warning("无法切换到FP16模式: %s，继续使用FP32", e)
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
        """
        加载 bge-m3 多模态模型（文本 + 图片）。
        """
        self.logger.info("正在加载多模态嵌入模型 bge-m3...")
        self.logger.info("基础模型路径: %s", self.base_model_path)
        self.logger.info("视觉权重路径: %s", self.visual_weight_path)
        try:
            model = Visualized_BGE(
                model_name_bge=self.base_model_path,
                model_weight=self.visual_weight_path,
                normlized=True,
                sentence_pooling_method="cls",
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Visualized_BGE 本身是 nn.Module，直接迁移到目标设备
            model.to(device)
            self.logger.info("✓ 多模态嵌入模型加载成功！")
            return VisualizedBGEEmbeddings(model)
        except Exception as e:
            self.logger.exception("加载多模态嵌入模型失败: %s", e)
            raise

    def _get_combined_vectorstore_dir(self) -> str:
        # 与新模型绑定，避免与旧向量库冲突
        return os.path.join(self.base_dir, "vectorstores", "combined_kb_bge_m3_visualized")

    def _init_vectorstore(self):
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

        # 向量库不存在时，从 uploads 目录构建
        documents_dir = os.path.join(self.base_dir, "uploads", "documents")
        images_dir = os.path.join(self.base_dir, "uploads", "images")
        self.logger.info("首次启动，从 uploads 目录构建向量库...")
        self.logger.info("文档目录: %s", documents_dir)
        self.logger.info("图片目录: %s", images_dir)
        self.rebuild_vectorstore(documents_dir, images_dir)

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
        重建向量库，扫描文档目录和图片目录（文本+图片一起向量化）
        """
        self.logger.info("正在重建向量库...")
        self.vectorstore = None
        self.retriever = None

        all_docs: List[Document] = []

        # 扫描文档目录
        if os.path.exists(documents_dir):
            for filename in os.listdir(documents_dir):
                file_path = os.path.join(documents_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        docs = self._load_docs(file_path)
                        chunks = self._split_docs(docs)
                        all_docs.extend(chunks)
                        self.logger.info("已加载并切分文档: %s", filename)
                    except Exception as e:
                        self.logger.error("处理文档失败 %s: %s", filename, e)

        # 扫描图片目录
        if images_dir and os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                file_path = os.path.join(images_dir, filename)
                if os.path.isfile(file_path):
                    # 图片以 image:// 前缀保存，便于在嵌入时识别
                    doc = Document(page_content=f"image://{file_path}", metadata={"source": file_path, "type": "image"})
                    all_docs.append(doc)
                    self.logger.info("已添加图片: %s", filename)

        if not all_docs:
            self.logger.info("没有找到任何文档或图片，向量库为空")
            vs_dir = self._get_combined_vectorstore_dir()
            if os.path.exists(vs_dir):
                shutil.rmtree(vs_dir)
            return

        # 批量向量化并创建向量库
        self.logger.info("开始批量向量化 %d 个项目（文本+图片）...", len(all_docs))
        self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # 保存
        vs_dir = self._get_combined_vectorstore_dir()
        os.makedirs(vs_dir, exist_ok=True)
        self.vectorstore.save_local(vs_dir)
        self.logger.info("✓ 向量库重建完成并保存到: %s", vs_dir)

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
        """添加图片到知识库并向量化"""
        self.logger.info("正在处理图片: %s", image_path)
        try:
            doc = Document(page_content=f"image://{image_path}", metadata={"source": image_path, "type": "image"})

            if self.vectorstore is None:
                self.logger.info("创建新向量库...")
                self.vectorstore = FAISS.from_documents([doc], self.embeddings)
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            else:
                self.logger.info("添加到现有向量库...")
                self.vectorstore.add_documents([doc])

            vs_dir = self._get_combined_vectorstore_dir()
            os.makedirs(vs_dir, exist_ok=True)
            self.vectorstore.save_local(vs_dir)
            self.logger.info("图片向量已添加并保存到: %s", vs_dir)
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
        
        # 校验：文字和图片不能同时为空
        if not question.strip() and not image_path:
            return "问题不能为空。", timings, []
        
        if not self.retriever:
            return "知识库为空，请先上传文档。", timings, []
        
        self.logger.info("用户问题: %s", question if question.strip() else "[无文字]") 
        try:
            t0 = time.time()
            query_vector = None
            # 根据输入的模态组合选择编码策略
            has_text = bool(question.strip())
            has_image = bool(image_path)
            if has_image and not os.path.exists(image_path):
                return f"错误：图片路径不存在 {image_path}", timings, []

            try:
                if has_image and has_text:
                    # 文本+图片联合向量
                    query_vector = self.embeddings.embed_multimodal(image_path, question)
                    # 提示词保留原问题
                elif has_image:
                    query_vector = self.embeddings.embed_image(image_path)
                    # 如果没有文字，补一个默认问题
                    if not has_text:
                        question = "请比较用户图片与知识库检索到的文本/图片的相似与不同，并描述用户图片。"
                else:
                    # 纯文本
                    docs_retrieved = self.retriever.invoke(question)
            except Exception as e:
                self.logger.warning("查询编码失败，将回退到单模态：%s", e)
                # 回退策略：如果两模态失败，尝试图片；否则文本
                if has_image:
                    query_vector = self.embeddings.embed_image(image_path)
                    if not has_text:
                        question = "描述这张图片的内容。"
                else:
                    docs_retrieved = self.retriever.invoke(question)

            if query_vector is not None:
                docs_retrieved = self.vectorstore.similarity_search_by_vector(query_vector, k=3)

            timings["retrieval"] = time.time() - t0
            def _fmt_content(doc: Document):
                if isinstance(doc.page_content, str) and doc.page_content.startswith("image://"):
                    return f"图片路径: {doc.page_content[len('image://'):]}"
                return doc.page_content

            context = "\n\n".join(_fmt_content(d) for d in docs_retrieved)

            # 收集命中图片，准备多模态输入
            kb_image_paths = [
                doc.metadata.get("source")
                for doc in docs_retrieved
                if isinstance(doc.page_content, str)
                and doc.page_content.startswith("image://")
                and doc.metadata.get("type") == "image"
                and doc.metadata.get("source")
            ]

            # 用户查询携带的图片（对话图片）也参与比较
            query_images = []
            if has_image:
                query_images.append(image_path)

            answer = ""
            sources = [
                {
                    "content": _fmt_content(doc),
                    "source": os.path.basename(doc.metadata.get("source", "未知来源")),
                    "type": doc.metadata.get("type", "text"),
                }
                for doc in docs_retrieved
            ]

            # 只有在有图可用且模型声明支持 vision 时走多模态
            use_vision = bool(kb_image_paths or query_images) and self.llm_config[self.current_model_type].get("supports_vision", False)

            if use_vision:
                try:
                    vision_payload = self._build_vision_payload(question, context, kb_image_paths, query_images)
                    self.logger.info("正在生成回答（图文）...")
                    t0 = time.time()
                    answer = self._invoke_vision(vision_payload)
                    timings["llm_generation"] = time.time() - t0
                    self.logger.info("回答生成完成（图文）")
                except Exception as e:
                    self.logger.warning("多模态调用失败，将回退文本：%s", e)
                    use_vision = False

            if not use_vision:
                prompt = (
                    "你是一个中文助理，请严格依据下面提供的知识库内容回答用户问题，"
                    "如果知识库中没有相关信息，就说不知道，不要编造，也不要扩展。\n\n"
                    f"【知识库内容】:\n{context}\n\n"
                    f"【用户问题】:\n{question}\n\n"
                    "请用简体中文回答："
                )
                self.logger.info("正在生成回答（文本）...")
                t0 = time.time()
                response = self.llm.invoke(prompt)
                timings["llm_generation"] = time.time() - t0
                answer = response.content
                self.logger.info("回答生成完成（文本）")

            return answer, timings, sources
        except Exception as e:
            self.logger.error("RAG 查询失败: %s", e)
            return f"抱歉，生成回答时出现错误: {str(e)}", timings, []

    def _split_text_for_tts(self, text: str, max_length: int = 150) -> List[str]:
        """将文本按句子分割，确保每段不超过最大长度
        
        注意：TTS pipeline可能有音频时长限制（约29秒）
        根据经验，150字符大约对应20-25秒的音频，留有余量避免超时
        使用最简单可靠的方法，确保不丢失任何文本
        """
        import re
        
        # 保存原始文本用于验证
        original_text = text
        
        # 使用最简单的方法：直接按字符遍历，在句子边界处优先分割
        segments = []
        current = ""
        i = 0
        
        while i < len(text):
            # 获取当前位置到max_length的文本
            remaining = text[i:]
            
            # 如果剩余文本不超过max_length，直接添加
            if len(current + remaining) <= max_length:
                if current:
                    current = (current + remaining).strip()
                else:
                    current = remaining.strip()
                break
            
            # 查找最佳分割点（在max_length范围内）
            search_end = min(i + max_length, len(text))
            search_text = text[i:search_end]
            
            # 优先在句号、感叹号、问号处分割
            best_pos = -1
            for pos in range(len(search_text) - 1, -1, -1):
                if search_text[pos] in ['。', '！', '？']:
                    best_pos = i + pos + 1
                    break
            
            # 如果没找到句号等，尝试在逗号、顿号处分割
            if best_pos == -1:
                for pos in range(len(search_text) - 1, -1, -1):
                    if search_text[pos] in ['，', '、']:
                        best_pos = i + pos + 1
                        break
            
            # 如果还是没找到，或者当前段加上这部分不超过限制，使用max_length
            if best_pos == -1:
                best_pos = i + max_length
            
            # 获取这段文本
            chunk = text[i:best_pos]
            
            # 检查是否可以添加到当前段
            if current:
                test = (current + chunk).strip()
            else:
                test = chunk.strip()
            
            if len(test) <= max_length:
                # 可以合并
                current = test
            else:
                # 不能合并，先保存当前段
                if current:
                    segments.append(current)
                # 如果chunk本身超过限制，需要进一步分割
                if len(chunk.strip()) > max_length:
                    # 按字符强制分割
                    for j in range(0, len(chunk), max_length):
                        sub_chunk = chunk[j:j + max_length].strip()
                        if sub_chunk:
                            segments.append(sub_chunk)
                    current = ""
                else:
                    current = chunk.strip()
            
            i = best_pos
        
        # 添加最后一段
        if current:
            segments.append(current)
        
        # 验证：确保所有文本都被包含（移除空白字符比较）
        original_clean = re.sub(r'\s+', '', original_text)
        segments_clean = ''.join(re.sub(r'\s+', '', s) for s in segments)
        
        if segments_clean != original_clean:
            self.logger.error("分段丢失文本！原文长度: %d, 分段后长度: %d", len(original_clean), len(segments_clean))
            missing = len(original_clean) - len(segments_clean)
            self.logger.error("丢失 %d 个字符", missing)
            
            # 找出丢失的文本
            missing_text = ""
            orig_pos = 0
            seg_pos = 0
            orig_clean_list = list(original_clean)
            seg_clean_list = list(segments_clean)
            
            while orig_pos < len(orig_clean_list) and seg_pos < len(seg_clean_list):
                if orig_clean_list[orig_pos] == seg_clean_list[seg_pos]:
                    orig_pos += 1
                    seg_pos += 1
                else:
                    missing_text += orig_clean_list[orig_pos]
                    orig_pos += 1
            
            if orig_pos < len(orig_clean_list):
                missing_text += ''.join(orig_clean_list[orig_pos:])
            
            if missing_text:
                self.logger.error("丢失的文本: %s", missing_text[:200])
            
            self.logger.error("原文前200字符: %s", original_text[:200])
            self.logger.error("分段结果数量: %d", len(segments))
            for idx, seg in enumerate(segments):
                self.logger.error("分段[%d] (%d字符): %s", idx, len(seg), seg[:100])
            
            # 使用备用方法：按字符强制分割，确保不丢失
            self.logger.warning("使用备用分割方法（按字符强制分割）")
            segments = []
            pos = 0
            while pos < len(original_text):
                # 尝试在句子边界处分割
                end_pos = min(pos + max_length, len(original_text))
                if end_pos < len(original_text):
                    # 尝试向后查找句号、感叹号、问号（最多50字符）
                    for look_ahead in range(min(50, len(original_text) - end_pos)):
                        if original_text[end_pos + look_ahead] in ['。', '！', '？']:
                            end_pos = end_pos + look_ahead + 1
                            break
                
                chunk = original_text[pos:end_pos].strip()
                if chunk:
                    segments.append(chunk)
                pos = end_pos
            
            # 再次验证
            segments_clean_backup = ''.join(re.sub(r'\s+', '', s) for s in segments)
            if segments_clean_backup != original_clean:
                self.logger.error("备用方法也丢失文本！原文 %d, 备用 %d", len(original_clean), len(segments_clean_backup))
                # 如果备用方法也失败，使用最简单的字符分割
                self.logger.error("使用最简单的字符分割方法")
                segments = []
                for i in range(0, len(original_text), max_length):
                    chunk = original_text[i:i + max_length].strip()
                    if chunk:
                        segments.append(chunk)
            else:
                self.logger.info("备用方法成功，所有文本都被保留")
        
        return segments if segments else [text]

    # --- 多模态生成 ---
    def _build_vision_payload(self, question: str, context: str, kb_image_paths: List[str], query_images: List[str]) -> Dict[str, Any]:
        """
        将文本与图片封装为 OpenAI 风格的多模态消息，适配本地 llama.cpp server。
        """
        # 限制图片数量与大小（简单限制）
        max_imgs = 3
        items = []

        def add_image_with_label(label: str, path: str):
            try:
                p = Path(path)
                if not p.exists() or not p.is_file():
                    self.logger.warning("图片不存在，跳过: %s", path)
                    return
                data = p.read_bytes()
                if len(data) > 5 * 1024 * 1024:  # 5MB 限制
                    self.logger.warning("图片过大(>5MB)，跳过: %s", path)
                    return
                b64 = base64.b64encode(data).decode("utf-8")
                mime = "image/png"
                if path.lower().endswith((".jpg", ".jpeg")):
                    mime = "image/jpeg"
                elif path.lower().endswith(".webp"):
                    mime = "image/webp"
                items.append({"type": "text", "text": label})
                items.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            except Exception as e:
                self.logger.warning("读取图片失败，跳过 %s，原因: %s", path, e)

        # 先放查询图片，再放 KB 图片，并带标签
        count = 0
        if query_images:
            for i, path in enumerate(query_images, 1):
                if count >= max_imgs:
                    break
                add_image_with_label(f"【用户图片{i}】", path)
                count += 1
        if kb_image_paths and count < max_imgs:
            for i, path in enumerate(kb_image_paths, 1):
                if count >= max_imgs:
                    break
                add_image_with_label(f"【知识库图片{i}】", path)
                count += 1

        # 根据是否有用户文本，调整引导
        if query_images and not context.strip():
            kb_hint = "（知识库未命中文本）"
        else:
            kb_hint = ""

        text_parts = [
            {
                "type": "text",
                "text": (
                    "你是一个中文助理，请结合提供的知识库文本和图片，以及用户提供的图片，回答用户问题。"
                    "如果缺少相关信息，就直接回答不知道。请尽量对比用户图片与知识库图片的相似与不同。"
                ),
            },
            {
                "type": "text",
                "text": f"【知识库文本】{kb_hint}:\n{context}" if context else "【知识库文本】：无",
            },
            {"type": "text", "text": f"【用户问题】:\n{question}"},
        ]

        # 将文本与图片一起作为 user 消息内容
        user_content = text_parts + items

        config = self.llm_config[self.current_model_type]
        payload = {
            "model": config["model_name"],
            "messages": [
                {"role": "system", "content": "你是一个仅依据给定内容回答的中文助手。"},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.01,
            "max_tokens": 512,
        }
        return payload

    def _invoke_vision(self, payload: Dict[str, Any]) -> str:
        """
        调用本地 llama.cpp 兼容的 /v1/chat/completions 接口，支持图文。
        """
        config = self.llm_config[self.current_model_type]
        # 清理掉误入的 fragment（可能带 #）
        api_base = config["api_base"].split("#")[0].rstrip("/")
        api_key = config.get("api_key") or ""
        url = f"{api_base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with httpx.Client(timeout=120) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI 格式
            choices = data.get("choices")
            if not choices:
                raise ValueError("响应无 choices")
            message = choices[0].get("message", {})
            content = message.get("content")
            if not content:
                raise ValueError("响应无 content")
            if isinstance(content, list):
                # content 也可能是多段结构，拼接文本部分
                content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
            return content

    def text_to_speech(self, text: str):
        try:
            import re
            # 移除 markdown 格式标记
            clean_text = text.replace("**", "").replace("*", "").strip()
            
            # 处理换行符：将换行符替换为合适的标点，确保文本连续
            # 1. 如果换行符前是句号、感叹号、问号，直接删除换行（已经是句子结束）
            clean_text = re.sub(r'([。！？])\s*\n+', r'\1', clean_text)
            # 2. 如果换行符前是冒号，直接删除换行（冒号本身有停顿效果）
            clean_text = re.sub(r'：\s*\n+', '：', clean_text)
            # 3. 其他所有换行符（包括多个连续换行）替换为逗号，确保文本连续
            clean_text = re.sub(r'\n+', '，', clean_text)
            
            # 清理多余的标点和空格
            clean_text = re.sub(r'，+', '，', clean_text)  # 多个连续逗号合并为一个
            clean_text = re.sub(r' +', ' ', clean_text)  # 多个连续空格合并为一个
            clean_text = re.sub(r'^，+', '', clean_text)  # 去除句首的逗号
            clean_text = re.sub(r'，\s*，', '，', clean_text)  # 逗号后紧跟逗号的情况
            
            # 再次去除首尾空白
            clean_text = clean_text.strip()
            if not clean_text:
                self.logger.warning("清理后的文本为空，无法生成语音")
                return None
            
            # 将文本分段处理，避免长度限制
            # 在句号、感叹号、问号后添加空格，确保TTS能识别句子边界
            clean_text = re.sub(r'([。！？])([^\s。！？])', r'\1 \2', clean_text)
            # 增加分段长度到150字符以减少分段次数，提升性能
            # 根据经验，150字符大约对应20-25秒的音频，仍在安全范围内
            text_length = len(clean_text)
            segments = self._split_text_for_tts(clean_text, max_length=150)
            if len(segments) > 1:
                self.logger.info("文本已分为 %d 段进行TTS处理（长度: %d 字符）", len(segments), text_length)
            
            # 生成每段的音频并合并
            audio_segments = []
            sample_rate = 24000
            # 句号后的停顿时长（秒）
            sentence_pause_duration = 0.4
            # 段落间的停顿时长（秒）- 减少段与段之间的停顿
            paragraph_pause_duration = 0.2
            
            # 使用torch.inference_mode()加速推理
            with torch.inference_mode():
                for i, segment in enumerate(segments):
                    try:
                        # 直接使用原始的voice_tensor，让pipeline自己处理设备转换
                        # 避免破坏voice_tensor的复杂结构
                        generator = self.tts_pipeline(segment, voice=self.voice_tensor, speed=self.speed_callable)
                        # 尝试获取所有生成的音频块
                        segment_audio_parts = []
                        try:
                            while True:
                                result = next(generator)
                                wav = result.audio
                                # 将 torch tensor 转换为 numpy array
                                if torch.is_tensor(wav):
                                    wav = wav.cpu().numpy()
                                # 确保是一维数组
                                if wav.ndim > 1:
                                    wav = wav.flatten()
                                # 确保数据类型为 float32
                                wav = wav.astype(np.float32)
                                segment_audio_parts.append(wav)
                        except StopIteration:
                            # 正常结束
                            pass
                        
                        if segment_audio_parts:
                            # 合并该段的所有音频块
                            segment_audio = np.concatenate(segment_audio_parts)
                            audio_segments.append(segment_audio)
                        else:
                            self.logger.warning("第 %d 段未生成任何音频", i + 1)
                    except Exception as e:
                        self.logger.error("生成第 %d 段语音失败: %s", i + 1, e)
                        # 继续处理其他段
                        continue
            
            if not audio_segments:
                self.logger.error("所有音频段生成失败")
                return None
            
            # 简化音频拼接：直接合并，减少复杂的后处理以提升性能
            if len(audio_segments) == 1:
                combined_audio = audio_segments[0]
            else:
                # 简单拼接，不做复杂的交叉淡入淡出处理以提升性能
                combined_audio = np.concatenate(audio_segments)
            
            # 转换为字节流
            wav_io = io.BytesIO()
            sf.write(wav_io, combined_audio, sample_rate, format="WAV")
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


