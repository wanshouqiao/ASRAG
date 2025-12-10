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
            text_vecs = self._encode_text_batch(list(text_batch))
            for i, v in zip(idxs, text_vecs):
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
        kb_path: str,
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


