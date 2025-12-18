"""
嵌入模块：负责加载和使用多模态嵌入模型。
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from PIL import Image

# 动态导入 visual_bge
try:
    from visual_bge.modeling import Visualized_BGE
except ImportError:
    raise ImportError("请确保 visual_bge 已安装或在 Python 路径中。")


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

        if any(v is None or len(v) == 0 for v in vectors):
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


def create_embedding_model(base_model_path: str, visual_weight_path: str) -> VisualizedBGEEmbeddings:
    """
    加载 bge-m3 多模态模型（文本 + 图片）。
    """
    logger = logging.getLogger(__name__)
    logger.info("正在加载多模态嵌入模型 bge-m3...")
    logger.info("基础模型路径: %s", base_model_path)
    logger.info("视觉权重路径: %s", visual_weight_path)
    try:
        model = Visualized_BGE(
            model_name_bge=base_model_path,
            model_weight=visual_weight_path,
            normlized=True,
            sentence_pooling_method="cls",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Visualized_BGE 本身是 nn.Module，直接迁移到目标设备
        model.to(device)
        logger.info("✓ 多模态嵌入模型加载成功！")
        return VisualizedBGEEmbeddings(model)
    except Exception as e:
        logger.exception("加载多模态嵌入模型失败: %s", e)
        raise
