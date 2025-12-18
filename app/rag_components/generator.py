"""
生成模块：负责调用 LLM 生成最终答案。
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List

import httpx
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """封装 LLM 调用，支持多模态输入"""

    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.current_model_type = "local"  # 默认
        self.llm = self._create_llm_instance("local")

    def _create_llm_instance(self, model_type: str) -> ChatOpenAI:
        config = self.llm_config[model_type]
        return ChatOpenAI(
            openai_api_base=config["api_base"],
            openai_api_key=config["api_key"],
            model_name=config["model_name"],
            temperature=0.01,
            max_tokens=512,
        )

    def switch_llm(self, model_type: str):
        """切换 LLM 模型"""
        if model_type not in self.llm_config:
            raise ValueError(f"不支持的模型类型: {model_type}")
        logger.info("切换 LLM 到: %s (%s)", model_type, self.llm_config[model_type]["model_name"])
        self.llm = self._create_llm_instance(model_type)
        self.current_model_type = model_type

    def generate_answer(self, question: str, context_docs: List[Document], query_image_path: str = None) -> str:
        """根据上下文和问题生成答案，自动处理多模态"""
        context = "\n\n".join(self._format_doc_content(d) for d in context_docs)

        kb_image_paths = [
            doc.metadata.get("source")
            for doc in context_docs
            if isinstance(doc.page_content, str)
            and doc.page_content.startswith("image://")
            and doc.metadata.get("type") == "image"
            and doc.metadata.get("source")
        ]

        query_images = [query_image_path] if query_image_path else []

        use_vision = (
            bool(kb_image_paths or query_images) and 
            self.llm_config[self.current_model_type].get("supports_vision", False)
        )

        answer = ""
        if use_vision:
            try:
                vision_payload = self._build_vision_payload(question, context, kb_image_paths, query_images)
                logger.info("正在生成回答（图文）...")
                answer = self._invoke_vision(vision_payload)
                logger.info("回答生成完成（图文）")
                return answer
            except Exception as e:
                logger.warning("多模态调用失败，将回退到纯文本模式：%s", e)
        
        # 纯文本模式
        prompt = (
            "你是一个中文助理，请严格依据下面提供的知识库内容回答用户问题，"
            "如果知识库中没有相关信息，就说不知道，不要编造，也不要扩展。\n\n"
            f"【知识库内容】:\n{context}\n\n"
            f"【用户问题】:\n{question}\n\n"
            "请用简体中文回答："
        )
        logger.info("正在生成回答（文本）...")
        response = self.llm.invoke(prompt)
        answer = response.content
        logger.info("回答生成完成（文本）")
        return answer

    def _format_doc_content(self, doc: Document) -> str:
        if isinstance(doc.page_content, str) and doc.page_content.startswith("image://"):
            return f"图片路径: {doc.page_content[len('image://'):]}"
        return doc.page_content

    def _build_vision_payload(self, question: str, context: str, kb_image_paths: List[str], query_images: List[str]) -> Dict[str, Any]:
        """
        将文本与图片封装为 OpenAI 风格的多模态消息，适配本地 llama.cpp server。
        """
        max_imgs = 3
        items = []

        def add_image_with_label(label: str, path: str):
            try:
                p = Path(path)
                if not p.exists() or not p.is_file():
                    logger.warning("图片不存在，跳过: %s", path)
                    return
                data = p.read_bytes()
                if len(data) > 5 * 1024 * 1024:  # 5MB 限制
                    logger.warning("图片过大(>5MB)，跳过: %s", path)
                    return
                b64 = base64.b64encode(data).decode("utf-8")
                mime = "image/png"
                if path.lower().endswith( (".jpg", ".jpeg")):
                    mime = "image/jpeg"
                elif path.lower().endswith(".webp"):
                    mime = "image/webp"
                items.append({"type": "text", "text": label})
                items.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            except Exception as e:
                logger.warning("读取图片失败，跳过 %s，原因: %s", path, e)

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

        user_content = text_parts + items

        config = self.llm_config[self.current_model_type]
        return {
            "model": config["model_name"],
            "messages": [
                {"role": "system", "content": "你是一个仅依据给定内容回答的中文助手。"},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.01,
            "max_tokens": 512,
        }

    def _invoke_vision(self, payload: Dict[str, Any]) -> str:
        """调用兼容 OpenAI 的多模态接口"""
        config = self.llm_config[self.current_model_type]
        api_base = config["api_base"].split("#")[0].rstrip("/")
        api_key = config.get("api_key", "")
        url = f"{api_base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with httpx.Client(timeout=120) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("API 响应中没有 choices")
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join([c.get("text", "") for c in content if isinstance(c, dict)])
            return content
