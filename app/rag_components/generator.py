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

        kb_images_with_titles = [
            (doc.metadata.get("source"), doc.metadata.get("title", ""))
            for doc in context_docs
            if isinstance(doc.page_content, str)
            and doc.page_content.startswith("image://")
            and doc.metadata.get("type") == "image"
            and doc.metadata.get("source")
        ]

        query_images = [query_image_path] if query_image_path else []

        use_vision = (
            bool(kb_images_with_titles or query_images) and 
            self.llm_config[self.current_model_type].get("supports_vision", False)
        )

        answer = ""
        if use_vision:
            try:
                vision_payload = self._build_vision_payload(question, context, kb_images_with_titles, query_images)
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

    def _build_vision_payload(self, question: str, context: str, kb_images_with_titles: List[tuple[str, str]], query_images: List[str]) -> Dict[str, Any]:
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
                add_image_with_label(f"【用户上传图片{i}】", path)
                count += 1
        if kb_images_with_titles and count < max_imgs:
            for i, (path, title) in enumerate(kb_images_with_titles, 1):
                if count >= max_imgs:
                    break
                # 使用标题来丰富图片描述
                label = f"【标准参考图片{i}: {title}】" if title else f"【标准参考图片{i}】"
                add_image_with_label(label, path)
                count += 1

        # --- 煤矿安全场景下的多模态提示词 ---
        system_prompt = (
            "你是一位资深的煤矿安全智能检测员。"
            "你的任务是分析用户上传的现场图片，并依据知识库中存储的标准操作图片，来判断现场是否存在安全隐患或操作不规范之处。"
        )

        # 根据有无用户问题，动态构建主提示
        if question.strip():
            user_question_prompt = f"结合用户提出的问题：“{question}”，请进行分析。"
        else:
            user_question_prompt = "请直接对图片内容进行异常检测。"

        main_prompt = f"""
请严格遵循以下步骤进行分析：

1. **场景识别**: 
   首先，请仔细观察【用户上传图片】，识别并简要描述图片中的主要作业场景、设备或人员活动。

2. **相关性判断**: 
   接下来，请逐一对比【用户上传图片】和提供给你的每一张【标准参考图片】。判断哪些【标准参考图片】与【用户上传图片】展示的是**相同的设备或作业场景**。

3. **异常检测与分析**:
   - **如果**你找到了相关的【标准参考图片】：
     请将【用户上传图片】与这些**相关的**【标准参考图片】进行详细比对。结合【相关文本信息】（如果有），清晰、具体地指出用户图片中存在的任何差异点，并判断这些差异是否构成安全隐患或操作不规范。请给出专业的分析和建议。
   - **如果**所有【标准参考图片】都与用户图片的场景不相关：
     请明确告知：“抱歉，知识库中未找到可供对比的相同场景的标准图片，无法进行有效的异常检测。”
   - **如果**没有提供【标准参考图片】：
     请基于你的通用知识，判断【用户上传图片】中是否存在明显的安全隐患。

{user_question_prompt}
"""

        text_parts = [
            {"type": "text", "text": main_prompt},
            {"type": "text", "text": f"【相关文本信息】:\n{context}" if context.strip() else "【相关文本信息】: 无"},
        ]

        # 将 user_content 的构建移到这里，并更新 system message
        user_content = text_parts + items

        config = self.llm_config[self.current_model_type]
        return {
            "model": config["model_name"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.01,
            "max_tokens": 512,  # 可根据需要调整
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
