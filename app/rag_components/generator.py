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

    def _invoke_llm_for_string(self, prompt: str) -> str:
        """
        用于简单文本生成的轻量级 LLM 调用。
        """
        config = self.llm_config[self.current_model_type]
        api_base = config["api_base"].split("#")[0].rstrip("/")
        api_key = config.get("api_key", "")
        url = f"{api_base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": config["model_name"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 50  # 限制 token 数量，因为只需要几个词的回答
        }

        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def classify_intent(self, question: str, has_image: bool) -> str:
        """
        判断用户意图，决定使用哪个处理流程。
        """
        logger.info("正在判断用户意图...")
        
        context_string = "用户已上传一张图片。" if has_image else "用户未上传图片。"
        
        prompt = f"""
你是一个煤矿安全领域的智能任务调度员。你的职责是根据用户的请求和上下文，判断应该调用哪个处理模块：“图像安全分析”或“文本知识问答”。

- “图像安全分析”模块：专门处理对图片中的设备、场景或人员进行安全状态的分析、检测、检查或评估。只有在用户上传了图片时才能使用此模块。
- “文本知识问答”模块：处理所有不涉及具体图像分析的请求，包括但不限于：询问安全规程、操作流程、通用知识，以及闲聊、讲故事等。

---
上下文：{context_string}
---

用户的请求是：“{question}”

请根据用户的请求和上下文，判断应该调用哪个模块。请严格只回答“图像安全分析”或“文本知识问答”。
"""
        try:
            response = self._invoke_llm_for_string(prompt)
            
            if "图像安全分析" in response and has_image:
                logger.info("意图判断结果: 图像安全分析")
                return "vision_analysis"
            else:
                logger.info("意图判断结果: 文本知识问答")
                return "text_qa"
        except Exception as e:
            logger.error(f"意图判断失败: {e}，将默认使用文本问答流程。")
            return "text_qa"

    def switch_llm(self, model_type: str):
        """切换 LLM 模型"""
        if model_type not in self.llm_config:
            raise ValueError(f"不支持的模型类型: {model_type}")
        logger.info("切换 LLM 到: %s (%s)", model_type, self.llm_config[model_type]["model_name"])
        self.llm = self._create_llm_instance(model_type)
        self.current_model_type = model_type

    def generate_answer(self, question: str, context_docs: List[Document], query_image_path: str = None, is_vision_request: bool = False) -> str:
        """根据上下文和问题生成答案，自动处理多模态"""
        context = "\n\n".join(self._format_doc_content(d) for d in context_docs)

        if is_vision_request and query_image_path:
            try:
                kb_images_with_titles = [
                    (doc.metadata.get("source"), doc.metadata.get("title", ""))
                    for doc in context_docs
                    if isinstance(doc.page_content, str)
                    and doc.page_content.startswith("image://")
                    and doc.metadata.get("type") == "image"
                    and doc.metadata.get("source")
                ]
                query_images = [query_image_path]
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
   - **如果**【用户上传图片】与某张【标准参考图片】内容**高度相似或完全相同**，请直接采纳该标准图片的标题作为主要分析结论，并围绕该结论进行详细阐述。
   - **如果**找到了相关的【标准参考图片】但内容不完全相同，请从以下多个维度进行综合比对和分析，找出【用户上传图片】中的异常点或不规范之处：
     *   **结构完整性**: 检查设备部件是否有破损、裂纹、严重磨损或缺失（例如传送带表面是否完好）。
     *   **位置状态**: 检查设备或部件的位置是否正确、对齐（例如传送带是否位于滚筒中心，是否存在跑偏）。
     *   **运行工况**: 检查是否有异物堆积、过载、泄漏等异常运行状态。
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

        user_content = text_parts + items

        config = self.llm_config[self.current_model_type]
        return {
            "model": config["model_name"],
            "messages": [
                {"role": "system", "content": system_prompt},
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
