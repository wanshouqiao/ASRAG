"""
生成模块：负责调用 LLM 生成最终答案。
"""

import base64
import logging
import os
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
            temperature=0.00,
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

    def test_model_connection(self, model_type: str) -> bool:
        """测试模型连接是否可用"""
        if model_type not in self.llm_config:
            return False
        try:
            config = self.llm_config[model_type]
            api_base = config["api_base"].split("#")[0].rstrip("/")
            api_key = config.get("api_key", "")
            url = f"{api_base}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # 发送一个简单的测试请求
            payload = {
                "model": config["model_name"],
                "messages": [
                    {"role": "user", "content": "test"}
                ],
                "temperature": 0.0,
                "max_tokens": 5
            }
            
            with httpx.Client(timeout=10) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                # 检查响应格式是否正确
                if "choices" in data and len(data["choices"]) > 0:
                    logger.info("模型连接测试成功: %s", model_type)
                    return True
                else:
                    logger.warning("模型响应格式异常: %s", model_type)
                    return False
        except httpx.TimeoutException:
            logger.error("模型连接超时: %s", model_type)
            return False
        except httpx.HTTPStatusError as e:
            logger.error("模型连接失败 (HTTP %d): %s", e.response.status_code, model_type)
            return False
        except Exception as e:
            logger.error("模型连接测试异常: %s - %s", model_type, e)
            return False

    def switch_llm(self, model_type: str):
        """切换 LLM 模型，切换前会测试连接"""
        if model_type not in self.llm_config:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 测试模型连接
        if not self.test_model_connection(model_type):
            raise ConnectionError(f"模型 '{model_type}' 连接失败，无法切换。请检查模型服务是否正在运行。")
        
        logger.info("切换 LLM 到: %s (%s)", model_type, self.llm_config[model_type]["model_name"])
        self.llm = self._create_llm_instance(model_type)
        self.current_model_type = model_type

    def add_model(self, model_id: str, api_base: str, api_key: str, model_name: str, supports_vision: bool = False, test_connection: bool = True):
        """添加新模型到配置，默认会测试连接"""
        if model_id in self.llm_config:
            raise ValueError(f"模型 ID '{model_id}' 已存在")
        
        # 先添加配置（临时）
        self.llm_config[model_id] = {
            "api_base": api_base,
            "api_key": api_key,
            "model_name": model_name,
            "supports_vision": supports_vision,
        }
        
        # 如果需要测试连接
        if test_connection:
            if not self.test_model_connection(model_id):
                # 连接失败，移除刚添加的配置
                del self.llm_config[model_id]
                raise ConnectionError(f"模型 '{model_id}' 连接失败，无法添加。请检查 API 地址和模型服务是否正在运行。")
        
        logger.info("已添加新模型: %s (%s)", model_id, model_name)

    def remove_model(self, model_id: str):
        """从配置中删除模型"""
        if model_id not in self.llm_config:
            raise ValueError(f"模型 ID '{model_id}' 不存在")
        if model_id == "local":
            raise ValueError("不能删除默认的 'local' 模型")
        # 如果当前使用的是要删除的模型，需要先切换
        if self.current_model_type == model_id:
            raise ValueError(f"不能删除当前正在使用的模型 '{model_id}'，请先切换到其他模型")
        del self.llm_config[model_id]
        logger.info("已删除模型: %s", model_id)

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
            "你是一个以煤矿安全为核心领域的智能知识库助手。你的首要任务是利用【知识库内容】来回答【用户问题】。\n\n"
            "请遵循以下指南：\n"
            "1. **综合回答**: 根据【知识库内容】，为【用户问题】提供一个全面而简洁的回答。如果用户只提供了一个主题（例如“煤矿安全”），你的任务就是对知识库中关于该主题的信息进行总结。\n"
            "2. **忠实原文**: 你的回答必须严格基于【知识库内容】，不要添加外部知识或进行编造。\n"
            "3. **处理不相关**: 如果【用户问题】的与知识库检索到的内容严格不相关，请直接回答：“抱歉，知识库中没有找到关于‘{question}’的相关信息。”\n\n"
            "---\n"
            f"【知识库内容】:\n{context}\n\n"
            f"【用户问题】:\n{question}\n\n"
            "---\n\n" 
            "请生成你的回答："
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
            "你是一个多功能的煤矿安全智能助手。"
            "你的主要职责是利用提供的图片和知识库信息回答用户问题。"
            "你能够执行专业的安全分析和图片分析，你也可以基于用户的问题进行对应的图文分析。"
        )

        # 统一将用户问题传递给模型，并进行简化
        if question.strip():
            user_question_prompt = f"请特别注意，用户提出了一个具体问题需要回答：‘{question}’"
        else:
            user_question_prompt = "用户的目标是进行通用的安全分析。"

        main_prompt = f"""
请严格遵循【思考】和【回答】两个阶段来完成任务。

---
### **【第一阶段：思考】**
（你必须在内部完成这些思考步骤，但不要在最终回答中展示此部分）

**1. 核心任务判断**:
首先，综合分析【用户问题】({user_question_prompt})、用户上传的图片以及知识库提供的参考图片，判断你的核心任务属于以下哪一类：
- **任务A (独立分析)**: 当用户的问题并非进行安全分析（例如，询问图片内容），或者用户上传的图片与知识库的参考图片完全不相关时（例如，用户上传风景照，而知识库是工业设备）。
- **任务B (对比分析)**: 当用户的问题是进行安全分析，并且用户上传的图片与知识库的参考图片主题相关时。

**2. 执行分析**:
- **如果判断为任务A**: 请忽略所有参考图片和相关文本。基于你作为“煤矿安全智能助手”的通用知识，直接分析【用户上传图片】并回答用户问题。
- **如果判断为任务B**: 请严格按照以下步骤进行详细的对比分析：
    a. **场景识别**: 识别【用户上传图片】的场景。
    b. **相关性判断**: 确定哪些参考图片是相关的。
    c. **安全状态检测**: 如果图片高度相似，则采纳参考图片的标题作为结论；否则，从结构完整性、位置状态、运行工况等维度进行详细比对，找出异常点。

---
### **【第二阶段：回答】**
（请根据你在第一步中判断的核心任务来组织回答）

- **如果执行了任务A (独立分析)**: 请直接、清晰地输出你对【用户上传图片】的分析结果。
- **如果执行了任务B (对比分析)**: 请严格按照以下报告格式回答：
    **1. 场景简述:**
    **2. 分析与诊断:**
    **3. 结论与建议:**

---
"""
        logger.warning("生成图文回答！")
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
            "temperature": 0.00,
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

    def generate_direct(self, question: str = "", image_path: str = None) -> str:
        """
        直接调用大模型，使用通用提示词
        用于通用模型模式
        """
        config = self.llm_config[self.current_model_type]
        has_text = bool(question.strip())
        has_image = bool(image_path)
        
        if not has_text and not has_image:
            return "请输入文本或图片"
        
        # 系统提示词
        system_prompt = """你是一个AI助手，请用中文回复用户的问题。"""
        
        # 如果有图片，使用多模态接口
        if has_image:
            if not os.path.exists(image_path):
                return f"错误：图片路径不存在 {image_path}"
            
            try:
                # 构建多模态消息
                items = []
                
                # 如果有文本，先添加文本
                if has_text:
                    items.append({"type": "text", "text": question})
                
                # 添加图片
                try:
                    p = Path(image_path)
                    data = p.read_bytes()
                    if len(data) > 10 * 1024 * 1024:  # 10MB 限制
                        return "图片过大(>10MB)"
                    b64 = base64.b64encode(data).decode("utf-8")
                    mime = "image/png"
                    if image_path.lower().endswith((".jpg", ".jpeg")):
                        mime = "image/jpeg"
                    elif image_path.lower().endswith(".webp"):
                        mime = "image/webp"
                    items.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
                except Exception as e:
                    logger.warning("读取图片失败: %s", e)
                    return f"读取图片失败: {e}"
                
                # 构建 payload，content 必须是数组
                user_content = items if len(items) > 0 else [{"type": "text", "text": ""}]
                
                payload = {
                    "model": config["model_name"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048,
                }
                
                logger.info("直接调用大模型（多模态）...")
                answer = self._invoke_vision(payload)
                logger.info("回答生成完成（多模态）")
                return answer
            except Exception as e:
                logger.error("多模态调用失败: %s", e)
                # 如果多模态失败，尝试纯文本
                if has_text:
                    return self._invoke_direct_text(question, system_prompt)
                return f"处理失败: {e}"
        else:
            # 纯文本模式
            return self._invoke_direct_text(question, system_prompt)
    
    def _invoke_direct_text(self, question: str, system_prompt: str = None) -> str:
        """直接调用大模型（纯文本，带系统提示词）"""
        logger.info("直接调用大模型（文本）...")
        
        # 使用 HTTP 直接调用，以便添加 system message
        config = self.llm_config[self.current_model_type]
        api_base = config["api_base"].split("#")[0].rstrip("/")
        api_key = config.get("api_key", "")
        url = f"{api_base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": config["model_name"],
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        
        with httpx.Client(timeout=120) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("API 响应中没有 choices")
            message = choices[0].get("message", {})
            answer = message.get("content", "")
            if isinstance(answer, list):
                answer = "".join([c.get("text", "") for c in answer if isinstance(c, dict)])
        
        logger.info("回答生成完成（文本）")
        return answer
