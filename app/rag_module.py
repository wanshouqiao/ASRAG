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

import numpy as np
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

    def rebuild_vectorstore(self, upload_dir: str):
        self.logger.info("正在重建向量库...")
        self.vectorstore = None
        self.retriever = None
        files = []
        if os.path.exists(upload_dir):
            files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
        if not files:
            self.logger.info("没有文件，向量库为空")
            vs_dir = self._get_combined_vectorstore_dir()
            import shutil

            if os.path.exists(vs_dir):
                shutil.rmtree(vs_dir)
            return

        all_chunks = []
        for filename in files:
            file_path = os.path.join(upload_dir, filename)
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
            self.logger.info("向量库重建完成，共 %d 个文件", len(files))

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

    def query(self, question: str) -> Tuple[str, Dict, List[Dict]]:
        timings = {}
        if not question.strip():
            return "问题不能为空。", timings, []
        if not self.retriever:
            return "知识库为空，请先上传文档。", timings, []
        self.logger.info("用户问题: %s", question)
        try:
            t0 = time.time()
            docs_retrieved = self.retriever.invoke(question)
            timings["retrieval"] = time.time() - t0
            context = "\n\n".join(d.page_content for d in docs_retrieved)
            # prompt = (
            #     "你是一个中文助理，请严格依据下面提供的知识库内容回答用户问题，"
            #     "如果知识库中没有相关信息，就说不知道，不要编造，也不要扩展。\n\n"
            #     f"【知识库内容】:\n{context}\n\n"
            #     f"【用户问题】:\n{question}\n\n"
            #     "请用简体中文回答："
            # )
            prompt = question
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

    def _split_text_for_tts(self, text: str, max_length: int = 100) -> List[str]:
        """将文本按句子分割，确保每段不超过最大长度
        
        注意：TTS pipeline可能有音频时长限制（约29秒），建议使用较小的max_length（100字符）
        根据经验，100字符大约对应15-20秒的音频，留有余量避免超时
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
            # 使用更小的分段长度（100字符），因为TTS pipeline可能有音频时长限制（约29秒）
            # 根据经验，100字符大约对应15-20秒的音频，留有余量
            text_length = len(clean_text)
            self.logger.info("清理后文本长度: %d 字符", text_length)
            segments = self._split_text_for_tts(clean_text, max_length=100)
            self.logger.info("文本已分为 %d 段进行TTS处理", len(segments))
            for i, seg in enumerate(segments):
                self.logger.debug("第 %d 段长度: %d 字符, 内容: %s...", i + 1, len(seg), seg[:50])
            
            # 生成每段的音频并合并
            audio_segments = []
            sample_rate = 24000
            # 句号后的停顿时长（秒）
            sentence_pause_duration = 0.4
            # 段落间的停顿时长（秒）- 减少段与段之间的停顿
            paragraph_pause_duration = 0.2
            
            for i, segment in enumerate(segments):
                self.logger.info("正在生成第 %d/%d 段语音: %s...", i + 1, len(segments), segment[:30])
                self.logger.debug("第 %d 段完整内容: %s", i + 1, segment)
                try:
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
                        segment_duration = len(segment_audio) / sample_rate
                        self.logger.info("第 %d 段生成完成，音频时长: %.2f 秒", i + 1, segment_duration)
                        
                        # 检查音频时长是否超过限制（约29秒）
                        max_audio_duration = 25.0  # 设置25秒作为安全限制，留有余量
                        if segment_duration > max_audio_duration:
                            self.logger.warning(
                                "第 %d 段音频时长 %.2f 秒超过限制 %.2f 秒，"
                                "该段文本可能未完全生成。段长度: %d 字符",
                                i + 1, segment_duration, max_audio_duration, len(segment)
                            )
                            self.logger.warning("该段内容: %s", segment[:100])
                        
                        audio_segments.append(segment_audio)
                    else:
                        self.logger.warning("第 %d 段未生成任何音频", i + 1)
                    
                    # 分段音频之间不添加停顿，直接拼接以保持流畅
                    # TTS生成的音频末尾通常已有自然的停顿，不需要额外添加
                except Exception as e:
                    self.logger.error("生成第 %d 段语音失败: %s", i + 1, e)
                    import traceback
                    self.logger.error(traceback.format_exc())
                    # 继续处理其他段
                    continue
            
            if not audio_segments:
                self.logger.error("所有音频段生成失败")
                return None
            
            # 使用交叉淡入淡出（crossfade）来平滑拼接音频段
            if len(audio_segments) == 0:
                return None
            elif len(audio_segments) == 1:
                combined_audio = audio_segments[0]
            else:
                # 交叉淡入淡出参数
                crossfade_duration = 0.05  # 交叉淡入淡出时长（秒）
                crossfade_samples = int(crossfade_duration * sample_rate)
                silence_threshold = 0.005  # 静音阈值（更敏感）
                
                optimized_segments = []
                
                for i, audio in enumerate(audio_segments):
                    if len(audio) == 0:
                        continue
                    
                    # 裁剪末尾的静音（更精确的检测）
                    trim_samples = int(0.15 * sample_rate)  # 最多裁剪0.15秒
                    trim_end = len(audio)
                    
                    # 使用滑动窗口检测静音，避免单个采样点的噪声
                    window_size = int(0.01 * sample_rate)  # 10ms窗口
                    for j in range(len(audio) - window_size, max(0, len(audio) - trim_samples - window_size), -window_size):
                        window = audio[j:j + window_size]
                        avg_amplitude = np.mean(np.abs(window))
                        if avg_amplitude > silence_threshold:
                            trim_end = j + window_size
                            break
                    
                    trimmed_audio = audio[:trim_end] if trim_end < len(audio) else audio
                    optimized_segments.append(trimmed_audio)
                
                # 使用交叉淡入淡出拼接
                if len(optimized_segments) == 0:
                    return None
                
                combined_audio = optimized_segments[0].copy()
                
                for i in range(1, len(optimized_segments)):
                    current_audio = optimized_segments[i]
                    
                    # 确保有足够的样本进行交叉淡入淡出
                    if len(combined_audio) >= crossfade_samples and len(current_audio) >= crossfade_samples:
                        # 获取前一段的末尾和后一段的开头
                        fade_out_end = combined_audio[-crossfade_samples:]
                        fade_in_start = current_audio[:crossfade_samples]
                        
                        # 创建淡出和淡入的权重曲线（线性）
                        fade_out_weights = np.linspace(1.0, 0.0, crossfade_samples)
                        fade_in_weights = np.linspace(0.0, 1.0, crossfade_samples)
                        
                        # 应用交叉淡入淡出
                        fade_out_audio = fade_out_end * fade_out_weights
                        fade_in_audio = fade_in_start * fade_in_weights
                        
                        # 混合交叉部分
                        crossfade_audio = fade_out_audio + fade_in_audio
                        
                        # 拼接：前一段（去掉末尾交叉部分）+ 交叉部分 + 后一段（去掉开头交叉部分）
                        combined_audio = np.concatenate([
                            combined_audio[:-crossfade_samples],
                            crossfade_audio,
                            current_audio[crossfade_samples:]
                        ])
                    else:
                        # 如果样本不够，直接拼接（添加很短的淡入淡出）
                        if len(current_audio) > 0:
                            # 对开头添加短暂的淡入
                            fade_in_len = min(int(0.02 * sample_rate), len(current_audio))
                            fade_in_weights = np.linspace(0.0, 1.0, fade_in_len)
                            current_audio[:fade_in_len] *= fade_in_weights
                            
                            combined_audio = np.concatenate([combined_audio, current_audio])
            
            # 转换为字节流
            wav_io = io.BytesIO()
            sf.write(wav_io, combined_audio, sample_rate, format="WAV")
            self.logger.info("语音生成完成，总时长约 %.2f 秒", len(combined_audio) / sample_rate)
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


