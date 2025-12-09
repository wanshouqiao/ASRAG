#!/usr/bin/env python3
"""
语音 RAG Web 应用
基于 Flask 的浏览器界面，支持语音输入和 RAG 问答

使用步骤：
1. 先启动 llama.cpp server：
cd /data/AI/LlamaCPPProject/llama.cpp-master/build/bin
./llama-server \
-m /data/AI/LlamaCPPProject/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
-ngl -1 \
--host 0.0.0.0 \
--port 8000

2. 启动funasr
conda activate D:\AI\LlamaCPPProject\env
cd /data/AI/LlamaCPPProject/
python asr/FunASR/runtime/python/websocket/funasr_wss_server.py \
  --port 10095 \
  --asr_model /data/AI/LlamaCPPProject/asr/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/ \
  --asr_model_online /data/AI/LlamaCPPProject/asr/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/ \
  --vad_model /data/AI/LlamaCPPProject/asr/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/ \
  --punc_model /data/AI/LlamaCPPProject/asr/modelscope/hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/ 

3. 启动rag
conda activate /data/AI/LlamaCPPProject/env
cd /data/AI/LlamaCPPProject/application
python web_voice_rag.py

4. 在浏览器中打开 http://xx:5000
"""

import os
import sys
import time
import logging
import numpy as np
import io
import wave
import base64
import torch
import json
import websocket
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from kokoro import KPipeline, KModel
import soundfile as sf
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    PyMuPDFLoader = None
from langchain_text_splitters import RecursiveCharacterTextSplitter

import warnings
# 忽略特定的警告
warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm is deprecated.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 简单的 CORS 处理
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


class WebVoiceRAG:
    """Web 语音 RAG 系统"""
    
    def __init__(self, 
                 funasr_ws_url="ws://127.0.0.1:10095/",
                 kb_path="knowledge.txt",
                 llm_api_base="http://127.0.0.1:8000/#",
                 llm_api_key="",
                 model_name="qwen2.5-7b-instruct",
                 embedding_model=r"/data/AI/LlamaCPPProject/embedding/bge-large-zh-v1.5",
                 collect_dialect_data=True):
        """初始化 Web 语音 RAG 系统"""
        logger.info("正在初始化 Web 语音 RAG 系统...")
        
        # 保存 FunASR WebSocket 服务地址
        logger.info("配置 FunASR WebSocket 服务...")
        self.funasr_ws_url = funasr_ws_url
        logger.info(f"FunASR 服务地址: {self.funasr_ws_url}")
        
        # 初始化 LLM
        logger.info("初始化 LLM...")
        self.llm_config = {
            "local": {
                "api_base": llm_api_base,
                "api_key": llm_api_key,
                "model_name": model_name
            },
            "deepseek": {
                "api_base": "https://api.deepseek.com",
                "api_key": "sk-2c57fe7aa3224eeaa64d54a64555d5da",
                "model_name": "deepseek-chat"
            }
        }
        self.current_model_type = "local" # 默认使用本地模型
        self.switch_llm("local")
        
        # 初始化嵌入模型
        logger.info("初始化嵌入模型...")
        self.embedding_model_name = embedding_model
        self.embeddings = self._init_embeddings()

        # 初始化向量库（使用统一的持久化存储）
        self.vectorstore = None
        self.retriever = None
        self._init_vectorstore(kb_path)
        
        # 初始化 TTS 模型
        logger.info("初始化 TTS 模块...")
        self.tts_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tts_repo_id = 'hexgrad/Kokoro-82M-v1.1-zh'
        # 获取当前文件的上一级目录
        parent_dir = os.path.dirname(os.path.dirname(__file__))

        # 重新拼接路径
        tts_model_path = '/data/AI/LlamaCPPProject/tts/ckpts/kokoro-v1.1/kokoro-v1_1-zh.pth'
        tts_config_path = '/data/AI/LlamaCPPProject/tts/ckpts/kokoro-v1.1/config.json'
        voice_path = '/data/AI/LlamaCPPProject/tts/ckpts/kokoro-v1.1/voices/zf_001.pt'

        self.voice_tensor = torch.load(voice_path, weights_only=True)
        
        # 初始化 TTS 模型
        self.tts_model = KModel(model=tts_model_path, config=tts_config_path, repo_id=tts_repo_id).to(self.tts_device).eval()
        
        # 创建英文 pipeline 用于处理英文部分
        self.en_pipeline = KPipeline(lang_code='a', repo_id=tts_repo_id, model=False)
        
        # 定义英文处理函数
        def en_callable(text):
            return next(self.en_pipeline(text)).phonemes
        
        # 创建支持中英文混合的中文 pipeline
        self.tts_pipeline = KPipeline(lang_code='z', repo_id=tts_repo_id, model=self.tts_model, en_callable=en_callable)
        
        # 速度控制函数
        def speed_callable(len_ps):
            speed = 0.8
            if len_ps <= 83:
                speed = 1
            elif len_ps < 183:
                speed = 1 - (len_ps - 83) / 500
            return speed * 1.1
        
        self.speed_callable = speed_callable
        
        # 初始化方言数据采集目录
        self.collect_dialect_data = collect_dialect_data
        self.dialect_data_dir = os.path.join(os.path.dirname(__file__), "dialect_data")
        self.dialect_audio_dir = os.path.join(self.dialect_data_dir, "audio")
        self.dialect_metadata_file = os.path.join(self.dialect_data_dir, "metadata.jsonl")
        if self.collect_dialect_data:
            os.makedirs(self.dialect_audio_dir, exist_ok=True)
            logger.info(f"方言数据采集目录: {self.dialect_data_dir}")
        else:
            logger.info("方言数据采集已关闭")
        
        logger.info("Web 语音 RAG 系统初始化完成！")

    def switch_llm(self, model_type: str):
        """切换 LLM 模型"""
        if model_type not in self.llm_config:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        config = self.llm_config[model_type]
        logger.info(f"切换 LLM 到: {model_type} ({config['model_name']})")
        
        self.llm = ChatOpenAI(
            openai_api_base=config["api_base"],
            openai_api_key=config["api_key"],
            model_name=config["model_name"],
            temperature=0.01,
            max_tokens=512
        )
        self.current_model_type = model_type

    def _init_embeddings(self):
        """初始化嵌入模型，优先使用本地目录，失败则回退在线模型"""
        embedding_path = os.path.join(os.path.dirname(__file__), "embedding")
        # 尝试本地
        if os.path.exists(embedding_path) and os.path.exists(os.path.join(embedding_path, "modules.json")):
            logger.info(f"使用本地嵌入模型: {embedding_path}")
            encode_kwargs = {'normalize_embeddings': True} # BGE 建议开启归一化
            return HuggingFaceEmbeddings(
                model_name=embedding_path,
                model_kwargs={"device": "cuda"},
                encode_kwargs=encode_kwargs
            )
        # 在线（会自动缓存）
        logger.info(f"使用在线嵌入模型: {self.embedding_model_name}")
        logger.info("注意：首次使用会下载模型到本地缓存，之后会使用缓存，无需再次下载")
        encode_kwargs = {'normalize_embeddings': True} # BGE 建议开启归一化
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs=encode_kwargs
        )

    def _get_combined_vectorstore_dir(self) -> str:
        """获取统一向量库的存储目录"""
        # 从完整路径中提取模型名称（最后一部分），并替换特殊字符
        model_name = os.path.basename(self.embedding_model_name)
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace('-', '_').replace(':', '_')
        return os.path.join(
            os.path.dirname(__file__),
            'vectorstores',
            f"combined_kb_{safe_model_name}"
        )

    def _init_vectorstore(self, default_kb_path: str):
        """初始化向量库：加载已有库，或从默认文件创建"""
        vs_dir = self._get_combined_vectorstore_dir()
        
        # 尝试加载已有向量库
        if os.path.exists(os.path.join(vs_dir, "index.faiss")):
            logger.info(f"检测到已保存的向量库，正在加载...")
            # 临时抑制 FAISS 的日志输出
            faiss_logger = logging.getLogger('faiss.loader')
            original_level = faiss_logger.level
            faiss_logger.setLevel(logging.ERROR)
            try:
                self.vectorstore = FAISS.load_local(
                    vs_dir, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                logger.info("✓ 向量库加载成功！所有已添加的文档可供检索")
                return
            except Exception as e:
                logger.error(f"加载向量库失败: {e}，将尝试重新创建")
            finally:
                # 恢复 FAISS 日志级别
                faiss_logger.setLevel(original_level)
        
        # 如果没有已有向量库，尝试从默认路径创建
        if os.path.exists(default_kb_path):
            logger.info(f"首次启动，从默认文件创建向量库: {os.path.basename(default_kb_path)}")
            self.add_document(default_kb_path)
        else:
            logger.warning(f"未找到已保存的向量库，也未找到默认知识库文件")
            logger.info("提示: 启动后可通过 Web 界面上传文档来创建知识库")
            self.vectorstore = None
            self.retriever = None

    def rebuild_vectorstore(self):
        """重建向量库（用于删除文件后）"""
        logger.info("正在重建向量库...")
        # 清空现有向量库
        self.vectorstore = None
        self.retriever = None
        
        # 获取 uploads 目录下所有文件
        if not os.path.exists(UPLOAD_DIR):
            files = []
        else:
            files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        
        if not files:
            logger.info("没有文件，向量库为空")
            # 清除磁盘上的向量库文件
            vs_dir = self._get_combined_vectorstore_dir()
            import shutil
            if os.path.exists(vs_dir):
                shutil.rmtree(vs_dir)
            return

        # 重新添加所有文件
        all_chunks = []
        for filename in files:
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    if PyMuPDFLoader:
                        loader = PyMuPDFLoader(file_path)
                        docs = loader.load()
                    else:
                        logger.warning("PyMuPDFLoader 不可用，回退到 PyPDFLoader")
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                else:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()

                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", "。", "，", " ", ""]
                ).split_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"读取文件 {filename} 失败: {e}")
        
        if all_chunks:
            self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # 保存
            vs_dir = self._get_combined_vectorstore_dir()
            if not os.path.exists(vs_dir):
                os.makedirs(vs_dir)
            self.vectorstore.save_local(vs_dir)
            logger.info(f"向量库重建完成，共 {len(files)} 个文件")

    def add_document(self, file_path: str):
        """向向量库添加文档"""
        logger.info(f"正在处理文档: {file_path}")
        try:
            if file_path.lower().endswith('.pdf'):
                if PyMuPDFLoader:
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                else:
                    logger.warning("PyMuPDFLoader 不可用，回退到 PyPDFLoader")
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
            else:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                separators=["\n\n", "\n", "。", "，", " ", ""]
            ).split_documents(docs)
            
            if not chunks:
                logger.warning("文档为空或无法切分")
                return

            if self.vectorstore is None:
                logger.info("创建新向量库...")
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            else:
                logger.info("添加到现有向量库...")
                self.vectorstore.add_documents(chunks)
            
            # 保存向量库
            vs_dir = self._get_combined_vectorstore_dir()
            if not os.path.exists(vs_dir):
                os.makedirs(vs_dir)
            self.vectorstore.save_local(vs_dir)
            logger.info(f"向量库已更新并保存到: {vs_dir}")
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise e
    
    def recognize_audio(self, audio_data: bytes, sample_rate: int = 16000, hotwords: str = "") -> tuple:
        """
        使用 FunASR WebSocket 服务识别音频数据
        
        Args:
            audio_data: WAV 格式的音频数据（字节）
            sample_rate: 目标采样率（默认 16kHz）
            hotwords: 热词配置，JSON 字符串格式，例如 '{"阿里巴巴": 20, "达摩院": 30}'
            
        Returns:
            (识别出的文本, 置信度分数)
        """
        try:
            # 读取 WAV 文件并转换为 PCM 数据
            wav_io = io.BytesIO(audio_data)
            with wave.open(wav_io, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate_wav = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                audio_bytes = wav_file.readframes(frames)
            
            # 转换为 numpy 数组
            if sample_width == 2:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            elif sample_width == 4:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.int16)
            else:
                audio_array = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16)
            
            # 如果是立体声，转换为单声道
            if n_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            # 如果采样率不是 16kHz，需要重采样
            if sample_rate_wav != 16000:
                logger.info(f"重采样从 {sample_rate_wav}Hz 到 16000Hz")
                from scipy import signal
                num_samples = int(len(audio_array) * 16000 / sample_rate_wav)
                audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
            
            # 转换回字节（PCM 格式）
            pcm_data = audio_array.tobytes()
            
            # 连接 FunASR WebSocket 服务
            logger.info(f"连接 FunASR 服务: {self.funasr_ws_url}")
            
            result_text = ""
            result_confidence = 0.0
            
            def on_message(ws, message):
                nonlocal result_text, result_confidence
                try:
                    data = json.loads(message)
                    
                    # 打印完整的返回数据，方便调试
                    logger.info(f"FunASR 返回数据: {json.dumps(data, ensure_ascii=False)}")
                    
                    text = ""
                    if 'text' in data:
                        text = data['text']
                    elif 'result' in data:
                        text = data['result']
                    
                    # 尝试提取置信度（FunASR可能在不同字段返回）
                    if 'confidence' in data:
                        result_confidence = float(data['confidence'])
                    elif 'score' in data:
                        result_confidence = float(data['score'])
                    elif 'timestamp' in data:  # 有时confidence在timestamp字段的子结构中
                        # 暂时设置为0，实际需要根据FunASR返回格式调整
                        result_confidence = 0.0
                    
                    if text:
                        result_text = text
                        logger.info(f"识别结果: {result_text}, 置信度: {result_confidence:.3f}")
                        ws.close()  # 收到结果后关闭连接
                    else:
                        logger.info("收到空结果，继续等待...")
                        
                except Exception as e:
                    logger.error(f"解析 WebSocket 消息失败: {e}")
            
            def on_error(ws, error):
                logger.error(f"WebSocket 错误: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket 连接关闭")
            
            def on_open(ws):
                logger.info("WebSocket 连接已建立")
                
                # 发送配置信息
                config = {
                    "mode": "offline",
                    "chunk_size": [5, 10, 5],
                    "chunk_interval": 10,
                    "wav_name": "microphone",
                    "is_speaking": True
                }
                
                if hotwords:
                    # 注意：服务端期望的键名是 "hotword" (单数)，而不是 "hotwords" (复数)
                    # 并且应该是字符串格式，而不是 JSON 对象
                    config["hotword"] = hotwords.strip() if isinstance(hotwords, str) else str(hotwords)
                    #logger.info(f"发送热词配置: {config['hotword']}")

                ws.send(json.dumps(config))
                
                # 发送音频数据
                # VAD 通常对 60ms (1920 bytes) 或更大的块处理效果较好
                chunk_size = 1920  # 60ms @ 16kHz
                for i in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[i:i+chunk_size]
                    ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                
                # 发送结束标志
                ws.send(json.dumps({"is_speaking": False}), opcode=websocket.ABNF.OPCODE_TEXT)
                logger.info("音频数据发送完成")
            
            # 创建 WebSocket 连接
            ws = websocket.WebSocketApp(
                self.funasr_ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                subprotocols=["binary"]
            )
            
            # 运行 WebSocket（设置超时）
            import threading
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            ws_thread.join(timeout=10)  # 最多等待 10 秒
            
            if ws_thread.is_alive():
                ws.close()
                logger.warning("WebSocket 超时，强制关闭")
            
            return (result_text if result_text else "", result_confidence)
            
        except Exception as e:
            logger.error(f"音频识别失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ("", 0.0)
    
    def save_audio_for_training(self, audio_data: bytes, recognized_text: str, 
                                confidence: float = 0.0, user_id: str = "anonymous",
                                corrected_text: str = None) -> str:
        """
        保存音频数据用于后续训练
        
        Args:
            audio_data: 原始音频数据（WAV格式）
            recognized_text: ASR识别出的文本
            confidence: 识别置信度分数
            user_id: 用户ID（已脱敏）
            corrected_text: 用户修正后的文本（如果有）
            
        Returns:
            保存的音频文件ID
        """
        try:
            if not self.collect_dialect_data:
                logger.debug("已禁用方言数据采集，跳过保存")
                return ""
            # 生成唯一ID
            import uuid
            from datetime import datetime
            
            audio_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            audio_filename = f"{audio_id}.wav"
            audio_path = os.path.join(self.dialect_audio_dir, audio_filename)
            
            # 确保目录存在
            os.makedirs(self.dialect_audio_dir, exist_ok=True)
            
            # 保存音频文件
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            
            # 计算音频时长
            wav_io = io.BytesIO(audio_data)
            with wave.open(wav_io, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
            
            # 保存元数据（使用 JSONL 格式，每行一个 JSON 对象）
            metadata = {
                "audio_id": audio_id,
                "audio_file": audio_filename,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "recognized_text": recognized_text,
                "corrected_text": corrected_text if corrected_text else None,
                "confidence": confidence,
                "duration": duration,
                "is_corrected": corrected_text is not None and corrected_text != recognized_text
            }
            
            with open(self.dialect_metadata_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
            
            logger.info(f"音频数据已保存: {audio_id}, 时长={duration:.2f}s, 置信度={confidence:.3f}")
            return audio_id
            
        except Exception as e:
            logger.error(f"保存音频数据失败: {e}")
            return ""
    
    def query(self, question: str) -> tuple[str, dict, list]:
        """使用 RAG 系统回答问题"""
        timings = {}
        if not question.strip():
            return "问题不能为空。", timings, []
        
        if not self.retriever:
            return "知识库为空，请先上传文档。", timings, []

        logger.info(f"用户问题: {question}")
        
        try:
            # 检索相关文档
            t0 = time.time()
            docs_retrieved = self.retriever.invoke(question)
            timings['retrieval'] = time.time() - t0
            context = "\n\n".join(d.page_content for d in docs_retrieved)
            
            # 构建提示词
            prompt = (
                "你是一个中文助理，请严格依据下面提供的知识库内容回答用户问题，"
                "如果知识库中没有相关信息，就说不知道，不要编造，也不要扩展。\n\n"
                f"【知识库内容】:\n{context}\n\n"
                f"【用户问题】:\n{question}\n\n"
                "请用简体中文回答："
            )
            
            # 调用 LLM
            logger.info("正在生成回答...")
            t0 = time.time()
            response = self.llm.invoke(prompt)
            timings['llm_generation'] = time.time() - t0
            answer = response.content
            logger.info("回答生成完成")
            
            # 格式化文档信息
            sources = []
            for doc in docs_retrieved:
                sources.append({
                    'content': doc.page_content,
                    'source': os.path.basename(doc.metadata.get('source', '未知来源'))
                })
                
            return answer, timings, sources
            
        except Exception as e:
            logger.error(f"RAG 查询失败: {e}")
            return f"抱歉，生成回答时出现错误: {str(e)}", timings, []
    
    def text_to_speech(self, text: str) -> bytes:
        """
        将文本转换为语音
        
        Args:
            text: 要转换的文本
            
        Returns:
            WAV 格式的音频数据（字节）
        """
        try:
            # 清理文本（移除 markdown 标记等）
            clean_text = text.replace('**', '').replace('*', '').strip()
            
            # 生成语音
            logger.info(f"正在生成语音: {clean_text[:50]}...")
            generator = self.tts_pipeline(clean_text, voice=self.voice_tensor, speed=self.speed_callable)
            result = next(generator)
            wav = result.audio
            
            # 转换为 WAV 字节
            wav_io = io.BytesIO()
            sf.write(wav_io, wav, 24000, format='WAV')
            wav_bytes = wav_io.getvalue()
            
            logger.info("语音生成完成")
            return wav_bytes
            
        except Exception as e:
            logger.error(f"TTS 生成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


# 全局 VoiceRAG 实例
voice_rag = None
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "app_config.json")
HOTWORDS_FILE = os.path.join(os.path.dirname(__file__), "hotwords.txt")

def load_config():
    """加载配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
    return {}

def save_config(config):
    """保存配置"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/hotwords', methods=['GET', 'POST'])
def manage_hotwords():
    """管理热词配置"""
    if request.method == 'GET':
        content = ""
        if os.path.exists(HOTWORDS_FILE):
            try:
                with open(HOTWORDS_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"读取热词文件失败: {e}")
        return jsonify({'hotwords': content})
    else:
        data = request.get_json()
        hotwords = data.get('hotwords', '')
        
        try:
            with open(HOTWORDS_FILE, 'w', encoding='utf-8') as f:
                f.write(hotwords)
        except Exception as e:
            logger.error(f"保存热词文件失败: {e}")
            return jsonify({'error': str(e)}), 500
        
        return jsonify({'success': True})


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """音频识别 API - 只识别，不保存"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '没有上传音频文件'}), 400
        
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # 获取热词
        hotwords = request.form.get('hotwords', '')
        
        # 识别音频
        t0 = time.time()
        recognized_text, confidence = voice_rag.recognize_audio(audio_data, hotwords=hotwords)
        asr_time = time.time() - t0
        
        # 生成临时ID（前端用于标识）
        import uuid
        from datetime import datetime
        temp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 将音频转为base64返回给前端（让前端暂存）
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return jsonify({
            'success': True,
            'text': recognized_text,
            'temp_id': temp_id,
            'audio_base64': audio_base64,  # 前端暂存
            'confidence': confidence,
            'timings': {'asr_time': asr_time}
        })
        
    except Exception as e:
        logger.error(f"识别 API 错误: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_recognition', methods=['POST'])
def save_recognition():
    """保存识别结果 - 用户确认或纠错后调用"""
    try:
        data = request.get_json()
        
        audio_base64 = data.get('audio_base64', '')
        recognized_text = data.get('recognized_text', '')
        corrected_text = data.get('corrected_text')  # 如果有纠错
        confidence = data.get('confidence', 0.0)
        temp_id = data.get('temp_id', '')
        
        if not voice_rag.collect_dialect_data:
            return jsonify({'error': '方言数据采集已关闭'}), 403

        if not audio_base64 or not recognized_text:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 解码音频数据
        audio_data = base64.b64decode(audio_base64)
        
        # 保存音频和元数据
        audio_id = voice_rag.save_audio_for_training(
            audio_data=audio_data,
            recognized_text=recognized_text,
            confidence=confidence,
            user_id="anonymous",
            corrected_text=corrected_text
        )
        
        return jsonify({
            'success': True,
            'audio_id': audio_id,
            'message': '数据已保存'
        })
        
    except Exception as e:
        logger.error(f"保存识别结果 API 错误: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query():
    """RAG 查询 API"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        # 使用 RAG 回答问题
        t0 = time.time()
        answer, rag_timings, sources = voice_rag.query(question)
        timings = {}
        timings['rag_time'] = time.time() - t0
        timings.update(rag_timings)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'timings': timings,
            'sources': sources
        })
        
    except Exception as e:
        logger.error(f"查询 API 错误: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model', methods=['GET', 'POST'])
def manage_model():
    """获取或切换当前模型"""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'current_model': voice_rag.current_model_type,
            'available_models': list(voice_rag.llm_config.keys())
        })
    
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        
        if not model_type:
            return jsonify({'error': '模型类型不能为空'}), 400
            
        voice_rag.switch_llm(model_type)
        
        # 保存配置
        config = load_config()
        config['model_type'] = model_type
        save_config(config)
        
        return jsonify({
            'success': True,
            'message': f'已切换到模型: {model_type}',
            'current_model': model_type
        })
        
    except Exception as e:
        logger.error(f"切换模型失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/correct', methods=['POST'])
def correct_recognition():
    """用户纠错接口 - 记录用户对识别结果的修正"""
    try:
        data = request.get_json()
        audio_id = data.get('audio_id')
        corrected_text = data.get('corrected_text', '').strip()

        if not voice_rag.collect_dialect_data:
            return jsonify({'error': '方言数据采集已关闭'}), 403
        
        if not audio_id:
            return jsonify({'error': '缺少 audio_id'}), 400
        
        if not corrected_text:
            return jsonify({'error': '修正文本不能为空'}), 400
        
        # 读取元数据文件，找到对应的记录并更新
        metadata_file = voice_rag.dialect_metadata_file
        if not os.path.exists(metadata_file):
            return jsonify({'error': '元数据文件不存在'}), 404
        
        # 读取所有记录
        records = []
        found = False
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record['audio_id'] == audio_id:
                        # 更新记录
                        record['corrected_text'] = corrected_text
                        record['is_corrected'] = True
                        record['correction_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                        found = True
                        logger.info(f"用户纠错: {audio_id}, 原文='{record['recognized_text']}', 修正='{corrected_text}'")
                    records.append(record)
        
        if not found:
            return jsonify({'error': f'未找到音频记录: {audio_id}'}), 404
        
        # 重写元数据文件
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return jsonify({
            'success': True,
            'message': '纠错信息已保存',
            'audio_id': audio_id
        })
        
    except Exception as e:
        logger.error(f"纠错 API 错误: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    """获取知识库文件列表"""
    if not os.path.exists(UPLOAD_DIR):
        return jsonify({'success': True, 'files': []})
    
    files = []
    for f in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(path):
            files.append({
                'name': f,
                'size': os.path.getsize(path),
                'mtime': os.path.getmtime(path)
            })
    # 按时间倒序
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return jsonify({'success': True, 'files': files})

@app.route('/api/files', methods=['DELETE'])
def delete_file():
    """删除知识库文件"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': '文件名不能为空'}), 400
            
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            # 重建向量库
            voice_rag.rebuild_vectorstore()
            return jsonify({'success': True, 'message': '文件已删除并重建知识库'})
        else:
            return jsonify({'error': '文件不存在'}), 404
            
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_kb', methods=['POST'])
def upload_kb():
    """上传知识库文件并向量化"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        # 生成唯一文件名（支持中文）
        filename = os.path.basename(file.filename)
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        unique_filename = f"{name}_{timestamp}{ext}"
        
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(UPLOAD_DIR, unique_filename)
        file.save(save_path)

        # 添加到向量库
        voice_rag.add_document(save_path)

        # 保存新的知识库路径到配置（可选，如果需要记录最后一次上传的文件）
        config = load_config()
        config['last_uploaded_kb'] = save_path
        save_config(config)

        return jsonify({
            'success': True,
            'message': f'文件已上传并添加到知识库: {unique_filename}',
            'kb_path': save_path
        })

    except Exception as e:
        logger.error(f"上传知识库失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize_and_query', methods=['POST'])
def recognize_and_query():
    """识别并查询 API（一步完成）"""
    start_time = time.time()
    timings = {}
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '没有上传音频文件'}), 400
        
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # 获取热词
        hotwords = request.form.get('hotwords', '')
        
        # 识别音频
        t0 = time.time()
        recognized_text = voice_rag.recognize_audio(audio_data, hotwords=hotwords)
        timings['asr_time'] = time.time() - t0
        
        if not recognized_text:
            return jsonify({
                'success': False,
                'error': '未能识别出文本，请重试'
            }), 400
        
        # 使用 RAG 回答问题
        t0 = time.time()
        answer, rag_timings, sources = voice_rag.query(recognized_text)
        timings['rag_time'] = time.time() - t0
        timings.update(rag_timings)
        
        # 生成语音
        t0 = time.time()
        audio_bytes = voice_rag.text_to_speech(answer)
        timings['tts_time'] = time.time() - t0
        
        timings['total_time'] = time.time() - start_time
        
        if audio_bytes:
            # 将音频转换为 base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return jsonify({
                'success': True,
                'recognized_text': recognized_text,
                'answer': answer,
                'audio': f'data:audio/wav;base64,{audio_base64}',
                'timings': timings,
                'sources': sources
            })
        else:
            # 如果 TTS 失败，仍然返回文本
            return jsonify({
                'success': True,
                'recognized_text': recognized_text,
                'answer': answer,
                'audio': None,
                'timings': timings,
                'sources': sources
            })
        
    except Exception as e:
        logger.error(f"识别并查询 API 错误: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/text_chat', methods=['POST'])
def text_chat():
    """文本对话 API（支持 TTS）"""
    start_time = time.time()
    timings = {}

    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        audio_id = data.get('audio_id')  # 获取前端传递的 audio_id
        
        if not text:
            return jsonify({'error': '输入文本不能为空'}), 400
        
        # 使用 RAG 回答问题
        t0 = time.time()
        answer, rag_timings, sources = voice_rag.query(text)
        timings['rag_time'] = time.time() - t0
        timings.update(rag_timings)
        
        # 生成语音
        t0 = time.time()
        audio_bytes = voice_rag.text_to_speech(answer)
        timings['tts_time'] = time.time() - t0
        
        timings['total_time'] = time.time() - start_time

        if audio_bytes:
            # 将音频转换为 base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return jsonify({
                'success': True,
                'recognized_text': text,  # 为了复用前端 displayResult
                'answer': answer,
                'audio': f'data:audio/wav;base64,{audio_base64}',
                'audio_id': audio_id,  # 返回 audio_id
                'timings': timings,
                'sources': sources
            })
        else:
            return jsonify({
                'success': True,
                'recognized_text': text,
                'answer': answer,
                'audio': None,
                'audio_id': audio_id,  # 返回 audio_id
                'timings': timings,
                'sources': sources
            })
        
    except Exception as e:
        logger.error(f"文本对话 API 错误: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    """主函数"""
    global voice_rag
    
    # 配置路径
    FUNASR_WS_URL = "ws://127.0.0.1:10095/"
    
    # LLM 配置
    LLM_API_BASE = "http://localhost:8000/#"
    LLM_API_KEY = ""
    MODEL_NAME = "qwen2.5-7b-instruct"
    
    # 优先从配置文件加载知识库路径
    config = load_config()
    
    # 保存模型配置供后续恢复
    saved_model_type = config.get('model_type', 'local')
    collect_dialect_data = config.get('collect_dialect_data', True)

    # KB_PATH 只用作首次创建向量库时的备用文件
    # 正常情况下会自动加载已保存的向量库（vectorstores/combined_kb_*）
    KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge.txt")
    
    # 初始化 VoiceRAG
    # 为了支持热重载 (debug=True)，我们需要避免在主进程中初始化重型模型
    # 只有在非 debug 模式，或者在 debug 模式的子进程 (WERKZEUG_RUN_MAIN='true') 中才初始化
    debug_mode = True
    
    if not debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        try:
            voice_rag = WebVoiceRAG(
                funasr_ws_url=FUNASR_WS_URL,
                kb_path=KB_PATH,
                llm_api_base=LLM_API_BASE,
                llm_api_key=LLM_API_KEY,
                model_name=MODEL_NAME,
                collect_dialect_data=collect_dialect_data
            )
            # 恢复上次使用的模型配置
            if saved_model_type and saved_model_type != 'local':
                try:
                    voice_rag.switch_llm(saved_model_type)
                    logger.info(f"已恢复模型配置: {saved_model_type}")
                except Exception as e:
                    logger.warning(f"恢复模型配置失败: {e}，使用默认配置")
        except Exception as e:
            logger.exception(f"初始化失败: {e}")
            sys.exit(1)
    else:
        logger.info("主进程启动，等待子进程初始化模型...")
        logger.info("提示: Debug 模式下会看到两次初始化过程，这是正常的热重载机制")
    
    # 启动 Flask 服务器
    print("\n" + "="*60)
    print("Web 语音 RAG 系统已启动！")
    print("="*60)
    print("请在浏览器中打开: http://localhost:5000")
    print("按 Ctrl+C 退出程序")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)


if __name__ == "__main__":
    main()

