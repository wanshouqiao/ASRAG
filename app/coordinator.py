#!/usr/bin/env python3
"""
协调文件：负责 Flask 应用、路由注册，调度 ASR 与 RAG 模块。
保持原有接口与行为，不新增功能。
"""

import base64
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, abort, send_file

from app.asr_module import ASRModule
from app.rag_module import RAGModule

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Flask 应用（显式指定模板与静态目录，避免因包目录变化找不到模板）
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
TEMPLATE_DIR = os.path.join(ROOT_DIR, "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "static")
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# ========== 管理端鉴权 ==========
from functools import wraps
import secrets

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")
_VALID_TOKENS = set()

def require_auth(fn):
    @wraps(fn)
    def _wrap(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        token = auth.split(" ", 1)[1].strip()
        if token not in _VALID_TOKENS:
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return _wrap


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response


# 路径与配置（ROOT_DIR 为项目根目录）
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
TEMP_AUDIO_DIR = os.path.join(ROOT_DIR, "temp_audio")
CONFIG_FILE = os.path.join(ROOT_DIR, "app_config.json")
HOTWORDS_FILE = os.path.join(ROOT_DIR, "hotwords.txt")
KB_PATH = os.path.join(ROOT_DIR, "knowledge.txt")

# 确保临时音频目录存在
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# 确保临时音频目录存在
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

asr_module = None
rag_module = None


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("加载配置文件失败: %s", e)
    return {}


def save_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("保存配置文件失败: %s", e)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin_page():
    return render_template("admin.html")

@app.route("/api/login", methods=["POST"])
def api_login():
    try:
        data = request.get_json(force=True)
        username = data.get("username", "")
        password = data.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            token = secrets.token_urlsafe(32)
            _VALID_TOKENS.add(token)
            return jsonify({"success": True, "token": token})
        return jsonify({"success": False, "error": "用户名或密码错误"}), 401
    except Exception as e:
        logger.error("登录失败: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


# --- 热词管理 ---
@app.route("/api/hotwords", methods=["GET", "POST"])
@require_auth
def manage_hotwords():
    if request.method == "GET":
        content = asr_module.read_hotwords()
        return jsonify({"hotwords": content})
    data = request.get_json()
    hotwords = data.get("hotwords", "")
    try:
        asr_module.write_hotwords(hotwords)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- ASR 识别 ---
@app.route("/api/recognize", methods=["POST"])
def recognize():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "没有上传音频文件"}), 400
        audio_file = request.files["audio"]
        audio_data = audio_file.read()
        hotwords = request.form.get("hotwords", "")
        t0 = time.time()
        recognized_text, confidence = asr_module.recognize_audio(audio_data, hotwords=hotwords)
        asr_time = time.time() - t0
        import uuid
        from datetime import datetime

        temp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        return jsonify(
            {
                "success": True,
                "text": recognized_text,
                "temp_id": temp_id,
                "audio_base64": audio_base64,
                "confidence": confidence,
                "timings": {"asr_time": asr_time},
            }
        )
    except Exception as e:
        logger.error("识别 API 错误: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/save_recognition", methods=["POST"])
def save_recognition():
    try:
        data = request.get_json()
        audio_base64 = data.get("audio_base64", "")
        recognized_text = data.get("recognized_text", "")
        corrected_text = data.get("corrected_text")
        confidence = data.get("confidence", 0.0)
        if not asr_module.collect_dialect_data:
            return jsonify({"error": "方言数据采集已关闭"}), 403
        if not audio_base64 or not recognized_text:
            return jsonify({"error": "缺少必要参数"}), 400
        audio_data = base64.b64decode(audio_base64)
        audio_id = asr_module.save_audio_for_training(
            audio_data=audio_data,
            recognized_text=recognized_text,
            confidence=confidence,
            user_id="anonymous",
            corrected_text=corrected_text,
        )
        return jsonify({"success": True, "audio_id": audio_id, "message": "数据已保存"})
    except Exception as e:
        logger.error("保存识别结果 API 错误: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/correct", methods=["POST"])
def correct_recognition():
    try:
        data = request.get_json()
        audio_id = data.get("audio_id")
        corrected_text = data.get("corrected_text", "").strip()
        if not asr_module.collect_dialect_data:
            return jsonify({"error": "方言数据采集已关闭"}), 403
        if not audio_id:
            return jsonify({"error": "缺少 audio_id"}), 400
        if not corrected_text:
            return jsonify({"error": "修正文本不能为空"}), 400
        ok, message = asr_module.apply_correction(audio_id, corrected_text)
        if not ok:
            return jsonify({"error": message}), 404 if "未找到" in message else 500
        return jsonify({"success": True, "message": message, "audio_id": audio_id})
    except Exception as e:
        logger.error("纠错 API 错误: %s", e)
        return jsonify({"error": str(e)}), 500


# --- RAG 查询 ---
@app.route("/api/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "问题不能为空"}), 400
        t0 = time.time()
        answer, rag_timings, sources = rag_module.query(question)
        timings = {"rag_time": time.time() - t0}
        timings.update(rag_timings)
        return jsonify({"success": True, "answer": answer, "timings": timings, "sources": sources})
    except Exception as e:
        logger.error("查询 API 错误: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/text_chat", methods=["POST"])
def text_chat():
    start_time = time.time()
    timings = {}
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        audio_id = data.get("audio_id")
        if not text:
            return jsonify({"error": "输入文本不能为空"}), 400
        t0 = time.time()
        answer, rag_timings, sources = rag_module.query(text)
        timings["rag_time"] = time.time() - t0
        timings.update(rag_timings)
        t0 = time.time()
        audio_bytes = rag_module.text_to_speech(answer)
        timings["tts_time"] = time.time() - t0
        timings["total_time"] = time.time() - start_time
        if audio_bytes:
            # 保存音频文件并返回URL，避免data URI长度限制
            audio_id = str(uuid.uuid4())
            audio_path = os.path.join(TEMP_AUDIO_DIR, f"{audio_id}.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            audio_size_mb = len(audio_bytes) / (1024 * 1024)
            logger.info("音频文件已保存: %s, 大小: %.2f MB", audio_id, audio_size_mb)
            audio_url = f"/api/audio/{audio_id}"
            return jsonify(
                {
                    "success": True,
                    "recognized_text": text,
                    "answer": answer,
                    "audio": audio_url,
                    "audio_id": audio_id,
                    "timings": timings,
                    "sources": sources,
                }
            )
        return jsonify(
            {
                "success": True,
                "recognized_text": text,
                "answer": answer,
                "audio": None,
                "audio_id": audio_id,
                "timings": timings,
                "sources": sources,
            }
        )
    except Exception as e:
        logger.error("文本对话 API 错误: %s", e)
        return jsonify({"error": str(e)}), 500


# --- 文件管理 ---
@app.route("/api/files", methods=["GET"])
@require_auth
def list_files():
    if not os.path.exists(UPLOAD_DIR):
        return jsonify({"success": True, "files": []})
    files = []
    for f in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(path):
            files.append({"name": f, "size": os.path.getsize(path), "mtime": os.path.getmtime(path)})
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify({"success": True, "files": files})


@app.route("/api/files", methods=["DELETE"])
@require_auth
def delete_file():
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            return jsonify({"error": "文件名不能为空"}), 400
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            rag_module.rebuild_vectorstore(UPLOAD_DIR)
            return jsonify({"success": True, "message": "文件已删除并重建知识库"})
        return jsonify({"error": "文件不存在"}), 404
    except Exception as e:
        logger.error("删除文件失败: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload_kb", methods=["POST"])
@require_auth
def upload_kb():
    try:
        if "file" not in request.files:
            return jsonify({"error": "没有上传文件"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "文件名为空"}), 400
        filename = os.path.basename(file.filename)
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        unique_filename = f"{name}_{timestamp}{ext}"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(UPLOAD_DIR, unique_filename)
        file.save(save_path)
        rag_module.add_document(save_path)
        config = load_config()
        config["last_uploaded_kb"] = save_path
        save_config(config)
        return jsonify(
            {"success": True, "message": f"文件已上传并添加到知识库: {unique_filename}", "kb_path": save_path}
        )
    except Exception as e:
        logger.error("上传知识库失败: %s", e)
        return jsonify({"error": str(e)}), 500


# --- 模型管理 ---
@app.route("/api/model", methods=["GET", "POST"])
@require_auth
def manage_model():
    if request.method == "GET":
        return jsonify(
            {
                "success": True,
                "current_model": rag_module.current_model_type,
                "available_models": list(rag_module.llm_config.keys()),
            }
        )
    try:
        data = request.get_json()
        model_type = data.get("model_type")
        if not model_type:
            return jsonify({"error": "模型类型不能为空"}), 400
        rag_module.switch_llm(model_type)
        config = load_config()
        config["model_type"] = model_type
        save_config(config)
        return jsonify(
            {"success": True, "message": f"已切换到模型: {model_type}", "current_model": model_type}
        )
    except Exception as e:
        logger.error("切换模型失败: %s", e)
        return jsonify({"error": str(e)}), 500


# --- 音频文件端点 ---
@app.route("/api/audio/<audio_id>", methods=["GET"])
def get_audio(audio_id):
    """返回临时音频文件"""
    try:
        audio_path = os.path.join(TEMP_AUDIO_DIR, f"{audio_id}.wav")
        if not os.path.exists(audio_path):
            return jsonify({"error": "音频文件不存在"}), 404
        return send_file(audio_path, mimetype="audio/wav")
    except Exception as e:
        logger.error("获取音频文件失败: %s", e)
        return jsonify({"error": str(e)}), 500


# --- 识别并查询 ---
@app.route("/api/recognize_and_query", methods=["POST"])
def recognize_and_query():
    start_time = time.time()
    timings = {}
    try:
        if "audio" not in request.files:
            return jsonify({"error": "没有上传音频文件"}), 400
        audio_file = request.files["audio"]
        audio_data = audio_file.read()
        hotwords = request.form.get("hotwords", "")
        t0 = time.time()
        recognized_text, confidence = asr_module.recognize_audio(audio_data, hotwords=hotwords)
        timings["asr_time"] = time.time() - t0
        if not recognized_text:
            return jsonify({"success": False, "error": "未能识别出文本，请重试"}), 400
        t0 = time.time()
        answer, rag_timings, sources = rag_module.query(recognized_text)
        timings["rag_time"] = time.time() - t0
        timings.update(rag_timings)
        t0 = time.time()
        audio_bytes = rag_module.text_to_speech(answer)
        timings["tts_time"] = time.time() - t0
        timings["total_time"] = time.time() - start_time
        if audio_bytes:
            # 保存音频文件并返回URL，避免data URI长度限制
            audio_id = str(uuid.uuid4())
            audio_path = os.path.join(TEMP_AUDIO_DIR, f"{audio_id}.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            audio_size_mb = len(audio_bytes) / (1024 * 1024)
            logger.info("音频文件已保存: %s, 大小: %.2f MB", audio_id, audio_size_mb)
            audio_url = f"/api/audio/{audio_id}"
            return jsonify(
                {
                    "success": True,
                    "recognized_text": recognized_text,
                    "answer": answer,
                    "audio": audio_url,
                    "audio_id": audio_id,
                    "timings": timings,
                    "sources": sources,
                }
            )
        return jsonify(
            {
                "success": True,
                "recognized_text": recognized_text,
                "answer": answer,
                "audio": None,
                "timings": timings,
                "sources": sources,
            }
        )
    except Exception as e:
        logger.error("识别并查询 API 错误: %s", e)
        return jsonify({"error": str(e)}), 500


def main():
    global asr_module, rag_module
    FUNASR_WS_URL = "wss://127.0.0.1:10095/"
    LLM_API_BASE = "http://localhost:8000/#"
    LLM_API_KEY = ""
    MODEL_NAME = "qwen2.5-7b-instruct"

    config = load_config()
    saved_model_type = config.get("model_type", "local")
    collect_dialect_data = config.get("collect_dialect_data", True)

    debug_mode = True
    if not debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        try:
            asr_module = ASRModule(
                funasr_ws_url=FUNASR_WS_URL,
                dialect_data_dir=os.path.join(ROOT_DIR, "dialect_data"),
                hotwords_file=HOTWORDS_FILE,
                collect_dialect_data=collect_dialect_data,
            )
            rag_module = RAGModule(
                kb_path=KB_PATH,
                llm_api_base=LLM_API_BASE,
                llm_api_key=LLM_API_KEY,
                model_name=MODEL_NAME,
                embedding_model=r"/data/AI/LlamaCPPProject/embedding/bge-large-zh-v1.5",
                base_dir=ROOT_DIR,
            )
            if saved_model_type and saved_model_type != "local":
                try:
                    rag_module.switch_llm(saved_model_type)
                    logger.info("已恢复模型配置: %s", saved_model_type)
                except Exception as e:
                    logger.warning("恢复模型配置失败: %s，使用默认配置", e)
        except Exception as e:
            logger.exception("初始化失败: %s", e)
            sys.exit(1)
    else:
        logger.info("主进程启动，等待子进程初始化模型...")
        logger.info("提示: Debug 模式下会看到两次初始化过程，这是正常的热重载机制")

    print("\n" + "=" * 60)
    print("Web 语音 RAG 系统已启动！")
    print("=" * 60)
    print("请在浏览器中打开: http://localhost:7000")
    print("按 Ctrl+C 退出程序")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=7000, debug=debug_mode, ssl_context=("lucky.taila62a2b.ts.net.crt","lucky.taila62a2b.ts.net.key"))


if __name__ == "__main__":
    main()


