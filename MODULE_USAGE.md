# 模块拆分说明（app/coordinator.py, app/asr_module.py, app/rag_module.py）

## 入口：`app/coordinator.py`
- 负责创建 Flask 应用、注册全部路由、加载/保存配置。
- 实例化两个模块：
  - `asr = ASRModule(funasr_ws_url, dialect_data_dir, hotwords_file, collect_dialect_data)`
  - `rag = RAGModule(kb_path, llm_api_base, llm_api_key, model_name, embedding_model, base_dir)`
- 复合场景（如 `/api/recognize_and_query`）在协调层先调用 ASR，再调用 RAG，再调用 TTS。
- `web_voice_rag.py` 仅保留为兼容入口：`from app.coordinator import main`。

## ASR 模块：`ASRModule`（app/asr_module.py）
- 输入/输出：
  - `recognize_audio(audio_bytes, hotwords="") -> (text, confidence)`
  - `save_audio_for_training(audio_bytes, recognized_text, confidence=0.0, user_id="anonymous", corrected_text=None) -> audio_id`
  - `apply_correction(audio_id, corrected_text) -> (ok: bool, message: str)`
  - `read_hotwords() -> str`
  - `write_hotwords(content: str)`
- 职责：FunASR WebSocket 识别、方言数据采集保存、纠错写回、热词文件读写。
- 依赖：`funasr_ws_url`、方言数据目录、`hotwords_file` 路径、`collect_dialect_data` 开关。

## RAG 模块：`RAGModule`（app/rag_module.py）
- 输入/输出：
  - `add_document(file_path)`
  - `rebuild_vectorstore(upload_dir)`
  - `query(question: str) -> (answer: str, timings: dict, sources: list)`
  - `switch_llm(model_type: str)`
  - `text_to_speech(text: str) -> wav_bytes | None`
- 职责：嵌入加载、向量库持久化/重建、文档切分与入库、检索与 LLM 生成、TTS。
- 依赖：`kb_path`（首次创建备用）、LLM 配置、嵌入模型路径、`base_dir`（存放向量库/embedding）、上传目录路径由调用者传入 `rebuild_vectorstore`。

## 路由与调用关系（全部定义在 `coordinator.py`）
- ASR 路由：`/api/recognize`、`/api/save_recognition`、`/api/hotwords`、`/api/correct` → 调用 `ASRModule`。
- RAG 路由：`/api/query`（支持文字、图片、TTS）、`/api/files`(GET/DELETE)、`/api/upload_kb`、`/api/model` → 调用 `RAGModule`。
- 组合路由：`/api/recognize_and_query` 先 `ASRModule.recognize_audio`，再 `RAGModule.query`，最后 `RAGModule.text_to_speech`。

