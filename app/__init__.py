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