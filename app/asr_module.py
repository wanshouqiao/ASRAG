#!/usr/bin/env python3
"""
ASR 模块：负责语音识别、方言数据采集与纠错、热词读写。
保持原有行为，不改业务逻辑。
"""

import base64
import io
import json
import logging
import os
import time
import uuid
import wave
from datetime import datetime
from typing import Tuple

import numpy as np
import websocket


class ASRModule:
    """封装 FunASR 相关操作"""

    def __init__(
        self,
        funasr_ws_url: str,
        dialect_data_dir: str,
        hotwords_file: str,
        collect_dialect_data: bool = True,
    ):
        self.logger = logging.getLogger(__name__)
        self.funasr_ws_url = funasr_ws_url
        self.collect_dialect_data = collect_dialect_data
        self.hotwords_file = hotwords_file

        # 方言数据目录与元数据
        self.dialect_data_dir = dialect_data_dir
        self.dialect_audio_dir = os.path.join(self.dialect_data_dir, "audio")
        self.dialect_metadata_file = os.path.join(self.dialect_data_dir, "metadata.jsonl")
        if self.collect_dialect_data:
            os.makedirs(self.dialect_audio_dir, exist_ok=True)
            self.logger.info("方言数据采集目录: %s", self.dialect_data_dir)
        else:
            self.logger.info("方言数据采集已关闭")

    # 热词文件读写
    def read_hotwords(self) -> str:
        if not os.path.exists(self.hotwords_file):
            return ""
        try:
            with open(self.hotwords_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error("读取热词文件失败: %s", e)
            return ""

    def get_formatted_hotwords(self) -> str:
        """读取热词文件并格式化为 SeacoParaformer 需要的空格分隔字符串"""
        content = self.read_hotwords()
        if not content:
            return ""
        lines = content.split('\n')
        words = []
        for line in lines:
            part = line.strip()
            if not part:
                continue
            # 保留权重，因为 SeacoParaformer 可能需要 "词 权重" 格式
            words.append(part)
        return " ".join(words)

    def write_hotwords(self, content: str):
        try:
            with open(self.hotwords_file, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            self.logger.error("保存热词文件失败: %s", e)
            raise

    def recognize_audio(self, audio_data: bytes, hotwords: str = "") -> Tuple[str, float]:
        """
        使用 FunASR WebSocket 服务识别音频数据
        返回: (识别文本, 置信度)
        """
        try:
            # 读取 WAV
            wav_io = io.BytesIO(audio_data)
            with wave.open(wav_io, "rb") as wav_file:
                frames = wav_file.getnframes()
                sample_rate_wav = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                audio_bytes = wav_file.readframes(frames)

            # PCM 转换
            if sample_width == 2:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            elif sample_width == 4:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.int16)
            else:
                audio_array = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16)

            # 单声道
            if n_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)

            # 重采样到 16k
            if sample_rate_wav != 16000:
                self.logger.info("重采样从 %sHz 到 16000Hz", sample_rate_wav)
                from scipy import signal

                num_samples = int(len(audio_array) * 16000 / sample_rate_wav)
                audio_array = signal.resample(audio_array, num_samples).astype(np.int16)

            pcm_data = audio_array.tobytes()

            self.logger.info("连接 FunASR 服务: %s", self.funasr_ws_url)
            result_text = ""
            result_confidence = 0.0

            def on_message(ws, message):
                nonlocal result_text, result_confidence
                try:
                    data = json.loads(message)
                    self.logger.info("FunASR 返回数据: %s", json.dumps(data, ensure_ascii=False))
                    text = data.get("text") or data.get("result") or ""
                    if "confidence" in data:
                        result_confidence = float(data["confidence"])
                    elif "score" in data:
                        result_confidence = float(data["score"])
                    elif "timestamp" in data:
                        result_confidence = 0.0
                    if text:
                        result_text = text
                        self.logger.info("识别结果: %s, 置信度: %.3f", result_text, result_confidence)
                        ws.close()
                except Exception as e:
                    self.logger.error("解析 WebSocket 消息失败: %s", e)

            def on_error(ws, error):
                self.logger.error("WebSocket 错误: %s", error)

            def on_close(ws, close_status_code, close_msg):
                self.logger.info("WebSocket 连接关闭")

            def on_open(ws):
                self.logger.info("WebSocket 连接已建立")
                config = {
                    "mode": "offline",
                    "chunk_size": [5, 10, 5],
                    "chunk_interval": 10,
                    "wav_name": "microphone",
                    "is_speaking": True,
                }
                if hotwords:
                    config["hotword"] = hotwords.strip() if isinstance(hotwords, str) else str(hotwords)
                
                ws.send(json.dumps(config))
                chunk_size = 1920
                for i in range(0, len(pcm_data), chunk_size):
                    ws.send(pcm_data[i : i + chunk_size], opcode=websocket.ABNF.OPCODE_BINARY)
                ws.send(json.dumps({"is_speaking": False}), opcode=websocket.ABNF.OPCODE_TEXT)
                self.logger.info("音频数据发送完成")

            ws = websocket.WebSocketApp(
                self.funasr_ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                subprotocols=["binary"],
            )
            import threading

            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            ws_thread.join(timeout=10)

            if ws_thread.is_alive():
                ws.close()
                self.logger.warning("WebSocket 超时，强制关闭")

            return result_text if result_text else "", result_confidence

        except Exception as e:
            self.logger.error("音频识别失败: %s", e)
            import traceback

            self.logger.error(traceback.format_exc())
            return "", 0.0

    def save_audio_for_training(
        self,
        audio_data: bytes,
        recognized_text: str,
        confidence: float = 0.0,
        user_id: str = "anonymous",
        corrected_text: str = None,
    ) -> str:
        """保存音频与元数据，用于方言数据采集"""
        try:
            if not self.collect_dialect_data:
                self.logger.debug("已禁用方言数据采集，跳过保存")
                return ""

            audio_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            audio_filename = f"{audio_id}.wav"
            audio_path = os.path.join(self.dialect_audio_dir, audio_filename)
            os.makedirs(self.dialect_audio_dir, exist_ok=True)

            with open(audio_path, "wb") as f:
                f.write(audio_data)

            wav_io = io.BytesIO(audio_data)
            with wave.open(wav_io, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)

            metadata = {
                "audio_id": audio_id,
                "audio_file": audio_filename,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "recognized_text": recognized_text,
                "corrected_text": corrected_text if corrected_text else None,
                "confidence": confidence,
                "duration": duration,
                "is_corrected": corrected_text is not None and corrected_text != recognized_text,
            }

            with open(self.dialect_metadata_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            self.logger.info("音频数据已保存: %s, 时长=%.2fs, 置信度=%.3f", audio_id, duration, confidence)
            return audio_id

        except Exception as e:
            self.logger.error("保存音频数据失败: %s", e)
            return ""

    def apply_correction(self, audio_id: str, corrected_text: str):
        """更新元数据中的纠错记录"""
        if not self.collect_dialect_data:
            return False, "方言数据采集已关闭"
        if not os.path.exists(self.dialect_metadata_file):
            return False, "元数据文件不存在"
        try:
            records = []
            found = False
            with open(self.dialect_metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        if record.get("audio_id") == audio_id:
                            record["corrected_text"] = corrected_text
                            record["is_corrected"] = True
                            record["correction_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            found = True
                        records.append(record)
            if not found:
                return False, f"未找到音频记录: {audio_id}"
            with open(self.dialect_metadata_file, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True, "纠错信息已保存"
        except Exception as e:
            self.logger.error("纠错更新失败: %s", e)
            return False, str(e)


