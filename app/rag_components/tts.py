"""
语音合成（TTS）模块：将文本转换为语音。
"""

import io
import logging
import re
from typing import List

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline

logger = logging.getLogger(__name__)

class TextToSpeech:
    """封装 TTS 模型加载与语音合成"""

    def __init__(self, model_path: str, config_path: str, voice_path: str, repo_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voice_tensor = torch.load(voice_path, weights_only=True)
        
        self.model = KModel(model=model_path, config=config_path, repo_id=repo_id).to(self.device).eval()
        logger.info("TTS 模型已加载到 %s", self.device)
        
        # 如果使用CUDA，可以考虑启用半精度（FP16）加速
        # 注意：当前禁用，因为可能与 voice_tensor 不兼容
        # self.use_fp16 = False
        # if self.device == "cuda":
        #     try:
        #         self.model = self.model.half()
        #         self.use_fp16 = True
        #         logger.info("TTS 模型已切换到 FP16 模式")
        #     except Exception as e:
        #         logger.warning("无法切换到 FP16 模式: %s", e)

        self.en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)

        def en_callable(text):
            return next(self.en_pipeline(text)).phonemes

        self.tts_pipeline = KPipeline(
            lang_code="z", repo_id=repo_id, model=self.model, en_callable=en_callable
        )
        self.speed_callable = self._build_speed_callable()

    def _build_speed_callable(self):
        def speed_callable(len_ps):
            speed = 0.8
            if len_ps <= 83:
                speed = 1
            elif len_ps < 183:
                speed = 1 - (len_ps - 83) / 500
            return speed * 1.1
        return speed_callable

    def _split_text_for_tts(self, text: str, max_length: int = 150) -> List[str]:
        """将文本按句子分割，确保每段不超过最大长度"""
        original_text = text
        segments = []
        pos = 0
        while pos < len(original_text):
            end_pos = min(pos + max_length, len(original_text))
            chunk = original_text[pos:end_pos].strip()
            if chunk:
                segments.append(chunk)
            pos = end_pos
        
        # 简单验证
        original_clean = re.sub(r'\s+', '', original_text)
        segments_clean = ''.join(re.sub(r'\s+', '', s) for s in segments)
        if segments_clean != original_clean:
            logger.warning("TTS 文本分割可能丢失字符，使用强制分割")
            segments = [original_text[i:i + max_length] for i in range(0, len(original_text), max_length)]

        return [s for s in segments if s]

    def synthesize(self, text: str) -> bytes | None:
        """合成语音并返回 WAV 字节流"""
        try:
            # 移除 markdown 格式标记
            clean_text = text.replace("**", "").replace("*", "").strip()
            
            # 将换行符替换为合适的标点
            clean_text = re.sub(r'([。！？])\s*\n+', r'\1', clean_text)
            clean_text = re.sub(r'：\s*\n+', '：', clean_text)
            clean_text = re.sub(r'\n+', '，', clean_text)
            
            # 清理多余的标点和空格
            clean_text = re.sub(r'，+', '，', clean_text)
            clean_text = re.sub(r' +', ' ', clean_text)
            clean_text = clean_text.strip(" ，")

            if not clean_text:
                logger.warning("清理后的文本为空，无法生成语音")
                return None

            segments = self._split_text_for_tts(clean_text, max_length=150)
            if len(segments) > 1:
                logger.info("文本已分为 %d 段进行 TTS (总长度: %d)", len(segments), len(clean_text))

            audio_segments = []
            sample_rate = 24000

            with torch.inference_mode():
                for i, segment in enumerate(segments):
                    try:
                        generator = self.tts_pipeline(segment, voice=self.voice_tensor, speed=self.speed_callable)
                        segment_audio_parts = []
                        for result in generator:
                            wav = result.audio
                            if torch.is_tensor(wav):
                                wav = wav.cpu().numpy()
                            if wav.ndim > 1:
                                wav = wav.flatten()
                            wav = wav.astype(np.float32)
                            segment_audio_parts.append(wav)
                        
                        if segment_audio_parts:
                            audio_segments.append(np.concatenate(segment_audio_parts))
                        else:
                            logger.warning("TTS 第 %d 段未生成音频", i + 1)
                    except Exception as e:
                        logger.error("生成第 %d 段语音失败: %s", i + 1, e)
                        continue
            
            if not audio_segments:
                logger.error("所有 TTS 音频段生成失败")
                return None

            combined_audio = np.concatenate(audio_segments)
            wav_io = io.BytesIO()
            sf.write(wav_io, combined_audio, sample_rate, format="WAV")
            return wav_io.getvalue()

        except Exception as e:
            logger.exception("TTS 生成失败: %s", e)
            return None

