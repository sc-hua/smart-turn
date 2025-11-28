"""Smart Turn 的 VAD 封装，供 server 复用。"""

import math
import os
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from .fsmn_vad import FSMNVADIterator
from .silero_vad import CHUNK, SileroVAD, ensure_model

RATE = 16000

VAD_THRESHOLD = 0.5
PRE_SPEECH_MS = 200
STOP_MS = 1000
MAX_DURATION_SECONDS = 8

DEFAULT_VAD_TYPE = "silero"  # 默认仍使用 Silero
SUPPORTED_VAD_TYPES = {"silero", "fsmn"}
FSMN_CHUNK_MS = 200
FSMN_VAD_MODEL_PATH = os.getenv(
    "FSMN_VAD_MODEL_PATH", "ckpts/speech_fsmn_vad_zh-cn-16k-common-pytorch"
)
FSMN_VAD_DEVICE = os.getenv("FSMN_VAD_DEVICE", "cpu")


def fmt4(value: float) -> float:
    """Round float to 4 decimal places for consistent WS payloads."""
    return round(float(value), 4)


class BaseVADPipeline:
    """各 VAD 实现复用的抽象，便于后续扩展。"""

    type_name = "base"

    def __init__(self, sample_rate: int, config: dict):
        self.sample_rate = sample_rate
        self.config = config

    @property
    def chunk_size(self) -> int:
        raise NotImplementedError

    def reset(self):
        pass

    def update_config(self, config: dict):
        self.config = config

    def process_chunk(self, chunk_int16: np.ndarray) -> tuple[List[dict], List[dict]]:
        """返回 (事件消息列表, 已完成语音段列表)。"""
        raise NotImplementedError


class SileroVADPipeline(BaseVADPipeline):
    """沿用原有 Silero 逻辑的封装。"""

    type_name = "silero"

    def __init__(self, config: dict):
        super().__init__(RATE, config)
        chunk_ms = (CHUNK / RATE) * 1000.0
        self.pre_chunks = math.ceil(PRE_SPEECH_MS / chunk_ms)
        self.stop_chunks = math.ceil(STOP_MS / chunk_ms)
        self.max_chunks = math.ceil(MAX_DURATION_SECONDS / (CHUNK / RATE))

        self.pre_buffer: Deque[np.ndarray] = deque(maxlen=self.pre_chunks)
        self.segment: List[np.ndarray] = []
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0
        self.last_vad_speech: Optional[bool] = None
        self.vad = SileroVAD(ensure_model())

    @property
    def chunk_size(self) -> int:
        return CHUNK

    def reset(self):
        self.pre_buffer.clear()
        self.segment.clear()
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0
        self.last_vad_speech = None
        self.vad._init_states()

    def update_config(self, config: dict):
        self.config = config
        # 更换阈值时沿用已有状态，无需重置

    def process_chunk(self, chunk_int16: np.ndarray) -> tuple[List[dict], List[dict]]:
        events: List[dict] = []
        segments: List[dict] = []
        chunk_f32 = (chunk_int16.astype(np.float32)) / 32768.0

        vad_prob = float(self.vad.prob(chunk_f32))
        vad_threshold = float(self.config.get("vad_threshold", VAD_THRESHOLD))
        is_speech = vad_prob > vad_threshold
        vad_prob_fmt = fmt4(vad_prob)
        vad_threshold_fmt = fmt4(vad_threshold)
        ts_ms = int(time.time() * 1000)

        if self.last_vad_speech is None or is_speech != self.last_vad_speech:
            self.last_vad_speech = is_speech
            events.append(
                {
                    "type": "vad",
                    "speech": bool(is_speech),
                    "probability": vad_prob_fmt,
                    "vad_threshold": vad_threshold_fmt,
                    "timestamp_ms": ts_ms,
                }
            )

        if not self.speech_active:
            self.pre_buffer.append(chunk_f32)
            if is_speech:
                self.segment = list(self.pre_buffer)
                self.segment.append(chunk_f32)
                self.speech_active = True
                self.trailing_silence = 0
                self.since_trigger_chunks = 1
            return events, segments

        self.segment.append(chunk_f32)
        self.since_trigger_chunks += 1
        self.trailing_silence = 0 if is_speech else self.trailing_silence + 1

        segment_should_end = (
            self.trailing_silence >= self.stop_chunks
            or self.since_trigger_chunks >= self.max_chunks
        )

        if not segment_should_end:
            return events, segments

        audio = np.concatenate(self.segment, dtype=np.float32)
        self._reset_segment_state()

        segments.append(
            {
                "audio": audio,
                "vad_probability": vad_prob,
                "vad_speech": bool(is_speech),
            }
        )
        return events, segments

    def _reset_segment_state(self):
        self.segment.clear()
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0
        self.pre_buffer.clear()


class FSMNVADPipeline(BaseVADPipeline):
    """基于 FSMN VAD 的分段实现，依赖 funasr。"""

    type_name = "fsmn"

    def __init__(self, config: dict):
        super().__init__(RATE, config)
        self.chunk_size_samples = max(int(self.sample_rate * FSMN_CHUNK_MS / 1000), 1)
        self.iterator = FSMNVADIterator(
            model_path=FSMN_VAD_MODEL_PATH,
            chunk_size_ms=FSMN_CHUNK_MS,
            sample_rate=self.sample_rate,
            device=FSMN_VAD_DEVICE,
            max_preroll_ms=PRE_SPEECH_MS,
        )
        self.current_segment: List[np.ndarray] = []
        self.segment_active = False
        self.max_samples = int(MAX_DURATION_SECONDS * self.sample_rate)

    @property
    def chunk_size(self) -> int:
        return self.chunk_size_samples

    def reset(self):
        self.iterator.reset()
        self.current_segment = []
        self.segment_active = False

    def update_config(self, config: dict):
        self.config = config

    def _current_segment_len(self) -> int:
        return sum(chunk.size for chunk in self.current_segment)

    def _finalize_segment(self, vad_prob: float, vad_speech: bool) -> dict:
        audio = (
            np.concatenate(self.current_segment, dtype=np.float32)
            if self.current_segment
            else np.array([], dtype=np.float32)
        )
        self.current_segment = []
        self.segment_active = False
        return {
            "audio": audio,
            "vad_probability": vad_prob,
            "vad_speech": vad_speech,
        }

    def process_chunk(self, chunk_int16: np.ndarray) -> tuple[List[dict], List[dict]]:
        events: List[dict] = []
        segments: List[dict] = []
        chunk_f32 = (chunk_int16.astype(np.float32)) / 32768.0

        for speech_dict, speech_samples in self.iterator(chunk_f32, is_final=False):
            ts_ms = int(time.time() * 1000)
            if "start" in speech_dict:
                self.current_segment = []
                self.segment_active = True
                events.append(
                    {
                        "type": "vad",
                        "speech": True,
                        "probability": 1.0,
                        "vad_threshold": fmt4(
                            self.config.get("vad_threshold", VAD_THRESHOLD)
                        ),
                        "timestamp_ms": ts_ms,
                    }
                )

            if self.segment_active:
                self.current_segment.append(speech_samples.astype(np.float32))

            # 强制截断，防止极端情况下长时间不结束
            if self.segment_active and self._current_segment_len() >= self.max_samples:
                segments.append(self._finalize_segment(vad_prob=1.0, vad_speech=True))
                events.append(
                    {
                        "type": "vad",
                        "speech": False,
                        "probability": 0.0,
                        "vad_threshold": fmt4(
                            self.config.get("vad_threshold", VAD_THRESHOLD)
                        ),
                        "timestamp_ms": ts_ms,
                    }
                )
                self.iterator.reset()
                continue

            if "end" in speech_dict and self.segment_active:
                segments.append(self._finalize_segment(vad_prob=1.0, vad_speech=False))
                events.append(
                    {
                        "type": "vad",
                        "speech": False,
                        "probability": 0.0,
                        "vad_threshold": fmt4(
                            self.config.get("vad_threshold", VAD_THRESHOLD)
                        ),
                        "timestamp_ms": ts_ms,
                    }
                )

        return events, segments
