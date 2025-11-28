"""
WebSocket server that accepts streaming 16 kHz mono audio and returns Smart Turn
endpoint predictions whenever a VAD-detected segment completes.

Protocol
--------
- Clients first send a config JSON text frame (e.g. {"vad_threshold": 0.5});
  server replies with a config echo.
- Clients then send binary frames containing raw little-endian int16 PCM audio
  sampled at 16 kHz, mono. Text "reset" clears VAD/segment state.
- Audio can arrive in any chunk size; the server buffers and processes it in
  VAD 专属窗口（Silero 默认 512 样本 ≈ 32 ms，FSMN 默认 200 ms）。
- Whenever VAD speech/silence flips, the server sends a JSON text frame:

    {"type": "vad", "speech": true, "probability": 0.82, "vad_threshold": 0.5, "timestamp_ms": ...}

- When the server decides a speech segment has ended (silence or max duration),
  it runs `predict_endpoint` on the collected audio (up to 8 seconds) and sends
  a JSON text frame like:

    {
      "type": "prediction",
      "prediction": 1,              # 1 = complete, 0 = incomplete
      "probability": 0.73,
      "duration_seconds": 3.12,
      "timestamp_ms": 1712345678901,
      "vad_probability": 0.12,
      "vad_speech": false,
      "vad_threshold": 0.5
    }

"""

import asyncio
import json
import math
import os
import time
import uuid
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import onnxruntime as ort
import websockets
from websockets.asyncio.server import serve

from inference import predict_endpoint  # expects 16 kHz mono float32 input
from fsmn_vad import FSMNVADIterator

# --- Audio / VAD configuration (fixed 16 kHz mono; Silero uses 512-sample chunks) ---
RATE = 16000
CHUNK = 512  # Silero VAD expects 512 samples at 16 kHz

VAD_THRESHOLD = 0.5
PREDICTION_THRESHOLD = 0.5
MIN_DURATION_SECONDS = 0.0
PRE_SPEECH_MS = 200
STOP_MS = 1000
MAX_DURATION_SECONDS = 8
DEFAULT_VAD_TYPE = "silero"  # 默认仍使用 Silero
SUPPORTED_VAD_TYPES = {"silero", "fsmn"}
FSMN_CHUNK_MS = 200
FSMN_VAD_MODEL_PATH = os.getenv("FSMN_VAD_MODEL_PATH", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
FSMN_VAD_DEVICE = os.getenv("FSMN_VAD_DEVICE", "cpu")

def fmt4(value: float) -> float:
    """Round float to 4 decimal places for consistent WS payloads."""
    return round(float(value), 4)


def format_config(cfg: dict) -> dict:
    """返回浮点数保留 4 位的小数配置副本。"""
    return {
        k: fmt4(v) if isinstance(v, (int, float)) else v
        for k, v in cfg.items()
    }

# Silero ONNX model
ONNX_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
ONNX_MODEL_PATH = "silero_vad.onnx"

# Reset VAD internal state every N seconds
MODEL_RESET_STATES_TIME = 5.0


class SileroVAD:
    """Minimal Silero VAD ONNX wrapper for 16 kHz, mono, chunk=512."""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        if 'CoreMLExecutionProvider' in available_providers:
            providers.append('CoreMLExecutionProvider')
        if 'MPSExecutionProvider' in available_providers:
            providers.append('MPSExecutionProvider')
        providers.append('CPUExecutionProvider')
        print(f"SileroVAD using providers: {providers}")

        self.session = ort.InferenceSession(
            model_path, providers=providers, sess_options=opts
        )
        self.context_size = 64  # Silero uses 64-sample context at 16 kHz
        self._state = None
        self._context = None
        self._last_reset_time = time.time()
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)  # (2, B, 128)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def maybe_reset(self):
        if (time.time() - self._last_reset_time) >= MODEL_RESET_STATES_TIME:
            self._init_states()
            self._last_reset_time = time.time()

    def prob(self, chunk_f32: np.ndarray) -> float:
        """Compute speech probability for one chunk of length 512 (float32, mono)."""
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK:
            raise ValueError(f"Expected {CHUNK} samples, got {x.shape[1]}")
        x = np.concatenate((self._context, x), axis=1)

        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64),
        }
        out, self._state = self.session.run(None, ort_inputs)

        # Update context (keep last 64 samples)
        self._context = x[:, -self.context_size:]
        self.maybe_reset()

        return float(out[0][0])


def ensure_model(path: str = ONNX_MODEL_PATH, url: str = ONNX_MODEL_URL) -> str:
    if not os.path.exists(path):
        print("Downloading Silero VAD ONNX model...")
        import urllib.request  # delayed import to keep startup light

        urllib.request.urlretrieve(url, path)
        print("ONNX model downloaded.")
    return path


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
        audio = np.concatenate(self.current_segment, dtype=np.float32) if self.current_segment else np.array([], dtype=np.float32)
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
                        "vad_threshold": fmt4(self.config.get("vad_threshold", VAD_THRESHOLD)),
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
                        "vad_threshold": fmt4(self.config.get("vad_threshold", VAD_THRESHOLD)),
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
                        "vad_threshold": fmt4(self.config.get("vad_threshold", VAD_THRESHOLD)),
                        "timestamp_ms": ts_ms,
                    }
                )

        return events, segments


class StreamingEndpointSession:
    """Connection-scoped state for streaming VAD + endpoint prediction。"""

    def __init__(self):
        self.leftover = np.array([], dtype=np.int16)
        self.config = {
            "vad_threshold": VAD_THRESHOLD,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "min_duration_seconds": MIN_DURATION_SECONDS,
            "vad_type": DEFAULT_VAD_TYPE,
        }
        self.vad_pipeline: BaseVADPipeline = self._build_vad_pipeline(self.config["vad_type"])

    def _build_vad_pipeline(self, vad_type: str) -> BaseVADPipeline:
        if vad_type == "fsmn":
            return FSMNVADPipeline(self.config)
        return SileroVADPipeline(self.config)

    def _set_vad_pipeline(self, vad_type: str):
        self.leftover = np.array([], dtype=np.int16)
        self.vad_pipeline = self._build_vad_pipeline(vad_type)

    def apply_config(self, config: dict) -> dict:
        """Apply client配置，当前支持 vad_threshold/prediction_threshold/min_duration_seconds/vad_type。"""
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")

        unknown_keys = set(config.keys()) - {
            "vad_threshold",
            "prediction_threshold",
            "min_duration_seconds",
            "vad_type",
        }
        if unknown_keys:
            raise ValueError(f"unsupported config fields: {', '.join(sorted(unknown_keys))}")

        new_config = dict(self.config)
        if "vad_threshold" in config:
            try:
                threshold = float(config["vad_threshold"])
            except (TypeError, ValueError):
                raise ValueError("vad_threshold must be a number between 0 and 1")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("vad_threshold must be between 0 and 1")
            new_config["vad_threshold"] = threshold
        if "prediction_threshold" in config:
            try:
                threshold = float(config["prediction_threshold"])
            except (TypeError, ValueError):
                raise ValueError("prediction_threshold must be a number between 0 and 1")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("prediction_threshold must be between 0 and 1")
            new_config["prediction_threshold"] = threshold
        if "min_duration_seconds" in config:
            try:
                min_dur = float(config["min_duration_seconds"])
            except (TypeError, ValueError):
                raise ValueError("min_duration_seconds must be a number >= 0")
            if min_dur < 0.0:
                raise ValueError("min_duration_seconds must be >= 0")
            if min_dur > MAX_DURATION_SECONDS:
                raise ValueError(f"min_duration_seconds must be <= {MAX_DURATION_SECONDS}")
            new_config["min_duration_seconds"] = min_dur
        if "vad_type" in config:
            vad_type = str(config["vad_type"]).strip().lower()
            if vad_type not in SUPPORTED_VAD_TYPES:
                raise ValueError(f"vad_type must be one of: {', '.join(sorted(SUPPORTED_VAD_TYPES))}")
            new_config["vad_type"] = vad_type

        switched_type = new_config.get("vad_type") != self.config.get("vad_type")
        self.config = new_config

        if switched_type:
            self._set_vad_pipeline(self.config["vad_type"])
        else:
            self.vad_pipeline.update_config(self.config)

        return dict(self.config)

    def reset(self):
        """Clear VAD buffers and segment state."""
        self.leftover = np.array([], dtype=np.int16)
        self.vad_pipeline.reset()

    def process_audio(self, int16_audio: np.ndarray) -> List[dict]:
        """
        Ingest new int16 samples, process in VAD 专属窗口, and return any
        payloads (vad变化或预测结果) that should be sent to the client.
        """
        messages: List[dict] = []

        if int16_audio.size == 0:
            return messages

        if self.leftover.size:
            int16_audio = np.concatenate([self.leftover, int16_audio])

        chunk_size = self.vad_pipeline.chunk_size
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        full_chunks = int16_audio.size // chunk_size
        self.leftover = int16_audio[full_chunks * chunk_size :]

        for i in range(full_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk_int16 = int16_audio[start:end]
            events, segments = self.vad_pipeline.process_chunk(chunk_int16)
            messages.extend(events)
            for seg in segments:
                messages.extend(self._process_segment(seg))

        return messages

    def _process_segment(self, segment: dict) -> List[dict]:
        """对单段语音执行最短时长校验与端点预测。"""
        events: List[dict] = []
        audio = segment.get("audio", np.array([], dtype=np.float32))
        if audio.size == 0:
            return events

        dur_sec = audio.size / RATE
        dur_sec_fmt = fmt4(dur_sec)
        min_dur = float(self.config.get("min_duration_seconds", MIN_DURATION_SECONDS))
        min_dur_fmt = fmt4(min_dur)

        vad_prob = fmt4(segment.get("vad_probability", 0.0))
        vad_speech = bool(segment.get("vad_speech", False))
        vad_threshold_fmt = fmt4(self.config.get("vad_threshold", VAD_THRESHOLD))

        if dur_sec < min_dur:
            events.append(
                {
                    "type": "skip",
                    "reason": "min_duration_not_met",
                    "duration_seconds": dur_sec_fmt,
                    "min_duration_seconds": min_dur_fmt,
                    "timestamp_ms": int(time.time() * 1000),
                    "vad_probability": vad_prob,
                    "vad_speech": vad_speech,
                    "vad_threshold": vad_threshold_fmt,
                    "prediction_threshold": fmt4(
                        self.config.get("prediction_threshold", PREDICTION_THRESHOLD)
                    ),
                }
            )
            return events

        t0 = time.perf_counter()
        result = predict_endpoint(audio.astype(np.float32))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prob = float(result.get("probability", 0.0))
        prob_fmt = fmt4(prob)
        pred_threshold = float(self.config.get("prediction_threshold", PREDICTION_THRESHOLD))
        pred_threshold_fmt = fmt4(pred_threshold)
        pred = 1 if prob > pred_threshold else 0

        events.append(
            {
                "type": "prediction",
                "prediction": pred,
                "probability": prob_fmt,
                "duration_seconds": dur_sec_fmt,
                "inference_ms": fmt4(latency_ms),
                "timestamp_ms": int(time.time() * 1000),
                "vad_probability": vad_prob,
                "vad_speech": vad_speech,
                "vad_threshold": vad_threshold_fmt,
                "prediction_threshold": pred_threshold_fmt,
            }
        )
        return events


async def handle_connection(websocket):
    peer = websocket.remote_address
    session_id = str(uuid.uuid4())
    print(f"[server] connection opened from {peer} session_id={session_id}")
    session = StreamingEndpointSession()
    config_received = False
    closed_logged = False
    ready_defaults = format_config(
        {
            "vad_threshold": VAD_THRESHOLD,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "min_duration_seconds": MIN_DURATION_SECONDS,
            "vad_type": DEFAULT_VAD_TYPE,
        }
    )
    await websocket.send(
        json.dumps(
            {
                "type": "ready",
                "message": 'send config JSON as first text frame (e.g. {"vad_type": "silero", "vad_threshold": 0.5, "prediction_threshold": 0.5}), then 16kHz mono int16 PCM as binary frames; text "reset" to clear state',
                "defaults": ready_defaults,
                "session_id": session_id,
            }
        )
    )

    try:
        async for message in websocket:
            if not config_received:
                if not isinstance(message, str):
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "send config JSON as first text frame before audio",
                                "session_id": session_id,
                            }
                        )
                    )
                    await websocket.close(code=1002, reason="config required before audio")
                    break
                try:
                    config = json.loads(message)
                    applied_config = session.apply_config(config if config is not None else {})
                    formatted_config = format_config(applied_config)
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "config must be a JSON object (text frame)",
                                "session_id": session_id,
                            }
                        )
                    )
                    await websocket.close(code=1002, reason="invalid config")
                    break
                except ValueError as exc:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"invalid config: {exc}",
                                "session_id": session_id,
                            }
                        )
                    )
                    await websocket.close(code=1002, reason="invalid config")
                    break

                config_received = True
                print(f"[server] config applied for {peer}: {applied_config}")
                await websocket.send(
                    json.dumps(
                        {
                            "type": "config",
                            "message": "config applied",
                            "config": formatted_config,
                            "session_id": session_id,
                        }
                    )
                )
                continue

            if isinstance(message, str):
                stripped = message.strip().lower()
                if stripped == "reset":
                    session.reset()
                    print(f"[server] state reset for {peer}")
                    await websocket.send(json.dumps({"type": "reset"}))
                    continue

                try:
                    config = json.loads(message)
                    if not isinstance(config, dict):
                        raise ValueError("config must be a JSON object")
                    applied_config = session.apply_config(config)
                    formatted_config = format_config(applied_config)
                    print(f"[server] config updated for {peer}: {applied_config}")
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "config",
                                "message": "config applied",
                                "config": formatted_config,
                                "session_id": session_id,
                            }
                        )
                    )
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "text frames must be config JSON or 'reset'",
                                "session_id": session_id,
                            }
                        )
                    )
                except ValueError as exc:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"invalid config: {exc}",
                                "session_id": session_id,
                            }
                        )
                    )
                continue

            # Binary audio frame
            if not isinstance(message, (bytes, bytearray)):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "unsupported message type",
                            "session_id": session_id,
                        }
                    )
                )
                continue

            if len(message) % 2 != 0:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "audio payload must be int16 PCM",
                            "session_id": session_id,
                        }
                    )
                )
                continue

            samples = np.frombuffer(message, dtype=np.int16)
            for payload in session.process_audio(samples):
                payload["session_id"] = session_id
                if payload.get("type") == "prediction":
                    print(
                        "[server] prediction"
                        f" pred={payload.get('prediction')}"
                        f" prob={payload.get('probability'):.4f}"
                        f" dur={payload.get('duration_seconds'):.2f}s"
                        f" infer_ms={payload.get('inference_ms'):.1f}"
                        f" from {peer}"
                    )
                await websocket.send(json.dumps(payload))

    except websockets.ConnectionClosed as exc:
        print(f"[server] connection closed from {peer} code={exc.code} reason={exc.reason}")
        closed_logged = True
    except Exception as exc:  # pragma: no cover - defensive logging for server mode
        err = {"type": "error", "message": f"server exception: {exc!r}", "session_id": session_id}
        try:
            await websocket.send(json.dumps(err))
        finally:
            raise
    finally:
        if not closed_logged:
            code = getattr(websocket, "close_code", None)
            reason = getattr(websocket, "close_reason", None)
            print(f"[server] connection closed from {peer} session_id={session_id} code={code} reason={reason}")


async def main():
    host = os.getenv("SMART_TURN_WS_HOST", "0.0.0.0")
    port = int(os.getenv("SMART_TURN_WS_PORT", "8765"))
    print(f"Starting Smart Turn WebSocket server on ws://{host}:{port}")
    async with serve(handle_connection, host, port, max_size=None):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
