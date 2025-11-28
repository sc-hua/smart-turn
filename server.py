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
  512-sample (32 ms) windows to match Silero VAD expectations.
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
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import onnxruntime as ort
import websockets
from websockets.asyncio.server import serve

from inference import predict_endpoint  # expects 16 kHz mono float32 input

# --- Audio / VAD configuration (fixed 16 kHz mono, 512-sample chunks) ---
RATE = 16000
CHUNK = 512  # Silero VAD expects 512 samples at 16 kHz

VAD_THRESHOLD = 0.5
PREDICTION_THRESHOLD = 0.5
PRE_SPEECH_MS = 200
STOP_MS = 1000
MAX_DURATION_SECONDS = 8

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


class StreamingEndpointSession:
    """Connection-scoped state for streaming VAD + endpoint prediction."""

    def __init__(self):
        chunk_ms = (CHUNK / RATE) * 1000.0
        self.pre_chunks = math.ceil(PRE_SPEECH_MS / chunk_ms)
        self.stop_chunks = math.ceil(STOP_MS / chunk_ms)
        self.max_chunks = math.ceil(MAX_DURATION_SECONDS / (CHUNK / RATE))

        self.pre_buffer: Deque[np.ndarray] = deque(maxlen=self.pre_chunks)
        self.segment: List[np.ndarray] = []
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0

        self.leftover = np.array([], dtype=np.int16)
        self.vad = SileroVAD(ensure_model())
        self.config = {
            "vad_threshold": VAD_THRESHOLD,
            "prediction_threshold": PREDICTION_THRESHOLD,
        }
        self.last_vad_speech: Optional[bool] = None

    def apply_config(self, config: dict) -> dict:
        """Apply client配置，当前支持 vad_threshold/prediction_threshold."""
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")

        unknown_keys = set(config.keys()) - {"vad_threshold", "prediction_threshold"}
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

        self.config = new_config
        return dict(self.config)

    def reset(self):
        """Clear VAD buffers and segment state."""
        self.pre_buffer.clear()
        self.segment.clear()
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0
        self.leftover = np.array([], dtype=np.int16)
        self.vad._init_states()
        self.last_vad_speech = None

    def process_audio(self, int16_audio: np.ndarray) -> List[dict]:
        """
        Ingest new int16 samples, process in CHUNK windows, and return any
        payloads (vad变化或预测结果) that should be sent to the client.
        """
        messages: List[dict] = []

        if int16_audio.size == 0:
            return messages

        # Combine with any partial samples from the previous message
        if self.leftover.size:
            int16_audio = np.concatenate([self.leftover, int16_audio])

        full_chunks = int16_audio.size // CHUNK
        self.leftover = int16_audio[full_chunks * CHUNK :]

        for i in range(full_chunks):
            start = i * CHUNK
            end = start + CHUNK
            chunk_int16 = int16_audio[start:end]
            chunk_f32 = (chunk_int16.astype(np.float32)) / 32768.0

            messages.extend(self._handle_chunk(chunk_f32))

        return messages

    def _handle_chunk(self, chunk_f32: np.ndarray) -> List[dict]:
        """Run VAD on a single chunk并返回需要下发的消息。"""
        events: List[dict] = []
        vad_prob = float(self.vad.prob(chunk_f32))
        vad_threshold = float(self.config.get("vad_threshold", VAD_THRESHOLD))
        is_speech = vad_prob > vad_threshold
        ts_ms = int(time.time() * 1000)

        # VAD状态变化时推送一次
        if self.last_vad_speech is None or is_speech != self.last_vad_speech:
            self.last_vad_speech = is_speech
            events.append(
                {
                    "type": "vad",
                    "speech": bool(is_speech),
                    "probability": vad_prob,
                    "vad_threshold": vad_threshold,
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
            return events

        # Already in a segment
        self.segment.append(chunk_f32)
        self.since_trigger_chunks += 1
        self.trailing_silence = 0 if is_speech else self.trailing_silence + 1

        segment_should_end = (
            self.trailing_silence >= self.stop_chunks
            or self.since_trigger_chunks >= self.max_chunks
        )

        if not segment_should_end:
            return events

        # Finalize segment
        audio = np.concatenate(self.segment, dtype=np.float32)
        self._reset_segment_state()

        dur_sec = audio.size / RATE
        t0 = time.perf_counter()
        result = predict_endpoint(audio)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prob = float(result.get("probability", 0.0))
        pred_threshold = float(self.config.get("prediction_threshold", PREDICTION_THRESHOLD))
        pred = 1 if prob > pred_threshold else 0

        events.append(
            {
                "type": "prediction",
                "prediction": pred,
                "probability": prob,
                "duration_seconds": float(dur_sec),
                "inference_ms": float(latency_ms),
                "timestamp_ms": int(time.time() * 1000),
                "vad_probability": vad_prob,
                "vad_speech": bool(is_speech),
                "vad_threshold": vad_threshold,
                "prediction_threshold": pred_threshold,
            }
        )
        return events

    def _reset_segment_state(self):
        self.segment.clear()
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0
        self.pre_buffer.clear()


async def handle_connection(websocket):
    peer = websocket.remote_address
    print(f"[server] connection opened from {peer}")
    session = StreamingEndpointSession()
    config_received = False
    await websocket.send(
        json.dumps(
            {
                "type": "ready",
                "message": 'send config JSON as first text frame (e.g. {"vad_threshold": 0.5, "prediction_threshold": 0.5}), then 16kHz mono int16 PCM as binary frames; text "reset" to clear state',
                "defaults": {
                    "vad_threshold": VAD_THRESHOLD,
                    "prediction_threshold": PREDICTION_THRESHOLD,
                },
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
                            }
                        )
                    )
                    await websocket.close(code=1002, reason="config required before audio")
                    break
                try:
                    config = json.loads(message)
                    applied_config = session.apply_config(config if config is not None else {})
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "config must be a JSON object (text frame)",
                            }
                        )
                    )
                    await websocket.close(code=1002, reason="invalid config")
                    break
                except ValueError as exc:
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"invalid config: {exc}"})
                    )
                    await websocket.close(code=1002, reason="invalid config")
                    break

                config_received = True
                print(f"[server] config applied for {peer}: {applied_config}")
                await websocket.send(
                    json.dumps(
                        {"type": "config", "message": "config applied", "config": applied_config}
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
                    print(f"[server] config updated for {peer}: {applied_config}")
                    await websocket.send(
                        json.dumps(
                            {"type": "config", "message": "config applied", "config": applied_config}
                        )
                    )
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "text frames must be config JSON or 'reset'",
                            }
                        )
                    )
                except ValueError as exc:
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"invalid config: {exc}"})
                    )
                continue

            # Binary audio frame
            if not isinstance(message, (bytes, bytearray)):
                await websocket.send(
                    json.dumps({"type": "error", "message": "unsupported message type"})
                )
                continue

            if len(message) % 2 != 0:
                await websocket.send(
                    json.dumps({"type": "error", "message": "audio payload must be int16 PCM"})
                )
                continue

            samples = np.frombuffer(message, dtype=np.int16)
            for payload in session.process_audio(samples):
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
    except Exception as exc:  # pragma: no cover - defensive logging for server mode
        err = {"type": "error", "message": f"server exception: {exc!r}"}
        try:
            await websocket.send(json.dumps(err))
        finally:
            raise


async def main():
    host = os.getenv("SMART_TURN_WS_HOST", "0.0.0.0")
    port = int(os.getenv("SMART_TURN_WS_PORT", "8765"))
    print(f"Starting Smart Turn WebSocket server on ws://{host}:{port}")
    async with serve(handle_connection, host, port, max_size=None):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
