"""
WebSocket server that accepts streaming 16 kHz mono audio and returns Smart Turn
endpoint predictions whenever a VAD-detected segment completes.

Protocol
--------
- Clients send binary frames containing raw little-endian int16 PCM audio
  sampled at 16 kHz, mono.
- Audio can arrive in any chunk size; the server buffers and processes it in
  512-sample (32 ms) windows to match Silero VAD expectations.
- When the server decides a speech segment has ended (silence or max duration),
  it runs `predict_endpoint` on the collected audio (up to 8 seconds) and sends
  a JSON text frame like:

    {
      "type": "prediction",
      "prediction": 1,              # 1 = complete, 0 = incomplete
      "probability": 0.73,
      "duration_seconds": 3.12,
      "timestamp_ms": 1712345678901
    }

- Optional text message `reset` clears VAD and segment state for the connection.
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

    def reset(self):
        """Clear VAD buffers and segment state."""
        self.pre_buffer.clear()
        self.segment.clear()
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0
        self.leftover = np.array([], dtype=np.int16)
        self.vad._init_states()

    def process_audio(self, int16_audio: np.ndarray) -> List[dict]:
        """
        Ingest new int16 samples, process in CHUNK windows, and return any
        prediction payloads that should be sent to the client.
        """
        predictions: List[dict] = []

        if int16_audio.size == 0:
            return predictions

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

            prediction = self._handle_chunk(chunk_f32)
            if prediction is not None:
                predictions.append(prediction)

        return predictions

    def _handle_chunk(self, chunk_f32: np.ndarray) -> Optional[dict]:
        """Run VAD on a single chunk and, if a segment ends, return prediction dict."""
        is_speech = self.vad.prob(chunk_f32) > VAD_THRESHOLD

        if not self.speech_active:
            self.pre_buffer.append(chunk_f32)
            if is_speech:
                self.segment = list(self.pre_buffer)
                self.segment.append(chunk_f32)
                self.speech_active = True
                self.trailing_silence = 0
                self.since_trigger_chunks = 1
            return None

        # Already in a segment
        self.segment.append(chunk_f32)
        self.since_trigger_chunks += 1
        self.trailing_silence = 0 if is_speech else self.trailing_silence + 1

        segment_should_end = (
            self.trailing_silence >= self.stop_chunks
            or self.since_trigger_chunks >= self.max_chunks
        )

        if not segment_should_end:
            return None

        # Finalize segment
        audio = np.concatenate(self.segment, dtype=np.float32)
        self._reset_segment_state()

        dur_sec = audio.size / RATE
        t0 = time.perf_counter()
        result = predict_endpoint(audio)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "type": "prediction",
            "prediction": int(result.get("prediction", 0)),
            "probability": float(result.get("probability", 0.0)),
            "duration_seconds": float(dur_sec),
            "inference_ms": float(latency_ms),
            "timestamp_ms": int(time.time() * 1000),
        }

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
    await websocket.send(
        json.dumps(
            {
                "type": "ready",
                "message": "send 16kHz mono int16 PCM as binary frames; text 'reset' to clear state",
            }
        )
    )

    try:
        async for message in websocket:
            if isinstance(message, str):
                if message.strip().lower() == "reset":
                    session.reset()
                    print(f"[server] state reset for {peer}")
                    await websocket.send(json.dumps({"type": "reset"}))
                else:
                    await websocket.send(
                        json.dumps({"type": "error", "message": "send audio as binary frames"})
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
