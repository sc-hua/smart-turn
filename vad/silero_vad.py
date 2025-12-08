"""Silero VAD ONNX 封装，独立成文件便于复用与测试。"""

import os
import time

import numpy as np
import onnxruntime as ort

# Silero 默认 chunk 大小
CHUNK = 512  # 512 samples @ 16 kHz ~= 32 ms

# Silero ONNX model
ONNX_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
ONNX_MODEL_PATH = "onnx_model/silero_vad.onnx"

# Reset VAD internal state every N seconds
MODEL_RESET_STATES_TIME = 5.0


def ensure_model(path: str = ONNX_MODEL_PATH, url: str = ONNX_MODEL_URL) -> str:
    if not os.path.exists(path):
        print("Downloading Silero VAD ONNX model...")
        import urllib.request  # delayed import to keep startup light

        urllib.request.urlretrieve(url, path)
        print("ONNX model downloaded.")
    return path


class SileroVAD:
    """Minimal Silero VAD ONNX wrapper for 16 kHz, mono, chunk=512."""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        available_providers = ort.get_available_providers()
        requested = os.getenv("SILERO_VAD_PROVIDERS")
        if requested:
            order = [name.strip() for name in requested.split(",") if name.strip()]
        else:
            # 默认使用 CPU，benchmark 表明 Silero 这类小模型在 GPU 上反而更慢。
            order = [
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "MPSExecutionProvider",
                "CoreMLExecutionProvider",
            ]

        providers = [p for p in order if p in available_providers]
        if not providers:
            providers = ["CPUExecutionProvider"]

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
        self._context = x[:, -self.context_size :]
        self.maybe_reset()

        return float(out[0][0])
